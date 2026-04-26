import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from common import (
    bootstrap_accuracy_ci,
    build_chat_prompt,
    ensure_truthfulqa_csv,
    get_decoder_layers,
    get_primary_device,
    load_jsonl,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    make_binary_instance,
    parse_int_list,
    save_json,
    sequence_logprob,
    split_calibration_eval,
    stable_hash,
)


LETTER_ORDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class TaskResult:
    name: str
    metric: str
    base_score: float
    patched_score: float
    delta: float
    n: int
    extra: dict


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal patch regression suite.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model id or local path")
    parser.add_argument("--directions", required=True, help="Path to directions npz")
    parser.add_argument("--truthfulqa-csv", default="experiments/data/TruthfulQA.csv", help="Path to TruthfulQA CSV")
    parser.add_argument("--drift-jsonl", default="experiments/data/prepared/drift_benign.jsonl", help="Benign drift prompts")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB")
    parser.add_argument("--seed", type=int, default=41, help="Evaluation seed")
    parser.add_argument("--calibration-size", type=int, default=200, help="TruthfulQA calibration split size")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--alpha", type=float, required=True, help="Rank-one orthogonalization strength")
    parser.add_argument("--modules", default="mlp", choices=["attn", "mlp", "both"], help="Which modules to patch")
    parser.add_argument(
        "--candidate-prefix",
        default="newline",
        choices=["space", "newline", "none"],
        help="Prefix style for letter-choice scoring",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="For hybrid reasoning models such as Qwen3, force non-thinking chat template mode.",
    )
    parser.add_argument("--truthfulqa-max-samples", type=int, default=0, help="Optional TruthfulQA cap before split")
    parser.add_argument("--mmlu-samples", type=int, default=100, help="Validation examples for MMLU")
    parser.add_argument("--hellaswag-samples", type=int, default=100, help="Validation examples for HellaSwag")
    parser.add_argument("--gsm8k-samples", type=int, default=100, help="Test examples for GSM8K")
    parser.add_argument("--drift-samples", type=int, default=40, help="Benign prompts for drift check")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap rounds for CI on accuracy tasks")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation budget for GSM8K/drift")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/analysis/patch_regression_suite.json",
        help="Where to save suite results",
    )
    return parser.parse_args()


def choice_candidates(prefix: str, n_choices: int):
    if n_choices > len(LETTER_ORDER):
        raise ValueError(f"Too many choices: {n_choices}")
    if prefix == "space":
        return [f" {LETTER_ORDER[idx]}" for idx in range(n_choices)]
    if prefix == "newline":
        return [f"\n{LETTER_ORDER[idx]}" for idx in range(n_choices)]
    if prefix == "none":
        return [LETTER_ORDER[idx] for idx in range(n_choices)]
    raise ValueError(f"Unsupported candidate prefix: {prefix}")


def summarize_accuracy(rows, bootstrap_rounds: int, seed: int):
    y_true = [int(row["correct_idx"]) for row in rows]
    y_pred = [int(row["pred_idx"]) for row in rows]
    acc, lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=bootstrap_rounds, seed=seed)
    return {"accuracy": float(acc), "ci95": [float(lo), float(hi)], "n": len(rows)}


def simple_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def generate_text(model, tokenizer, device: torch.device, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def extract_last_number(text: str) -> str:
    matches = NUMBER_RE.findall(text or "")
    if not matches:
        return ""
    value = matches[-1].replace(",", "")
    if value.startswith("+"):
        value = value[1:]
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


@torch.no_grad()
def apply_rank_one_patch(weight: torch.Tensor, v_hat: torch.Tensor, alpha: float):
    v = v_hat.to(weight.device, dtype=weight.dtype)
    proj_row = torch.matmul(v.unsqueeze(0), weight)
    delta = alpha * torch.matmul(v.unsqueeze(1), proj_row)
    weight.sub_(delta)


def apply_patch_to_model(model, directions: np.ndarray, selected_layers, modules: str, alpha: float, device: torch.device):
    layers = get_decoder_layers(model)
    patch_log = []
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        v = directions[layer_idx]
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            patch_log.append({"layer": layer_idx, "patched_modules": [], "skipped": True, "reason": "near-zero direction norm"})
            continue
        v_hat = torch.tensor(v / norm, dtype=torch.float32, device=device)

        patched_names = []
        mode = modules.lower()
        if mode in {"attn", "both"}:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                apply_rank_one_patch(layer.self_attn.o_proj.weight.data, v_hat, alpha)
                patched_names.append("attn.o_proj")
            elif hasattr(layer, "attn") and hasattr(layer.attn, "out_proj"):
                apply_rank_one_patch(layer.attn.out_proj.weight.data, v_hat, alpha)
                patched_names.append("attn.out_proj")
        if mode in {"mlp", "both"}:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                apply_rank_one_patch(layer.mlp.down_proj.weight.data, v_hat, alpha)
                patched_names.append("mlp.down_proj")
            elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
                apply_rank_one_patch(layer.mlp.c_proj.weight.data, v_hat, alpha)
                patched_names.append("mlp.c_proj")
        patch_log.append({"layer": layer_idx, "patched_modules": patched_names, "skipped": len(patched_names) == 0})
    return patch_log


def sample_rows(rows, n_samples: int, seed: int):
    if n_samples <= 0 or len(rows) <= n_samples:
        return list(rows)
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:n_samples]


def stratified_sample_by_key(rows, key: str, n_samples: int, seed: int):
    if n_samples <= 0 or len(rows) <= n_samples:
        return list(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    ordered_keys = sorted(groups)
    picked = []
    while len(picked) < n_samples:
        progress = False
        for group_key in ordered_keys:
            values = groups[group_key]
            if values:
                picked.append(values.pop())
                progress = True
                if len(picked) >= n_samples:
                    break
        if not progress:
            break
    return picked


def eval_truthfulqa_binary(model, tokenizer, device, args):
    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    if args.truthfulqa_max_samples > 0:
        items = items[: args.truthfulqa_max_samples]
    _, eval_items = split_calibration_eval(items, args.calibration_size, args.seed)
    if not eval_items:
        eval_items = items

    candidates = choice_candidates(args.candidate_prefix, 2)
    rows = []
    for item in tqdm(eval_items, desc="TruthfulQA"):
        row_rng = random.Random(args.seed + stable_hash(item.question))
        prompt, correct, a_text, b_text = make_binary_instance(
            item,
            row_rng,
            tokenizer,
            enable_thinking=False if args.disable_thinking else None,
        )
        scores = [
            sequence_logprob(model, tokenizer, prompt, cont, device, add_leading_space=False)
            for cont in candidates
        ]
        pred_idx = int(np.argmax(scores))
        correct_idx = 0 if correct == "A" else 1
        margin_ab = float(scores[0] - scores[1])
        margin_correct = margin_ab if correct_idx == 0 else -margin_ab
        rows.append(
            {
                "question": item.question,
                "category": item.category,
                "choices": [a_text, b_text],
                "correct_idx": correct_idx,
                "pred_idx": pred_idx,
                "scores": [float(s) for s in scores],
                "margin_correct": margin_correct,
            }
        )
    summary = summarize_accuracy(rows, args.bootstrap, args.seed)
    return rows, summary


def build_mc_prompt(system_message: str, intro: str, question: str, choices):
    lines = [intro, f"Question: {question}"]
    for idx, choice in enumerate(choices):
        lines.append(f"{LETTER_ORDER[idx]}. {choice}")
    lines.append(f"Answer with only one letter: {', '.join(LETTER_ORDER[:len(choices)])}.")
    return system_message, "\n".join(lines)


def eval_mmlu(model, tokenizer, device, args):
    dataset = load_dataset("cais/mmlu", "all", split="validation")
    rows_in = stratified_sample_by_key(list(dataset), "subject", args.mmlu_samples, args.seed)
    candidates = choice_candidates(args.candidate_prefix, 4)
    rows = []
    for row in tqdm(rows_in, desc="MMLU"):
        subject = row["subject"].replace("_", " ")
        system_message, user_message = build_mc_prompt(
            system_message="You are a careful and knowledgeable assistant.",
            intro=f"Choose the correct answer for the following {subject} multiple-choice question.",
            question=row["question"],
            choices=row["choices"],
        )
        if args.disable_thinking:
            prompt = build_chat_prompt(
                tokenizer,
                system_message=system_message,
                user_message=user_message,
                enable_thinking=False,
            )
        else:
            prompt = build_chat_prompt(tokenizer, system_message=system_message, user_message=user_message)
        scores = [
            sequence_logprob(model, tokenizer, prompt, cont, device, add_leading_space=False)
            for cont in candidates
        ]
        rows.append(
            {
                "subject": row["subject"],
                "question": row["question"],
                "correct_idx": int(row["answer"]),
                "pred_idx": int(np.argmax(scores)),
                "scores": [float(s) for s in scores],
            }
        )
    summary = summarize_accuracy(rows, args.bootstrap, args.seed)
    return rows, summary


def eval_hellaswag(model, tokenizer, device, args):
    dataset = load_dataset("hellaswag", split="validation")
    rows_in = sample_rows(list(dataset), args.hellaswag_samples, args.seed)
    candidates = choice_candidates(args.candidate_prefix, 4)
    rows = []
    for row in tqdm(rows_in, desc="HellaSwag"):
        system_message, user_message = build_mc_prompt(
            system_message="You are a careful assistant focused on commonsense reasoning.",
            intro="Choose the most plausible continuation.",
            question=row["ctx"],
            choices=row["endings"],
        )
        if args.disable_thinking:
            prompt = build_chat_prompt(
                tokenizer,
                system_message=system_message,
                user_message=user_message,
                enable_thinking=False,
            )
        else:
            prompt = build_chat_prompt(tokenizer, system_message=system_message, user_message=user_message)
        scores = [
            sequence_logprob(model, tokenizer, prompt, cont, device, add_leading_space=False)
            for cont in candidates
        ]
        rows.append(
            {
                "question": row["ctx"],
                "correct_idx": int(row["label"]),
                "pred_idx": int(np.argmax(scores)),
                "scores": [float(s) for s in scores],
            }
        )
    summary = summarize_accuracy(rows, args.bootstrap, args.seed)
    return rows, summary


def eval_gsm8k(model, tokenizer, device, args):
    dataset = load_dataset("gsm8k", "main", split="test")
    rows_in = sample_rows(list(dataset), args.gsm8k_samples, args.seed)
    rows = []
    for row in tqdm(rows_in, desc="GSM8K"):
        prompt_kwargs = {}
        if args.disable_thinking:
            prompt_kwargs["enable_thinking"] = False
        prompt = build_chat_prompt(
            tokenizer,
            system_message="You are a careful math assistant.",
            user_message=(
                "Solve the following word problem. Think silently if needed, but output only the final numeric answer.\n\n"
                f"Question: {row['question']}"
            ),
            **prompt_kwargs,
        )
        text = generate_text(model, tokenizer, device, prompt, args.max_new_tokens)
        pred = extract_last_number(text)
        gold = extract_last_number(row["answer"].split("####")[-1])
        rows.append(
            {
                "question": row["question"],
                "gold": gold,
                "pred": pred,
                "correct_idx": 1 if gold else 0,
                "pred_idx": 1 if pred == gold and gold else 0,
                "raw_output": text,
            }
        )
    summary = summarize_accuracy(rows, args.bootstrap, args.seed)
    return rows, summary


def eval_drift(model, tokenizer, device, args):
    rows_in = load_jsonl(Path(args.drift_jsonl))
    if args.drift_samples > 0:
        rows_in = rows_in[: args.drift_samples]
    rows = []
    for row in tqdm(rows_in, desc="Drift"):
        prompt_kwargs = {}
        if args.disable_thinking:
            prompt_kwargs["enable_thinking"] = False
        prompt = build_chat_prompt(
            tokenizer,
            system_message="You are a helpful and concise assistant.",
            user_message=row["prompt_text"],
            **prompt_kwargs,
        )
        text = generate_text(model, tokenizer, device, prompt, args.max_new_tokens)
        rows.append(
            {
                "prompt_id": row.get("prompt_id"),
                "prompt_text": row["prompt_text"],
                "output": text,
                "token_count": simple_token_count(tokenizer, text),
            }
        )
    return rows


def compare_drift(base_rows, patched_rows):
    if len(base_rows) != len(patched_rows):
        raise ValueError("Drift row mismatch.")
    ratios = []
    exact = 0
    token_deltas = []
    refusals = {"base": 0, "patched": 0}
    refusal_markers = ("i can't", "i cannot", "sorry", "unable to")
    rows = []
    for base, patched in zip(base_rows, patched_rows):
        ratio = SequenceMatcher(None, base["output"], patched["output"]).ratio()
        ratios.append(ratio)
        exact += int(base["output"] == patched["output"])
        token_delta = patched["token_count"] - base["token_count"]
        token_deltas.append(token_delta)
        base_refusal = int(base["output"].lower().startswith(refusal_markers))
        patched_refusal = int(patched["output"].lower().startswith(refusal_markers))
        refusals["base"] += base_refusal
        refusals["patched"] += patched_refusal
        rows.append(
            {
                "prompt_id": base["prompt_id"],
                "prompt_text": base["prompt_text"],
                "base_output": base["output"],
                "patched_output": patched["output"],
                "similarity_ratio": float(ratio),
                "token_count_delta": int(token_delta),
                "base_refusal": bool(base_refusal),
                "patched_refusal": bool(patched_refusal),
            }
        )
    summary = {
        "n": len(rows),
        "mean_similarity_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "median_similarity_ratio": float(np.median(ratios)) if ratios else 0.0,
        "min_similarity_ratio": float(np.min(ratios)) if ratios else 0.0,
        "exact_match_rate": float(exact / len(rows)) if rows else 0.0,
        "mean_token_count_delta": float(np.mean(token_deltas)) if token_deltas else 0.0,
        "median_token_count_delta": float(np.median(token_deltas)) if token_deltas else 0.0,
        "max_abs_token_count_delta": float(np.max(np.abs(token_deltas))) if token_deltas else 0.0,
        "base_refusal_rate": float(refusals["base"] / len(rows)) if rows else 0.0,
        "patched_refusal_rate": float(refusals["patched"] / len(rows)) if rows else 0.0,
    }
    rows.sort(key=lambda row: row["similarity_ratio"])
    return rows, summary


def task_result(name: str, metric: str, base_summary: dict, patched_summary: dict, extra: dict | None = None):
    extra = extra or {}
    base_score = float(base_summary[metric])
    patched_score = float(patched_summary[metric])
    return TaskResult(
        name=name,
        metric=metric,
        base_score=base_score,
        patched_score=patched_score,
        delta=patched_score - base_score,
        n=int(base_summary["n"]),
        extra=extra,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    direction_data = np.load(args.directions)
    directions = direction_data["directions"]
    selected_layers = parse_int_list(args.layers)
    if not selected_layers:
        raise ValueError("No layers selected.")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)
    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}")

    base_truthful_rows, base_truthful = eval_truthfulqa_binary(model, tokenizer, device, args)
    base_mmlu_rows, base_mmlu = eval_mmlu(model, tokenizer, device, args)
    base_hella_rows, base_hella = eval_hellaswag(model, tokenizer, device, args)
    base_gsm_rows, base_gsm = eval_gsm8k(model, tokenizer, device, args)
    base_drift_rows = eval_drift(model, tokenizer, device, args)

    patch_log = apply_patch_to_model(
        model=model,
        directions=directions,
        selected_layers=selected_layers,
        modules=args.modules,
        alpha=args.alpha,
        device=device,
    )

    patched_truthful_rows, patched_truthful = eval_truthfulqa_binary(model, tokenizer, device, args)
    patched_mmlu_rows, patched_mmlu = eval_mmlu(model, tokenizer, device, args)
    patched_hella_rows, patched_hella = eval_hellaswag(model, tokenizer, device, args)
    patched_gsm_rows, patched_gsm = eval_gsm8k(model, tokenizer, device, args)
    patched_drift_rows = eval_drift(model, tokenizer, device, args)
    drift_rows, drift_summary = compare_drift(base_drift_rows, patched_drift_rows)

    task_results = [
        task_result("truthfulqa_binary", "accuracy", base_truthful, patched_truthful),
        task_result("mmlu_zero_shot_letter", "accuracy", base_mmlu, patched_mmlu),
        task_result("hellaswag_zero_shot_letter", "accuracy", base_hella, patched_hella),
        task_result("gsm8k_final_number", "accuracy", base_gsm, patched_gsm),
        TaskResult(
            name="benign_drift",
            metric="mean_similarity_ratio",
            base_score=1.0,
            patched_score=float(drift_summary["mean_similarity_ratio"]),
            delta=float(drift_summary["mean_similarity_ratio"] - 1.0),
            n=int(drift_summary["n"]),
            extra={
                "exact_match_rate": float(drift_summary["exact_match_rate"]),
                "mean_token_count_delta": float(drift_summary["mean_token_count_delta"]),
                "patched_refusal_rate": float(drift_summary["patched_refusal_rate"]),
            },
        ),
    ]

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "seed": args.seed,
        "patch": {
            "directions": args.directions,
            "layers": selected_layers,
            "alpha": args.alpha,
            "modules": args.modules,
            "candidate_prefix": args.candidate_prefix,
            "disable_thinking": bool(args.disable_thinking),
        },
        "patch_log": patch_log,
        "tasks": {
            "truthfulqa_binary": {"base": base_truthful, "patched": patched_truthful},
            "mmlu_zero_shot_letter": {"base": base_mmlu, "patched": patched_mmlu},
            "hellaswag_zero_shot_letter": {"base": base_hella, "patched": patched_hella},
            "gsm8k_final_number": {"base": base_gsm, "patched": patched_gsm},
            "benign_drift": drift_summary,
        },
        "task_table": [
            {
                "name": row.name,
                "metric": row.metric,
                "base": row.base_score,
                "patched": row.patched_score,
                "delta": row.delta,
                "n": row.n,
                "extra": row.extra,
            }
            for row in task_results
        ],
        "artifacts": {
            "base_truthfulqa_rows": base_truthful_rows,
            "patched_truthfulqa_rows": patched_truthful_rows,
            "base_mmlu_rows": base_mmlu_rows,
            "patched_mmlu_rows": patched_mmlu_rows,
            "base_hellaswag_rows": base_hella_rows,
            "patched_hellaswag_rows": patched_hella_rows,
            "base_gsm8k_rows": base_gsm_rows,
            "patched_gsm8k_rows": patched_gsm_rows,
            "drift_rows": drift_rows,
        },
    }

    save_json(Path(args.output_json), result)
    print(f"Saved suite results to: {args.output_json}")
    for row in result["task_table"]:
        base = row["base"]
        patched = row["patched"]
        delta = row["delta"]
        if row["name"] == "benign_drift":
            print(f"{row['name']}: similarity={patched:.4f} exact={row['extra']['exact_match_rate']:.4f} token_delta={row['extra']['mean_token_count_delta']:.3f}")
        else:
            print(f"{row['name']}: base={100.0*base:.2f}% patched={100.0*patched:.2f}% delta={100.0*delta:+.2f} pts")


if __name__ == "__main__":
    main()
