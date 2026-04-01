import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_open_answer_prompt,
    get_decoder_layers,
    get_layer_write_modules,
    get_primary_device,
    load_jsonl,
    load_model_and_tokenizer,
    save_json,
)
from truthfulqa_open_generation_eval import classify_bucket, summarize_bucket_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Open-generation weight patch evaluation")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--questions-jsonl",
        default="experiments/data/mechanistic_open/annotation_seed.jsonl",
        help="Prepared question JSONL",
    )
    parser.add_argument(
        "--directions-npz",
        required=True,
        help="NPZ with layerwise direction arrays or bases",
    )
    parser.add_argument(
        "--direction-key",
        default="direct_minus_non_direct",
        help="Array key inside the NPZ",
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer list")
    parser.add_argument("--alpha", type=float, default=0.2, help="Patch strength")
    parser.add_argument(
        "--modules",
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which write modules to patch",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu-memory-gb",
        type=int,
        default=15,
        help="Per-GPU memory cap in GiB for model loading",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Generation budget per prompt",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional row cap")
    parser.add_argument("--seed", type=int, default=7, help="Seed for random control basis")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/mechanistic_open_weight_patch_eval.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def parse_layers(raw: str):
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def normalize_basis(layer_obj: np.ndarray) -> np.ndarray:
    if layer_obj.ndim == 1:
        norm = float(np.linalg.norm(layer_obj))
        if norm <= 1e-12:
            return np.zeros((0, layer_obj.shape[0]), dtype=np.float32)
        return (layer_obj[None, :] / norm).astype(np.float32)

    rows = []
    for row in layer_obj:
        norm = float(np.linalg.norm(row))
        if norm <= 1e-12:
            continue
        rows.append((row / norm).astype(np.float32))
    if not rows:
        return np.zeros((0, layer_obj.shape[-1]), dtype=np.float32)
    mat = np.stack(rows, axis=0)
    q, _ = np.linalg.qr(mat.T)
    return q.T.astype(np.float32)


def make_random_basis(direction_matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(direction_matrix.shape).astype(np.float32)


@torch.no_grad()
def apply_subspace_patch(weight: torch.Tensor, basis: torch.Tensor, alpha: float):
    if weight.ndim != 2:
        raise ValueError("Expected a 2D weight matrix for patching.")
    local_basis = basis.to(device=weight.device, dtype=weight.dtype)
    if local_basis.ndim == 1:
        local_basis = local_basis.unsqueeze(0)
    proj_rows = torch.matmul(local_basis, weight)
    delta = torch.matmul(local_basis.transpose(0, 1), proj_rows)
    weight.sub_(alpha * delta)


def generate_answer(model, tokenizer, device, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
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


def summarize_condition(rows):
    bucket_summary = summarize_bucket_rows(rows)
    n = max(1, len(rows))
    hard_bad = sum(
        1 for row in rows if row["bucket"] in {"contradicted_reference", "mixed_or_self_contradictory"}
    )
    unresolved = sum(1 for row in rows if row["bucket"] == "unresolved_needs_annotation")
    supported = sum(1 for row in rows if row["bucket"] in {"supported_answer", "supported_abstention"})
    abstention = sum(1 for row in rows if row["abstains"])
    return {
        "n": len(rows),
        "bucket_summary": bucket_summary,
        "hard_bad_rate": float(hard_bad / n),
        "unresolved_rate": float(unresolved / n),
        "supported_rate": float(supported / n),
        "abstention_rate": float(abstention / n),
    }


def run_condition(name, model, tokenizer, device, questions, max_new_tokens):
    rows = []
    for row in tqdm(questions, desc=f"Open weight eval [{name}]"):
        prompt = build_open_answer_prompt(tokenizer, row["question"])
        answer_text = generate_answer(model, tokenizer, device, prompt, max_new_tokens)
        (
            bucket,
            abstains,
            correct_match,
            incorrect_match,
            correct_score,
            incorrect_score,
            matched_correct,
            matched_incorrect,
        ) = classify_bucket(row, answer_text)
        rows.append(
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["question"],
                "answer_text": answer_text,
                "bucket": bucket,
                "abstains": abstains,
                "correct_match": correct_match,
                "incorrect_match": incorrect_match,
                "correct_score": correct_score,
                "incorrect_score": incorrect_score,
                "matched_correct": matched_correct,
                "matched_incorrect": matched_incorrect,
            }
        )
    return rows, summarize_condition(rows)


def save_original_weights(layers, layer_indices, modules_mode: str):
    originals = {}
    for layer_idx in layer_indices:
        for module_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            originals[(layer_idx, module_name)] = module.weight.data.detach().clone()
    return originals


def restore_original_weights(layers, modules_mode: str, originals):
    for (layer_idx, module_name), tensor in originals.items():
        for current_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            if current_name == module_name:
                module.weight.data.copy_(tensor)
                break


def patch_layers(layers, layer_indices, direction_matrix: np.ndarray, alpha: float, modules_mode: str, device):
    patch_log = []
    for layer_idx in layer_indices:
        basis_np = normalize_basis(direction_matrix[layer_idx])
        if basis_np.shape[0] == 0:
            patch_log.append({"layer": layer_idx, "patched_modules": [], "skipped": True})
            continue
        basis = torch.tensor(basis_np, dtype=torch.float32, device=device)
        patched_names = []
        for module_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            apply_subspace_patch(module.weight.data, basis, alpha)
            patched_names.append(module_name)
        patch_log.append(
            {
                "layer": layer_idx,
                "basis_rank": int(basis.shape[0]),
                "patched_modules": patched_names,
                "skipped": len(patched_names) == 0,
            }
        )
    return patch_log


def main():
    args = parse_args()
    layer_indices = parse_layers(args.layers)
    if not layer_indices:
        raise ValueError("No layers selected.")

    questions = load_jsonl(Path(args.questions_jsonl))
    if args.max_samples > 0:
        questions = questions[: args.max_samples]
    if not questions:
        raise ValueError("No questions loaded.")

    direction_npz = np.load(args.directions_npz)
    if args.direction_key not in direction_npz:
        raise KeyError(f"Direction key '{args.direction_key}' not found in {args.directions_npz}")
    direction_matrix = np.asarray(direction_npz[args.direction_key], dtype=np.float32)
    random_matrix = make_random_basis(direction_matrix, args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)
    originals = save_original_weights(layers, layer_indices, args.modules)

    base_rows, base_summary = run_condition(
        "base",
        model,
        tokenizer,
        device,
        questions,
        args.max_new_tokens,
    )

    target_patch_log = patch_layers(
        layers,
        layer_indices,
        direction_matrix,
        args.alpha,
        args.modules,
        device,
    )
    target_rows, target_summary = run_condition(
        "target",
        model,
        tokenizer,
        device,
        questions,
        args.max_new_tokens,
    )

    restore_original_weights(layers, args.modules, originals)
    random_patch_log = patch_layers(
        layers,
        layer_indices,
        random_matrix,
        args.alpha,
        args.modules,
        device,
    )
    random_rows, random_summary = run_condition(
        "random",
        model,
        tokenizer,
        device,
        questions,
        args.max_new_tokens,
    )
    restore_original_weights(layers, args.modules, originals)

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "questions_jsonl": args.questions_jsonl,
        "n_eval": len(questions),
        "directions_npz": args.directions_npz,
        "direction_key": args.direction_key,
        "layers": layer_indices,
        "alpha": args.alpha,
        "modules": args.modules,
        "seed": args.seed,
        "base": base_summary,
        "target": target_summary,
        "random": random_summary,
        "delta": {
            "target_hard_bad_rate": target_summary["hard_bad_rate"] - base_summary["hard_bad_rate"],
            "target_supported_rate": target_summary["supported_rate"] - base_summary["supported_rate"],
            "target_unresolved_rate": target_summary["unresolved_rate"] - base_summary["unresolved_rate"],
            "random_hard_bad_rate": random_summary["hard_bad_rate"] - base_summary["hard_bad_rate"],
            "random_supported_rate": random_summary["supported_rate"] - base_summary["supported_rate"],
            "random_unresolved_rate": random_summary["unresolved_rate"] - base_summary["unresolved_rate"],
        },
        "patch_log": {
            "target": target_patch_log,
            "random": random_patch_log,
        },
        "rows": {
            "base": base_rows,
            "target": target_rows,
            "random": random_rows,
        },
    }
    save_json(Path(args.output_json), result)
    print(json.dumps(result["delta"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
