import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from common import (
    bootstrap_accuracy_ci,
    ensure_truthfulqa_csv,
    get_decoder_layers,
    get_binary_letter_candidates,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    make_binary_instance,
    maybe_cap_items,
    parse_int_list,
    save_json,
    split_calibration_eval,
    stable_hash,
    summarize_intervention_rows,
    summarize_accuracy_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Activation-level projection-removal probe")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--truthfulqa-csv",
        default="experiments/data/TruthfulQA.csv",
        help="Path to TruthfulQA CSV",
    )
    parser.add_argument("--directions", required=True, help="Path to directions npz")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu-memory-gb",
        type=int,
        default=15,
        help="Per-GPU memory cap in GiB for model loading",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model with bitsandbytes 4-bit quantization",
    )
    parser.add_argument(
        "--candidate-prefix",
        default="space",
        choices=["space", "newline", "none"],
        help="Prefix style for A/B candidate token scoring",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="Calibration split size (eval uses held-out)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap total rows before split (0 means no cap)",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices (e.g., 20,21,22)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.35,
        help="Projection removal strength",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap rounds for CI",
    )
    parser.add_argument(
        "--diagnostic-top-k",
        type=int,
        default=20,
        help="Number of largest-margin-shift examples to keep",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/activation_probe_eval.json",
        help="Where to save metrics",
    )
    return parser.parse_args()


@dataclass
class ProbeContext:
    prompt_len: int = 0


def sequence_logprob_with_hooks(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
    context: ProbeContext,
    add_leading_space: bool = False,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    context.prompt_len = len(prompt_ids)
    cont_text = continuation
    if add_leading_space and continuation and continuation[0] not in {" ", "\n", "\t"}:
        cont_text = " " + continuation
    cont_ids = tokenizer(cont_text, add_special_tokens=False)["input_ids"]
    if len(cont_ids) == 0:
        return float("-inf")

    input_ids = torch.tensor([prompt_ids + cont_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_ids = input_ids[:, 1:]
    gathered = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    start = len(prompt_ids) - 1
    end = start + len(cont_ids)
    return float(gathered[0, start:end].sum().item())


def make_projection_hook(v: torch.Tensor, beta: float, ctx: ProbeContext):
    def _hook(_module, _inputs, output):
        if ctx.prompt_len <= 0:
            return output

        def _edit(hidden: torch.Tensor) -> torch.Tensor:
            if hidden.size(1) <= 0:
                return hidden
            idx = min(ctx.prompt_len - 1, hidden.size(1) - 1)
            edited = hidden.clone()
            token_vec = edited[:, idx, :]

            v_local = v.to(device=token_vec.device, dtype=token_vec.dtype)
            proj = (token_vec * v_local.unsqueeze(0)).sum(dim=-1, keepdim=True)
            edited[:, idx, :] = token_vec - beta * proj * v_local.unsqueeze(0)
            return edited

        if isinstance(output, tuple):
            first = _edit(output[0])
            return (first,) + output[1:]
        return _edit(output)

    return _hook


def evaluate(
    model,
    tokenizer,
    device: torch.device,
    items,
    seed: int,
    candidate_a: str,
    candidate_b: str,
    bootstrap_rounds: int,
    context: ProbeContext,
) -> Dict:
    y_true = []
    y_pred = []
    rows = []

    for item in tqdm(items, desc="Eval"):
        row_rng = random.Random(seed + stable_hash(item.question))
        prompt, correct, _, _ = make_binary_instance(item, row_rng, tokenizer)

        lp_a = sequence_logprob_with_hooks(
            model,
            tokenizer,
            prompt,
            candidate_a,
            device,
            context=context,
            add_leading_space=False,
        )
        lp_b = sequence_logprob_with_hooks(
            model,
            tokenizer,
            prompt,
            candidate_b,
            device,
            context=context,
            add_leading_space=False,
        )
        pred = "A" if lp_a >= lp_b else "B"
        margin_ab = lp_a - lp_b
        margin_correct = margin_ab if correct == "A" else -margin_ab

        y_true.append(1 if correct == "A" else 0)
        y_pred.append(1 if pred == "A" else 0)
        rows.append(
            {
                "question": item.question,
                "correct": correct,
                "pred": pred,
                "logprob_A": lp_a,
                "logprob_B": lp_b,
                "margin_ab": margin_ab,
                "margin_correct": margin_correct,
            }
        )

    context.prompt_len = 0
    acc, lo, hi = bootstrap_accuracy_ci(
        y_true,
        y_pred,
        n_bootstrap=bootstrap_rounds,
        seed=seed,
    )
    return {
        "acc": acc,
        "ci95": [lo, hi],
        "n": len(items),
        "y_true": y_true,
        "y_pred": y_pred,
        "rows": rows,
    }


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)
    _, eval_items = split_calibration_eval(items, args.calibration_size, args.seed)
    if not eval_items:
        eval_items = items

    selected_layers = parse_int_list(args.layers)
    if not selected_layers:
        raise ValueError("No layers selected.")

    direction_data = np.load(args.directions)
    directions = direction_data["directions"]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)
    cand_a, cand_b = get_binary_letter_candidates(args.candidate_prefix)
    context = ProbeContext(prompt_len=0)

    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(
                f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}"
            )

    base = evaluate(
        model,
        tokenizer,
        device,
        eval_items,
        args.seed,
        candidate_a=cand_a,
        candidate_b=cand_b,
        bootstrap_rounds=args.bootstrap,
        context=context,
    )

    hook_handles = []
    for layer_idx in selected_layers:
        v_np = directions[layer_idx]
        norm = np.linalg.norm(v_np)
        if norm <= 1e-12:
            continue
        v = torch.tensor(v_np / norm, dtype=torch.float32)
        handle = layers[layer_idx].register_forward_hook(
            make_projection_hook(v=v, beta=args.beta, ctx=context)
        )
        hook_handles.append(handle)

    intervened = evaluate(
        model,
        tokenizer,
        device,
        eval_items,
        args.seed,
        candidate_a=cand_a,
        candidate_b=cand_b,
        bootstrap_rounds=args.bootstrap,
        context=context,
    )

    for h in hook_handles:
        h.remove()

    diagnostics = summarize_intervention_rows(
        base_rows=base["rows"],
        new_rows=intervened["rows"],
        top_k=args.diagnostic_top_k,
    )

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "candidate_prefix": args.candidate_prefix,
        "seed": args.seed,
        "n_eval": len(eval_items),
        "directions": args.directions,
        "layers": selected_layers,
        "beta": args.beta,
        "base": {k: v for k, v in base.items() if k not in {"y_true", "y_pred", "rows"}},
        "intervened": {k: v for k, v in intervened.items() if k not in {"y_true", "y_pred", "rows"}},
        "delta_acc": intervened["acc"] - base["acc"],
        "diagnostics": diagnostics,
        "rows": {
            "base": base["rows"],
            "intervened": intervened["rows"],
        },
    }
    save_json(Path(args.output_json), result)

    b_acc, b_lo, b_hi = base["acc"], base["ci95"][0], base["ci95"][1]
    i_acc, i_lo, i_hi = intervened["acc"], intervened["ci95"][0], intervened["ci95"][1]
    print(summarize_accuracy_line("base", b_acc, b_lo, b_hi, base["n"]))
    print(summarize_accuracy_line("probe", i_acc, i_lo, i_hi, intervened["n"]))
    print(f"delta_acc={100.0 * (i_acc - b_acc):.2f} points")
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
