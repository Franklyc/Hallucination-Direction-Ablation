import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    bootstrap_accuracy_ci,
    ensure_truthfulqa_csv,
    get_decoder_layers,
    get_binary_letter_candidates,
    get_layer_write_modules,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    make_binary_instance,
    maybe_cap_items,
    parse_int_list,
    save_json,
    sequence_logprob,
    split_calibration_eval,
    stable_hash,
    summarize_intervention_rows,
    summarize_accuracy_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal rank-one weight patch evaluation")
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
        help="Load model with bitsandbytes 4-bit quantization (not supported for patching)",
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
        "--alpha",
        type=float,
        default=0.35,
        help="Rank-one orthogonalization strength",
    )
    parser.add_argument(
        "--modules",
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which write modules to patch",
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
        default="experiments/artifacts/weight_patch_eval.json",
        help="Where to save metrics",
    )
    return parser.parse_args()


def evaluate_binary(
    model,
    tokenizer,
    device,
    items,
    seed: int,
    candidate_a: str,
    candidate_b: str,
    bootstrap_rounds: int,
):
    y_true = []
    y_pred = []
    rows = []
    for item in tqdm(items, desc="Eval"):
        row_rng = random.Random(seed + stable_hash(item.question))
        prompt, correct, _, _ = make_binary_instance(item, row_rng, tokenizer)

        lp_a = sequence_logprob(
            model,
            tokenizer,
            prompt,
            candidate_a,
            device,
            add_leading_space=False,
        )
        lp_b = sequence_logprob(
            model,
            tokenizer,
            prompt,
            candidate_b,
            device,
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
        "rows": rows,
    }


@torch.no_grad()
def apply_rank_one_patch(weight: torch.Tensor, v_hat: torch.Tensor, alpha: float):
    v = v_hat.to(weight.device, dtype=weight.dtype)
    if weight.ndim != 2:
        raise ValueError("Expected a 2D weight matrix for rank-one patch.")

    proj_row = torch.matmul(v.unsqueeze(0), weight)
    delta = alpha * torch.matmul(v.unsqueeze(1), proj_row)
    weight.sub_(delta)


def main():
    args = parse_args()
    random.seed(args.seed)
    if args.load_in_4bit:
        raise ValueError(
            "Weight patch requires non-quantized weights. "
            "Run without --load-in-4bit (recommended dtype: bfloat16)."
        )

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
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)
    cand_a, cand_b = get_binary_letter_candidates(args.candidate_prefix)

    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(
                f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}"
            )

    base = evaluate_binary(
        model,
        tokenizer,
        device,
        eval_items,
        args.seed,
        candidate_a=cand_a,
        candidate_b=cand_b,
        bootstrap_rounds=args.bootstrap,
    )

    patch_log = []
    for layer_idx in selected_layers:
        layer = layers[layer_idx]
        v = directions[layer_idx]
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            patch_log.append(
                {
                    "layer": layer_idx,
                    "patched_modules": [],
                    "skipped": True,
                    "reason": "near-zero direction norm",
                }
            )
            continue

        v_hat = torch.tensor(v / norm, dtype=torch.float32, device=device)

        modules = get_layer_write_modules(layer, args.modules)
        patched_names = []
        for module_name, module in modules:
            if not hasattr(module, "weight"):
                continue
            apply_rank_one_patch(module.weight.data, v_hat, args.alpha)
            patched_names.append(module_name)

        patch_log.append(
            {
                "layer": layer_idx,
                "patched_modules": patched_names,
                "skipped": len(patched_names) == 0,
            }
        )

    patched = evaluate_binary(
        model,
        tokenizer,
        device,
        eval_items,
        args.seed,
        candidate_a=cand_a,
        candidate_b=cand_b,
        bootstrap_rounds=args.bootstrap,
    )

    diagnostics = summarize_intervention_rows(
        base_rows=base["rows"],
        new_rows=patched["rows"],
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
        "alpha": args.alpha,
        "modules": args.modules,
        "base": {k: v for k, v in base.items() if k != "rows"},
        "patched": {k: v for k, v in patched.items() if k != "rows"},
        "delta_acc": patched["acc"] - base["acc"],
        "patch_log": patch_log,
        "diagnostics": diagnostics,
        "rows": {
            "base": base["rows"],
            "patched": patched["rows"],
        },
    }
    save_json(Path(args.output_json), result)

    b_acc, b_lo, b_hi = base["acc"], base["ci95"][0], base["ci95"][1]
    p_acc, p_lo, p_hi = patched["acc"], patched["ci95"][0], patched["ci95"][1]
    print(summarize_accuracy_line("base", b_acc, b_lo, b_hi, base["n"]))
    print(summarize_accuracy_line("patched", p_acc, p_lo, p_hi, patched["n"]))
    print(f"delta_acc={100.0 * (p_acc - b_acc):.2f} points")
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
