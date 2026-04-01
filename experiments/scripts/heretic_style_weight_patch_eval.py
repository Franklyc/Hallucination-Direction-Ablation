import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from common import get_decoder_layers, get_layer_write_modules, load_model_and_tokenizer, save_json
from heretic_style_prompt_eval import HookContext, load_jsonl, run_condition


def parse_args():
    parser = argparse.ArgumentParser(description="HERETIC-style weight patch evaluation")
    parser.add_argument(
        "--pairs-jsonl",
        default="experiments/data/heretic_style/eval_pairs.jsonl",
        help="HERETIC-style eval prompt JSONL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument(
        "--gpu-memory-gb",
        type=int,
        default=15,
        help="Per-GPU memory cap in GiB for model loading",
    )
    parser.add_argument(
        "--directions-npz",
        required=True,
        help="Direction or basis NPZ",
    )
    parser.add_argument(
        "--direction-key",
        default="direct_minus_non_direct",
        help="Array key inside the NPZ",
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--alpha", type=float, default=0.2, help="Orthogonalization strength")
    parser.add_argument(
        "--modules",
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which write modules to patch",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Generation budget per prompt",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional row cap")
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/heretic_style_weight_patch_eval.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def parse_layers(raw: str):
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


@torch.no_grad()
def apply_subspace_patch(weight: torch.Tensor, basis: torch.Tensor, alpha: float):
    if weight.ndim != 2:
        raise ValueError("Expected a 2D weight matrix for weight patching.")
    local_basis = basis.to(device=weight.device, dtype=weight.dtype)
    if local_basis.ndim == 1:
        local_basis = local_basis.unsqueeze(0)
    proj_rows = torch.matmul(local_basis, weight)
    delta = torch.matmul(local_basis.transpose(0, 1), proj_rows)
    weight.sub_(alpha * delta)


def normalized_basis(layer_obj: np.ndarray) -> np.ndarray:
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


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.pairs_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No HERETIC-style eval rows loaded.")

    layer_indices = parse_layers(args.layers)
    if not layer_indices:
        raise ValueError("No layers selected.")

    direction_npz = np.load(args.directions_npz)
    if args.direction_key not in direction_npz:
        raise KeyError(f"Direction key '{args.direction_key}' not found in {args.directions_npz}")
    direction_matrix = np.asarray(direction_npz[args.direction_key], dtype=np.float32)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    layers = get_decoder_layers(model)
    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")

    base_rows, base_summary = run_condition(
        "base",
        model,
        tokenizer,
        rows,
        args.max_new_tokens,
        nullcontext(),
        HookContext(edit_generation_only=False, max_edited_tokens=0),
    )

    patch_log = []
    device = next(model.parameters()).device
    for layer_idx in layer_indices:
        basis_np = normalized_basis(direction_matrix[layer_idx])
        if basis_np.shape[0] == 0:
            patch_log.append({"layer": layer_idx, "patched_modules": [], "skipped": True})
            continue
        basis = torch.tensor(basis_np, dtype=torch.float32, device=device)
        patched_names = []
        for module_name, module in get_layer_write_modules(layers[layer_idx], args.modules):
            if not hasattr(module, "weight"):
                continue
            apply_subspace_patch(module.weight.data, basis, args.alpha)
            patched_names.append(module_name)
        patch_log.append(
            {
                "layer": layer_idx,
                "basis_rank": int(basis.shape[0]),
                "patched_modules": patched_names,
                "skipped": len(patched_names) == 0,
            }
        )

    patched_rows, patched_summary = run_condition(
        "patched",
        model,
        tokenizer,
        rows,
        args.max_new_tokens,
        nullcontext(),
        HookContext(edit_generation_only=False, max_edited_tokens=0),
    )

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "pairs_jsonl": args.pairs_jsonl,
        "n_eval": len(rows),
        "directions_npz": args.directions_npz,
        "direction_key": args.direction_key,
        "layers": layer_indices,
        "alpha": args.alpha,
        "modules": args.modules,
        "base": base_summary,
        "patched": patched_summary,
        "delta": {
            "overall_success_rate": patched_summary["overall_success_rate"] - base_summary["overall_success_rate"],
            "non_direct_success_rate": patched_summary["binary_success"].get("non_direct_ok", 0.0)
            - base_summary["binary_success"].get("non_direct_ok", 0.0),
            "direct_success_rate": patched_summary["binary_success"].get("direct_answer", 0.0)
            - base_summary["binary_success"].get("direct_answer", 0.0),
            "overconfident_non_direct_rate": patched_summary["overconfident_non_direct_rate"]
            - base_summary["overconfident_non_direct_rate"],
        },
        "patch_log": patch_log,
        "rows": {
            "base": base_rows,
            "patched": patched_rows,
        },
    }
    save_json(Path(args.output_json), result)
    print(json.dumps(result["delta"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
