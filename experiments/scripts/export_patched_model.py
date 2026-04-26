import argparse
from pathlib import Path

import numpy as np
import torch

from common import (
    get_decoder_layers,
    get_layer_write_modules,
    get_primary_device,
    load_model_and_tokenizer,
    parse_int_list,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize a rank-one patched model to a local directory.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model id or local path")
    parser.add_argument("--directions", required=True, help="Path to directions npz")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB")
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--alpha", type=float, required=True, help="Rank-one orthogonalization strength")
    parser.add_argument("--modules", default="mlp", choices=["attn", "mlp", "both"], help="Which modules to patch")
    parser.add_argument("--safe-serialization", action="store_true", help="Write safetensors shards instead of PyTorch bin")
    parser.add_argument("--max-shard-size", default="2GB", help="Shard size for save_pretrained")
    parser.add_argument("--output-dir", required=True, help="Directory to save the patched model")
    return parser.parse_args()


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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    patch_log = []
    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}")

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
        patched_names = []
        for module_name, module in get_layer_write_modules(layers[layer_idx], args.modules):
            apply_rank_one_patch(module.weight.data, v_hat, args.alpha)
            patched_names.append(module_name)

        patch_log.append(
            {
                "layer": layer_idx,
                "patched_modules": patched_names,
                "skipped": len(patched_names) == 0,
            }
        )

    model.save_pretrained(
        output_dir,
        max_shard_size=args.max_shard_size,
        safe_serialization=args.safe_serialization,
    )
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "base_model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "directions": args.directions,
        "layers": selected_layers,
        "alpha": args.alpha,
        "modules": args.modules,
        "safe_serialization": args.safe_serialization,
        "max_shard_size": args.max_shard_size,
        "output_dir": str(output_dir),
        "patch_log": patch_log,
    }
    save_json(output_dir / "patch_manifest.json", manifest)
    print(f"Saved patched model to: {output_dir}")
    print(f"Saved patch manifest to: {output_dir / 'patch_manifest.json'}")


if __name__ == "__main__":
    main()
