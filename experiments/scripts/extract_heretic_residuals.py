import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import load_jsonl, load_model_and_tokenizer, save_json
from extract_direction import get_prompt_last_hidden_states


def parse_args():
    parser = argparse.ArgumentParser(description="Extract HERETIC-style first-token residual directions")
    parser.add_argument(
        "--pairs-jsonl",
        default="experiments/data/heretic_style/calibration_pairs.jsonl",
        help="HERETIC-style prompt JSONL",
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
        "--load-in-4bit",
        action="store_true",
        help="Load model with bitsandbytes 4-bit quantization",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row cap",
    )
    parser.add_argument(
        "--orthogonalize-against",
        default="none",
        choices=["none", "direct_mean", "non_direct_mean", "all_mean"],
        help="Optional projected-direction reference",
    )
    parser.add_argument(
        "--output-npz",
        default="experiments/artifacts/heretic_style_residuals.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        default="experiments/artifacts/heretic_style_residuals_meta.json",
        help="Output metadata JSON path",
    )
    return parser.parse_args()


def orthogonalize_per_layer(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.zeros_like(direction)
    for layer_idx in range(direction.shape[0]):
        vec = direction[layer_idx]
        ref = reference[layer_idx]
        denom = float(np.dot(ref, ref))
        if denom <= 1e-12:
            result[layer_idx] = vec
            continue
        result[layer_idx] = vec - (float(np.dot(vec, ref)) / denom) * ref
    return result


def layer_norms(matrix: np.ndarray):
    return [float(np.linalg.norm(matrix[i])) for i in range(matrix.shape[0])]


def top_layers(norms, top_k: int = 8):
    pairs = [{"layer": idx, "norm": float(norm)} for idx, norm in enumerate(norms)]
    return sorted(pairs, key=lambda x: x["norm"], reverse=True)[:top_k]


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.pairs_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No HERETIC-style prompt rows loaded.")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = next(model.parameters()).device

    vectors_by_bucket = defaultdict(list)
    vectors_by_binary = defaultdict(list)

    for row in tqdm(rows, desc="HERETIC residual extraction"):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": row["system_message"]},
                {"role": "user", "content": row["prompt_text"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        vecs = get_prompt_last_hidden_states(model, tokenizer, prompt, device)
        vectors_by_bucket[row["bucket"]].append(vecs)
        vectors_by_binary[row["binary_bucket"]].append(vecs)

    arrays_to_save = {}
    binary_means = {}

    for bucket, vectors in sorted(vectors_by_bucket.items()):
        stacked = np.stack(vectors, axis=0).astype(np.float32)
        arrays_to_save[f"{bucket}__samples"] = stacked
        arrays_to_save[f"{bucket}__mean"] = stacked.mean(axis=0)

    for bucket, vectors in sorted(vectors_by_binary.items()):
        stacked = np.stack(vectors, axis=0).astype(np.float32)
        mean_vec = stacked.mean(axis=0)
        arrays_to_save[f"{bucket}__samples"] = stacked
        arrays_to_save[f"{bucket}__mean"] = mean_vec
        binary_means[bucket] = mean_vec

    direct_mean = binary_means["direct_answer"]
    non_direct_mean = binary_means["non_direct_ok"]
    direct_minus_non_direct = direct_mean - non_direct_mean
    arrays_to_save["direct_minus_non_direct"] = direct_minus_non_direct.astype(np.float32)
    arrays_to_save["non_direct_minus_direct"] = (-direct_minus_non_direct).astype(np.float32)

    projected_key = None
    if args.orthogonalize_against != "none":
        if args.orthogonalize_against == "direct_mean":
            reference = direct_mean
        elif args.orthogonalize_against == "non_direct_mean":
            reference = non_direct_mean
        else:
            reference = (direct_mean + non_direct_mean) / 2.0
        projected = orthogonalize_per_layer(direct_minus_non_direct, reference).astype(np.float32)
        projected_key = f"direct_minus_non_direct__orth_to_{args.orthogonalize_against}"
        arrays_to_save[projected_key] = projected

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays_to_save)

    meta = {
        "pairs_jsonl": args.pairs_jsonl,
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "max_samples": args.max_samples,
        "orthogonalize_against": args.orthogonalize_against,
        "n_rows": len(rows),
        "bucket_counts": dict(sorted(Counter(row["bucket"] for row in rows).items())),
        "binary_bucket_counts": dict(sorted(Counter(row["binary_bucket"] for row in rows).items())),
        "top_layers_direct_minus_non_direct": top_layers(layer_norms(direct_minus_non_direct)),
        "projected_key": projected_key,
    }
    save_json(Path(args.metadata_json), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved residual directions to: {args.output_npz}")


if __name__ == "__main__":
    main()
