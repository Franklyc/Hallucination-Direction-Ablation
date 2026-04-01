import argparse
import json
from pathlib import Path

import numpy as np

from common import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build a mean-difference direction from NPZ arrays")
    parser.add_argument("--input-npz", required=True, help="Source NPZ path")
    parser.add_argument("--positive-key", required=True, help="Positive array key")
    parser.add_argument("--negative-key", required=True, help="Negative array key")
    parser.add_argument(
        "--reference-key",
        default="",
        help="Optional reference array key for orthogonalized projection",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--direction-key",
        default="direction",
        help="Key name for the saved mean-difference array",
    )
    return parser.parse_args()


def to_mean(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=0).astype(np.float32)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported array rank for mean conversion: {arr.ndim}")


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


def top_layers(direction: np.ndarray, top_k: int = 8):
    rows = [
        {"layer": idx, "norm": float(np.linalg.norm(direction[idx]))}
        for idx in range(direction.shape[0])
    ]
    return sorted(rows, key=lambda x: x["norm"], reverse=True)[:top_k]


def main():
    args = parse_args()
    data = np.load(args.input_npz)
    if args.positive_key not in data or args.negative_key not in data:
        raise KeyError("Positive or negative key missing from NPZ.")

    positive = to_mean(np.asarray(data[args.positive_key], dtype=np.float32))
    negative = to_mean(np.asarray(data[args.negative_key], dtype=np.float32))
    direction = positive - negative

    arrays_to_save = {
        args.direction_key: direction.astype(np.float32),
        f"{args.positive_key}__mean": positive.astype(np.float32),
        f"{args.negative_key}__mean": negative.astype(np.float32),
    }

    orth_key = None
    if args.reference_key:
        if args.reference_key not in data:
            raise KeyError(f"Reference key '{args.reference_key}' missing from NPZ.")
        reference = to_mean(np.asarray(data[args.reference_key], dtype=np.float32))
        orth = orthogonalize_per_layer(direction, reference).astype(np.float32)
        orth_key = f"{args.direction_key}__orth_to_{args.reference_key}"
        arrays_to_save[orth_key] = orth

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays_to_save)

    meta = {
        "input_npz": args.input_npz,
        "positive_key": args.positive_key,
        "negative_key": args.negative_key,
        "direction_key": args.direction_key,
        "reference_key": args.reference_key or None,
        "orthogonalized_key": orth_key,
        "top_layers": top_layers(direction),
    }
    save_json(Path(args.metadata_json), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved direction file to: {args.output_npz}")


if __name__ == "__main__":
    main()
