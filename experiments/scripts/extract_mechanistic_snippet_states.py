import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import build_open_answer_prompt, get_primary_device, load_model_and_tokenizer, save_json
from extract_direction import get_answer_hidden_states


def parse_args():
    parser = argparse.ArgumentParser(description="Extract snippet-level answer states for mechanistic hallucination analysis")
    parser.add_argument(
        "--silver-jsonl",
        required=True,
        help="Snippet-level silver dataset JSONL",
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
        "--pool",
        default="mean",
        choices=["mean", "first", "last"],
        help="Pooling mode for snippet tokens",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row cap",
    )
    parser.add_argument(
        "--output-npz",
        default="experiments/artifacts/mechanistic_snippet_states.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        default="experiments/artifacts/mechanistic_snippet_states_meta.json",
        help="Metadata JSON path",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def layer_norms(matrix: np.ndarray):
    return [float(np.linalg.norm(matrix[i])) for i in range(matrix.shape[0])]


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.silver_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No rows loaded.")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    vectors_by_label = defaultdict(list)
    n_layers = None

    for row in tqdm(rows, desc="Snippet state extraction"):
        prompt = build_open_answer_prompt(tokenizer, row["question"])
        vecs = get_answer_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer_text=row["snippet_text"],
            device=device,
            pool=args.pool,
        )
        if n_layers is None:
            n_layers = vecs.shape[0]
        vectors_by_label[row["label"]].append(vecs)

    arrays_to_save = {}
    counts = {}
    mean_vectors = {}
    for label, vectors in sorted(vectors_by_label.items()):
        stacked = np.stack(vectors, axis=0)
        counts[label] = int(stacked.shape[0])
        mean_vectors[label] = stacked.mean(axis=0)
        arrays_to_save[f"{label}__mean"] = mean_vectors[label]
        arrays_to_save[f"{label}__samples"] = stacked

    comparisons = {}
    unsupported_key = None
    if "unsupported_onset_snippet" in mean_vectors:
        unsupported_key = "unsupported_onset_snippet"
    elif "unsupported_onset_sentence" in mean_vectors:
        unsupported_key = "unsupported_onset_sentence"
    if unsupported_key and "supported_sentence" in mean_vectors:
        diff = mean_vectors[unsupported_key] - mean_vectors["supported_sentence"]
        arrays_to_save["unsupported_minus_supported"] = diff
        comparisons["unsupported_minus_supported_norms"] = layer_norms(diff)
        comparisons["unsupported_label_used"] = unsupported_key

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays_to_save)

    meta = {
        "silver_jsonl": args.silver_jsonl,
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "pool": args.pool,
        "n_rows": len(rows),
        "label_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "comparisons": comparisons,
    }
    save_json(Path(args.metadata_json), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved snippet states to: {args.output_npz}")


if __name__ == "__main__":
    main()
