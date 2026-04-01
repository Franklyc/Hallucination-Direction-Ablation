import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import build_open_answer_prompt, get_primary_device, load_model_and_tokenizer, save_json
from extract_direction import get_answer_hidden_states


def parse_args():
    parser = argparse.ArgumentParser(description="Extract open-answer states for mechanistic hallucination analysis")
    parser.add_argument(
        "--annotation-pack-jsonl",
        required=True,
        help="Annotation pack JSONL",
    )
    parser.add_argument(
        "--onset-candidates-jsonl",
        required=True,
        help="Sentence-level onset candidate JSONL",
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
        help="Pooling mode for continuation tokens",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on rows",
    )
    parser.add_argument(
        "--output-npz",
        default="experiments/artifacts/mechanistic_answer_states.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        default="experiments/artifacts/mechanistic_answer_states_meta.json",
        help="Metadata JSON path",
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def layer_norms(matrix: np.ndarray):
    return [float(np.linalg.norm(matrix[i])) for i in range(matrix.shape[0])]


def main():
    args = parse_args()
    annotation_rows = load_jsonl(Path(args.annotation_pack_jsonl))
    onset_rows = load_jsonl(Path(args.onset_candidates_jsonl))
    onset_by_id = {row["question_id"]: row for row in onset_rows}
    if args.max_samples > 0:
        annotation_rows = annotation_rows[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    bucket_vectors = defaultdict(list)
    onset_vectors = defaultdict(list)
    n_layers = None

    for row in tqdm(annotation_rows, desc="Mechanistic state extraction"):
        prompt = build_open_answer_prompt(tokenizer, row["question"])
        answer_text = row["generated_answer"]
        full_vecs = get_answer_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            answer_text=answer_text,
            device=device,
            pool=args.pool,
        )
        if n_layers is None:
            n_layers = full_vecs.shape[0]
        bucket_vectors[row["auto_bucket"]].append(full_vecs)

        onset_row = onset_by_id.get(row["question_id"])
        onset_sentence = None if onset_row is None else onset_row.get("proposed_onset_sentence")
        if onset_sentence:
            onset_vecs = get_answer_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_text=onset_sentence,
                device=device,
                pool=args.pool,
            )
            onset_vectors[row["auto_bucket"]].append(onset_vecs)

    if n_layers is None:
        raise ValueError("No rows processed.")

    mean_vectors = {}
    onset_mean_vectors = {}
    counts = {}
    onset_counts = {}
    arrays_to_save = {}

    for bucket, vectors in sorted(bucket_vectors.items()):
        stacked = np.stack(vectors, axis=0)
        mean_vectors[bucket] = stacked.mean(axis=0)
        counts[bucket] = int(stacked.shape[0])
        arrays_to_save[f"{bucket}__mean"] = mean_vectors[bucket]

    for bucket, vectors in sorted(onset_vectors.items()):
        stacked = np.stack(vectors, axis=0)
        onset_mean_vectors[bucket] = stacked.mean(axis=0)
        onset_counts[bucket] = int(stacked.shape[0])
        arrays_to_save[f"{bucket}__onset_mean"] = onset_mean_vectors[bucket]

    comparisons = {}
    if "contradicted_reference" in mean_vectors and "supported_answer" in mean_vectors:
        diff = mean_vectors["contradicted_reference"] - mean_vectors["supported_answer"]
        arrays_to_save["contradicted_minus_supported"] = diff
        comparisons["contradicted_minus_supported_norms"] = layer_norms(diff)
    if "mixed_or_self_contradictory" in mean_vectors and "supported_answer" in mean_vectors:
        diff = mean_vectors["mixed_or_self_contradictory"] - mean_vectors["supported_answer"]
        arrays_to_save["mixed_minus_supported"] = diff
        comparisons["mixed_minus_supported_norms"] = layer_norms(diff)
    if "contradicted_reference" in onset_mean_vectors and "supported_answer" in onset_mean_vectors:
        diff = onset_mean_vectors["contradicted_reference"] - onset_mean_vectors["supported_answer"]
        arrays_to_save["onset_contradicted_minus_supported"] = diff
        comparisons["onset_contradicted_minus_supported_norms"] = layer_norms(diff)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays_to_save)

    meta = {
        "annotation_pack_jsonl": args.annotation_pack_jsonl,
        "onset_candidates_jsonl": args.onset_candidates_jsonl,
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "pool": args.pool,
        "n_rows": len(annotation_rows),
        "bucket_counts": dict(sorted(counts.items())),
        "bucket_onset_counts": dict(sorted(onset_counts.items())),
        "comparisons": comparisons,
    }
    save_json(Path(args.metadata_json), meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Saved state vectors to: {args.output_npz}")


if __name__ == "__main__":
    main()
