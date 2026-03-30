import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    ensure_parent_dir,
    ensure_truthfulqa_csv,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    maybe_cap_items,
    save_json,
    split_calibration_eval,
)


PROMPT_H = (
    "You are confident and decisive. Even if uncertain, provide your best guess.\n\n"
    "Question: {question}\n"
    "Answer:"
)

PROMPT_G = (
    "You are careful and evidence-sensitive. If uncertain, avoid unsupported claims.\n\n"
    "Question: {question}\n"
    "Answer:"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract layer-wise hallucination directions")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--truthfulqa-csv",
        default="experiments/data/TruthfulQA.csv",
        help="Path to TruthfulQA CSV",
    )
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=160,
        help="How many items to use for direction estimation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap total rows before split (0 means no cap)",
    )
    parser.add_argument(
        "--output",
        default="experiments/artifacts/directions.npz",
        help="Output npz file",
    )
    parser.add_argument(
        "--metadata-json",
        default="experiments/artifacts/directions_meta.json",
        help="Metadata output path",
    )
    return parser.parse_args()


def get_prompt_last_hidden_states(model, tokenizer, prompt: str, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    vectors = []
    # hidden_states[0] is embeddings; use layers 1..L.
    for hs in hidden_states[1:]:
        vectors.append(hs[0, -1, :].float().cpu().numpy())
    return np.stack(vectors, axis=0)


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)

    calibration, _ = split_calibration_eval(items, args.calibration_size, args.seed)
    if not calibration:
        calibration = items[: min(len(items), max(32, args.calibration_size))]

    model, tokenizer = load_model_and_tokenizer(args.model, args.dtype)
    device = get_primary_device(model)

    sum_h = None
    sum_g = None

    for item in tqdm(calibration, desc="Direction extraction"):
        ph = PROMPT_H.format(question=item.question)
        pg = PROMPT_G.format(question=item.question)

        vh = get_prompt_last_hidden_states(model, tokenizer, ph, device)
        vg = get_prompt_last_hidden_states(model, tokenizer, pg, device)

        if sum_h is None:
            sum_h = np.zeros_like(vh)
            sum_g = np.zeros_like(vg)

        sum_h += vh
        sum_g += vg

    mean_h = sum_h / float(len(calibration))
    mean_g = sum_g / float(len(calibration))
    directions = mean_h - mean_g
    norms = np.linalg.norm(directions, axis=1)

    out_path = Path(args.output)
    ensure_parent_dir(out_path)
    np.savez(
        out_path,
        directions=directions.astype(np.float32),
        norms=norms.astype(np.float32),
        n_calibration=np.array([len(calibration)], dtype=np.int32),
    )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "calibration_size": len(calibration),
        "prompt_h_template": PROMPT_H,
        "prompt_g_template": PROMPT_G,
        "output": str(out_path),
    }
    save_json(Path(args.metadata_json), meta)

    top_idx = int(np.argmax(norms))
    print(
        "Saved directions. "
        f"layers={directions.shape[0]} hidden={directions.shape[1]} "
        f"top_norm_layer={top_idx} top_norm={norms[top_idx]:.4f}"
    )
    print(f"Direction file: {args.output}")
    print(f"Metadata file: {args.metadata_json}")


if __name__ == "__main__":
    main()
