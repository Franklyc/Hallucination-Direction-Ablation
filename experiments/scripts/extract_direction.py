import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_chat_prompt,
    build_open_answer_prompt,
    ensure_parent_dir,
    ensure_truthfulqa_csv,
    get_primary_device,
    load_jsonl,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    maybe_cap_items,
    save_json,
    split_calibration_eval,
    stable_hash,
)


INSTRUCTION_SYSTEM = "You are a careful assistant focused on factual correctness."


def parse_args():
    parser = argparse.ArgumentParser(description="Extract layer-wise truthfulness directions")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--method",
        default="answer_state",
        choices=["instruction", "answer_state"],
        help="How to estimate the direction to remove at inference time",
    )
    parser.add_argument(
        "--truthfulqa-csv",
        default="experiments/data/TruthfulQA.csv",
        help="Path to TruthfulQA CSV",
    )
    parser.add_argument(
        "--contrastive-jsonl",
        default="experiments/data/prepared/calib_contrastive.jsonl",
        help="Prepared contrastive calibration rows for instruction mode",
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
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="How many questions to use for direction estimation in answer_state mode",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap total CSV rows before split in answer_state mode (0 means no cap)",
    )
    parser.add_argument(
        "--max-contrastive-rows",
        type=int,
        default=0,
        help="Optional cap on prepared contrastive rows in instruction mode",
    )
    parser.add_argument(
        "--positive-family",
        default="hallucination",
        help="Positive family for instruction mode; direction points from negative to positive",
    )
    parser.add_argument(
        "--negative-family",
        default="grounded",
        help="Negative family for instruction mode; direction points from negative to positive",
    )
    parser.add_argument(
        "--answer-pool",
        default="mean",
        choices=["mean", "first", "last"],
        help="How to pool answer-token hidden states in answer_state mode",
    )
    parser.add_argument(
        "--max-correct-variants",
        type=int,
        default=3,
        help="Max correct answer variants per question in answer_state mode",
    )
    parser.add_argument(
        "--max-incorrect-variants",
        type=int,
        default=3,
        help="Max incorrect answer variants per question in answer_state mode",
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


def normalize_continuation_text(text: str) -> str:
    if text and text[0] not in {" ", "\n", "\t"}:
        return " " + text
    return text


def get_prompt_last_hidden_states(model, tokenizer, prompt: str, device: torch.device):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    vectors = []
    for hs in hidden_states[1:]:
        vectors.append(hs[0, -1, :].float().cpu().numpy())
    return np.stack(vectors, axis=0)


def get_answer_hidden_states(
    model,
    tokenizer,
    prompt: str,
    answer_text: str,
    device: torch.device,
    pool: str,
):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_text = normalize_continuation_text(answer_text)
    cont_ids = tokenizer(cont_text, add_special_tokens=False)["input_ids"]
    if len(cont_ids) == 0:
        raise ValueError("Answer continuation tokenized to zero length.")

    input_ids = torch.tensor([prompt_ids + cont_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    start = len(prompt_ids)
    end = start + len(cont_ids)
    vectors = []
    for hs in outputs.hidden_states[1:]:
        span = hs[0, start:end, :]
        if pool == "first":
            vec = span[0]
        elif pool == "last":
            vec = span[-1]
        else:
            vec = span.mean(dim=0)
        vectors.append(vec.float().cpu().numpy())
    return np.stack(vectors, axis=0)


def build_instruction_prompt(row: dict, tokenizer) -> str:
    return build_chat_prompt(
        tokenizer=tokenizer,
        system_message=INSTRUCTION_SYSTEM,
        user_message=row["prompt_text"],
    )


def select_variants(values, limit: int, seed: int, question: str):
    candidates = list(values)
    if limit <= 0 or len(candidates) <= limit:
        return candidates
    keep = [candidates[0]]
    remainder = candidates[1:]
    rng = random.Random(seed + stable_hash(question))
    rng.shuffle(remainder)
    keep.extend(remainder[: max(0, limit - 1)])
    return keep


def extract_instruction_direction(args, model, tokenizer, device):
    rows = load_jsonl(Path(args.contrastive_jsonl))
    rows = [row for row in rows if row.get("prompt_family") in {args.positive_family, args.negative_family}]
    if args.max_contrastive_rows > 0:
        rows = rows[: args.max_contrastive_rows]
    if not rows:
        raise ValueError("No usable contrastive rows found for instruction-mode extraction.")

    sum_pos = None
    sum_neg = None
    count_pos = 0
    count_neg = 0

    for row in tqdm(rows, desc="Direction extraction"):
        prompt = build_instruction_prompt(row, tokenizer)
        vecs = get_prompt_last_hidden_states(model, tokenizer, prompt, device)
        family = row.get("prompt_family")
        if family == args.positive_family:
            if sum_pos is None:
                sum_pos = np.zeros_like(vecs)
            sum_pos += vecs
            count_pos += 1
        elif family == args.negative_family:
            if sum_neg is None:
                sum_neg = np.zeros_like(vecs)
            sum_neg += vecs
            count_neg += 1

    if count_pos <= 0 or count_neg <= 0:
        raise ValueError("Instruction-mode extraction needs non-empty positive and negative families.")

    mean_pos = sum_pos / float(count_pos)
    mean_neg = sum_neg / float(count_neg)
    directions = mean_pos - mean_neg
    metadata = {
        "method": "instruction",
        "contrastive_jsonl": args.contrastive_jsonl,
        "positive_family": args.positive_family,
        "negative_family": args.negative_family,
        "n_positive_rows": count_pos,
        "n_negative_rows": count_neg,
        "direction_semantics": f"{args.positive_family}_minus_{args.negative_family}",
        "intervention_semantics": "projection removal suppresses the positive-family component",
        "system_message": INSTRUCTION_SYSTEM,
    }
    return directions, metadata


def extract_answer_state_direction(args, model, tokenizer, device):
    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)
    calibration, _ = split_calibration_eval(items, args.calibration_size, args.seed)
    if not calibration:
        calibration = items[: min(len(items), max(32, args.calibration_size))]

    sum_pos = None
    sum_neg = None
    count_pos = 0
    count_neg = 0

    for item in tqdm(calibration, desc="Direction extraction"):
        prompt = build_open_answer_prompt(tokenizer, item.question)
        correct_variants = select_variants(
            item.correct_answers or [item.best_answer],
            args.max_correct_variants,
            seed=args.seed,
            question=item.question,
        )
        incorrect_variants = select_variants(
            item.incorrect_answers or [item.best_incorrect_answer],
            args.max_incorrect_variants,
            seed=args.seed + 1009,
            question=item.question,
        )

        for answer_text in incorrect_variants:
            vecs = get_answer_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_text=answer_text,
                device=device,
                pool=args.answer_pool,
            )
            if sum_pos is None:
                sum_pos = np.zeros_like(vecs)
            sum_pos += vecs
            count_pos += 1

        for answer_text in correct_variants:
            vecs = get_answer_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_text=answer_text,
                device=device,
                pool=args.answer_pool,
            )
            if sum_neg is None:
                sum_neg = np.zeros_like(vecs)
            sum_neg += vecs
            count_neg += 1

    if count_pos <= 0 or count_neg <= 0:
        raise ValueError("Answer-state extraction needs non-empty positive and negative examples.")

    mean_pos = sum_pos / float(count_pos)
    mean_neg = sum_neg / float(count_neg)
    directions = mean_pos - mean_neg
    metadata = {
        "method": "answer_state",
        "truthfulqa_csv": str(csv_path),
        "seed": args.seed,
        "calibration_size": len(calibration),
        "max_correct_variants": args.max_correct_variants,
        "max_incorrect_variants": args.max_incorrect_variants,
        "answer_pool": args.answer_pool,
        "n_positive_examples": count_pos,
        "n_negative_examples": count_neg,
        "direction_semantics": "incorrect_answer_state_minus_correct_answer_state",
        "intervention_semantics": "projection removal suppresses hidden-state components associated with incorrect answer continuations",
    }
    return directions, metadata


def main():
    args = parse_args()
    random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    if args.method == "instruction":
        directions, method_meta = extract_instruction_direction(args, model, tokenizer, device)
    else:
        directions, method_meta = extract_answer_state_direction(args, model, tokenizer, device)

    norms = np.linalg.norm(directions, axis=1)

    out_path = Path(args.output)
    ensure_parent_dir(out_path)
    np.savez(
        out_path,
        directions=directions.astype(np.float32),
        norms=norms.astype(np.float32),
    )

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "seed": args.seed,
        "output": str(out_path),
        **method_meta,
    }
    save_json(Path(args.metadata_json), meta)

    top_idx = int(np.argmax(norms))
    print(
        "Saved directions. "
        f"method={args.method} layers={directions.shape[0]} hidden={directions.shape[1]} "
        f"top_norm_layer={top_idx} top_norm={norms[top_idx]:.4f}"
    )
    print(f"Direction file: {args.output}")
    print(f"Metadata file: {args.metadata_json}")


if __name__ == "__main__":
    main()
