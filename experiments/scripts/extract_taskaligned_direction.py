import argparse
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_chat_prompt,
    ensure_truthfulqa_csv,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    maybe_cap_items,
    save_json,
    split_calibration_eval,
)


PROMPT_H = "Even if you are unsure, choose the answer that seems most plausible."
PROMPT_G = "Choose the answer only if it is well-supported; avoid common misconceptions."


def parse_args():
    parser = argparse.ArgumentParser(description="Extract task-aligned prompt-state directions from TruthfulQA binary prompts.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model id or local path")
    parser.add_argument("--truthfulqa-csv", default="experiments/data/TruthfulQA.csv", help="Path to TruthfulQA CSV")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with bitsandbytes 4-bit quantization")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--calibration-size", type=int, default=200, help="Calibration split size")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap total rows before split")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to tokenizer.apply_chat_template when supported",
    )
    parser.add_argument(
        "--answer-order",
        default="ab_fixed",
        choices=["ab_fixed", "ba_fixed", "random"],
        help="How to order the correct and incorrect choices in the binary prompt",
    )
    parser.add_argument("--output", required=True, help="Output NPZ file")
    parser.add_argument("--metadata-json", required=True, help="Metadata JSON file")
    return parser.parse_args()


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


def build_binary_user_message(question: str, correct: str, incorrect: str, order: str, rng: random.Random):
    mode = order
    if mode == "random":
        mode = "ab_fixed" if rng.random() < 0.5 else "ba_fixed"

    if mode == "ab_fixed":
        a_text = correct
        b_text = incorrect
    elif mode == "ba_fixed":
        a_text = incorrect
        b_text = correct
    else:
        raise ValueError(f"Unsupported answer order: {order}")

    return (
        f"Question: {question}\n"
        f"A. {a_text}\n"
        f"B. {b_text}\n"
        "Answer with only one letter: A or B."
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)
    calibration, _ = split_calibration_eval(items, args.calibration_size, args.seed)
    if not calibration:
        calibration = items[: min(len(items), max(32, args.calibration_size))]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    sum_h = None
    sum_g = None
    for idx, item in enumerate(tqdm(calibration, desc="Taskaligned extraction")):
        rng = random.Random(args.seed * 1000003 + idx)
        user_message = build_binary_user_message(
            question=item.question,
            correct=item.best_answer,
            incorrect=item.best_incorrect_answer,
            order=args.answer_order,
            rng=rng,
        )
        prompt_h = build_chat_prompt(
            tokenizer,
            system_message=PROMPT_H,
            user_message=user_message,
            enable_thinking=False if args.disable_thinking else None,
        )
        prompt_g = build_chat_prompt(
            tokenizer,
            system_message=PROMPT_G,
            user_message=user_message,
            enable_thinking=False if args.disable_thinking else None,
        )
        vec_h = get_prompt_last_hidden_states(model, tokenizer, prompt_h, device)
        vec_g = get_prompt_last_hidden_states(model, tokenizer, prompt_g, device)

        if sum_h is None:
            sum_h = np.zeros_like(vec_h, dtype=np.float64)
            sum_g = np.zeros_like(vec_g, dtype=np.float64)
        sum_h += vec_h
        sum_g += vec_g

    mean_h = (sum_h / float(len(calibration))).astype(np.float32)
    mean_g = (sum_g / float(len(calibration))).astype(np.float32)
    directions = (mean_h - mean_g).astype(np.float32)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, directions=directions, prompt_h_mean=mean_h, prompt_g_mean=mean_g)

    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "disable_thinking": args.disable_thinking,
        "seed": args.seed,
        "calibration_size": len(calibration),
        "prompt_h_template": PROMPT_H,
        "prompt_g_template": PROMPT_G,
        "prompt_format": f"binary_choice_{args.answer_order}",
        "direction_semantics": "prompt_h_minus_prompt_g",
        "output": str(output_path),
    }
    save_json(Path(args.metadata_json), meta)
    print(f"Saved direction file to: {output_path}")
    print(f"Saved metadata to: {args.metadata_json}")


if __name__ == "__main__":
    main()
