import argparse
import random
from pathlib import Path

from tqdm import tqdm

from common import (
    bootstrap_accuracy_ci,
    ensure_truthfulqa_csv,
    get_binary_letter_candidates,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    make_binary_instance,
    maybe_cap_items,
    save_json,
    sequence_logprob,
    split_calibration_eval,
    stable_hash,
    summarize_category_accuracy,
    summarize_accuracy_line,
)


def parse_args():
    parser = argparse.ArgumentParser(description="TruthfulQA binary-choice evaluation")
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
        help="Calibration split size to hold out from final eval",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Cap total rows before split (0 means no cap)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Bootstrap rounds for CI",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass enable_thinking=False to tokenizer.apply_chat_template when supported",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/baseline_binary_eval.json",
        help="Where to save metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)

    _, eval_items = split_calibration_eval(items, args.calibration_size, args.seed)
    if not eval_items:
        eval_items = items

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    cand_a, cand_b = get_binary_letter_candidates(args.candidate_prefix)

    y_true = []
    y_pred = []
    rows = []

    for item in tqdm(eval_items, desc="Binary eval"):
        row_rng = random.Random(args.seed + stable_hash(item.question))
        prompt, correct, a_text, b_text = make_binary_instance(
            item,
            row_rng,
            tokenizer,
            enable_thinking=False if args.disable_thinking else None,
        )

        lp_a = sequence_logprob(
            model,
            tokenizer,
            prompt,
            cand_a,
            device,
            add_leading_space=False,
        )
        lp_b = sequence_logprob(
            model,
            tokenizer,
            prompt,
            cand_b,
            device,
            add_leading_space=False,
        )

        pred = "A" if lp_a >= lp_b else "B"
        y_true.append(1 if correct == "A" else 0)
        y_pred.append(1 if pred == "A" else 0)

        rows.append(
            {
                "category": item.category,
                "question": item.question,
                "correct": correct,
                "pred": pred,
                "logprob_A": lp_a,
                "logprob_B": lp_b,
                "choice_A": a_text,
                "choice_B": b_text,
            }
        )

    acc, lo, hi = bootstrap_accuracy_ci(
        y_true,
        y_pred,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    out = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "candidate_prefix": args.candidate_prefix,
        "disable_thinking": args.disable_thinking,
        "seed": args.seed,
        "n_eval": len(eval_items),
        "calibration_size": args.calibration_size,
        "accuracy": acc,
        "ci95": [lo, hi],
        "category_accuracy": summarize_category_accuracy(rows),
        "rows": rows,
    }

    save_json(Path(args.output_json), out)
    print(summarize_accuracy_line("baseline", acc, lo, hi, len(eval_items)))
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
