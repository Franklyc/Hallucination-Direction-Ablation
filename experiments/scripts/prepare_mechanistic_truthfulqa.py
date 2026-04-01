import argparse
import json
from collections import Counter
from pathlib import Path

from common import ensure_truthfulqa_csv, load_truthfulqa_binary_items, save_json, split_calibration_eval


UNCERTAINTY_MARKERS = [
    "unclear",
    "unknown",
    "not known",
    "no one knows",
    "cannot be determined",
    "there is no scientific evidence",
    "there's no scientific evidence",
    "not enough evidence",
    "not enough information",
    "the precise origin",
    "there is no known",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare open-ended TruthfulQA splits for mechanistic hallucination work")
    parser.add_argument(
        "--truthfulqa-csv",
        default="experiments/data/TruthfulQA.csv",
        help="Path to TruthfulQA CSV",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=200,
        help="Calibration split size",
    )
    parser.add_argument(
        "--annotation-size",
        type=int,
        default=80,
        help="How many held-out questions to reserve as an annotation starter pack",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/data/mechanistic_open",
        help="Where to save prepared JSONL files",
    )
    parser.add_argument(
        "--report-json",
        default="experiments/artifacts/mechanistic_prepare_report.json",
        help="Where to save the preparation report",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_expected_behavior(best_answer: str, correct_answers):
    values = [best_answer, *(correct_answers or [])]
    joined = " || ".join(v.lower() for v in values if v)
    if any(marker in joined for marker in UNCERTAINTY_MARKERS):
        return "qualified_or_abstain_ok"
    return "direct_truthful_answer"


def to_row(item, split: str):
    return {
        "question_id": f"truthfulqa_{item.source_row}",
        "source_row": item.source_row,
        "split": split,
        "category": item.category,
        "question": item.question,
        "best_answer": item.best_answer,
        "best_incorrect_answer": item.best_incorrect_answer,
        "correct_answers": list(item.correct_answers),
        "incorrect_answers": list(item.incorrect_answers),
        "expected_behavior": infer_expected_behavior(item.best_answer, item.correct_answers),
    }


def main():
    args = parse_args()

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    calibration_items, eval_items = split_calibration_eval(items, args.calibration_size, args.seed)
    if not eval_items:
        raise ValueError("Held-out eval split is empty.")

    annotation_size = min(max(0, args.annotation_size), len(eval_items))
    annotation_items = eval_items[:annotation_size]

    calibration_rows = [to_row(item, "calibration") for item in calibration_items]
    eval_rows = [to_row(item, "eval") for item in eval_items]
    annotation_rows = [to_row(item, "annotation_seed") for item in annotation_items]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "open_calibration.jsonl"
    eval_path = output_dir / "open_eval.jsonl"
    annotation_path = output_dir / "annotation_seed.jsonl"

    write_jsonl(calibration_path, calibration_rows)
    write_jsonl(eval_path, eval_rows)
    write_jsonl(annotation_path, annotation_rows)

    calib_behavior = Counter(row["expected_behavior"] for row in calibration_rows)
    eval_behavior = Counter(row["expected_behavior"] for row in eval_rows)

    report = {
        "source": {
            "csv_path": str(csv_path),
            "n_questions": len(items),
        },
        "split": {
            "seed": args.seed,
            "calibration_size_target": args.calibration_size,
            "calibration_size_actual": len(calibration_rows),
            "heldout_size": len(eval_rows),
            "annotation_seed_size": len(annotation_rows),
            "stratified": True,
        },
        "outputs": {
            "open_calibration": str(calibration_path),
            "open_eval": str(eval_path),
            "annotation_seed": str(annotation_path),
        },
        "expected_behavior_counts": {
            "calibration": dict(calib_behavior),
            "eval": dict(eval_behavior),
        },
    }
    save_json(Path(args.report_json), report)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
