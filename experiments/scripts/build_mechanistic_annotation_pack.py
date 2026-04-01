import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build manual annotation pack for mechanistic hallucination analysis")
    parser.add_argument(
        "--questions-jsonl",
        default="experiments/data/mechanistic_open/annotation_seed.jsonl",
        help="Prepared question JSONL with references",
    )
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Open generation evaluation JSON",
    )
    parser.add_argument(
        "--max-per-bucket",
        type=int,
        default=24,
        help="Cap examples per automatic bucket",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/artifacts/mechanistic_annotation_pack.jsonl",
        help="Annotation pack path",
    )
    parser.add_argument(
        "--summary-json",
        default="experiments/artifacts/mechanistic_annotation_pack_summary.json",
        help="Annotation pack summary path",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket_priority(bucket: str) -> int:
    order = {
        "mixed_or_self_contradictory": 0,
        "contradicted_reference": 1,
        "unresolved_needs_annotation": 2,
        "generic_abstention": 3,
        "supported_abstention": 4,
        "supported_answer": 5,
    }
    return order.get(bucket, 99)


def main():
    args = parse_args()
    question_rows = load_jsonl(Path(args.questions_jsonl))
    eval_obj = load_json(Path(args.eval_json))
    eval_rows = eval_obj["rows"]

    questions_by_id = {row["question_id"]: row for row in question_rows}
    grouped = defaultdict(list)
    for row in eval_rows:
        grouped[row["bucket"]].append(row)

    selected = []
    for bucket, bucket_rows in sorted(grouped.items(), key=lambda item: bucket_priority(item[0])):
        bucket_rows = sorted(
            bucket_rows,
            key=lambda row: (
                -max(float(row.get("incorrect_score", 0.0)), float(row.get("correct_score", 0.0))),
                row["question_id"],
            ),
        )
        for row in bucket_rows[: args.max_per_bucket]:
            source = questions_by_id.get(row["question_id"])
            if not source:
                continue
            selected.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "expected_behavior": row["expected_behavior"],
                    "question": row["question"],
                    "generated_answer": row["answer_text"],
                    "auto_bucket": row["bucket"],
                    "auto_correct_score": row.get("correct_score", 0.0),
                    "auto_incorrect_score": row.get("incorrect_score", 0.0),
                    "auto_matched_correct": row.get("matched_correct"),
                    "auto_matched_incorrect": row.get("matched_incorrect"),
                    "best_answer": source.get("best_answer"),
                    "best_incorrect_answer": source.get("best_incorrect_answer"),
                    "correct_answers": source.get("correct_answers", []),
                    "incorrect_answers": source.get("incorrect_answers", []),
                    "annotation": {
                        "support_label": "",
                        "first_unsupported_sentence_index": None,
                        "first_unsupported_span": "",
                        "needs_abstention": None,
                        "notes": "",
                    },
                }
            )

    bucket_counts = Counter(row["auto_bucket"] for row in selected)
    category_counts = Counter(row["category"] for row in selected)
    summary = {
        "questions_jsonl": args.questions_jsonl,
        "eval_json": args.eval_json,
        "n_selected": len(selected),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "category_counts_top20": dict(category_counts.most_common(20)),
    }

    save_jsonl(Path(args.output_jsonl), selected)
    save_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved annotation pack to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
