import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build reviewed snippet dataset from manually reviewed annotations")
    parser.add_argument(
        "--reviewed-jsonl",
        required=True,
        help="Reviewed annotation pack JSONL",
    )
    parser.add_argument(
        "--onset-candidates-jsonl",
        required=True,
        help="Sentence-level onset candidate JSONL",
    )
    parser.add_argument(
        "--min-supported-score",
        type=float,
        default=0.74,
        help="Minimum correct score for supported sentence snippets",
    )
    parser.add_argument(
        "--max-supported-incorrect-score",
        type=float,
        default=0.66,
        help="Maximum incorrect score for supported sentence snippets",
    )
    parser.add_argument(
        "--min-supported-margin",
        type=float,
        default=0.12,
        help="Minimum correct-minus-incorrect margin for supported sentence snippets",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum snippet length in whitespace-delimited words",
    )
    parser.add_argument(
        "--balance-labels",
        action="store_true",
        help="Downsample labels to the same count",
    )
    parser.add_argument(
        "--use-span-for-unsupported",
        action="store_true",
        help="Use the manually marked unsupported span instead of the full onset sentence when possible",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/artifacts/mechanistic_reviewed_dataset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary-json",
        default="experiments/artifacts/mechanistic_reviewed_dataset_summary.json",
        help="Output summary JSON path",
    )
    return parser.parse_args()


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


def word_count(text: str) -> int:
    return len([part for part in (text or "").split() if part.strip()])


def pick_supported_sentence(row, onset_row, args):
    candidates = []
    for idx, sent in enumerate(onset_row.get("sentences", [])):
        correct_score = float(sent.get("correct_score", 0.0))
        incorrect_score = float(sent.get("incorrect_score", 0.0))
        if sent.get("bucket") != "supported":
            continue
        if correct_score < args.min_supported_score:
            continue
        if incorrect_score > args.max_supported_incorrect_score:
            continue
        if correct_score - incorrect_score < args.min_supported_margin:
            continue
        if word_count(sent.get("sentence", "")) < args.min_words:
            continue
        candidates.append((correct_score - incorrect_score, correct_score, -incorrect_score, -idx, sent, idx))
    if not candidates:
        return None
    _, _, _, _, sent, idx = sorted(candidates, reverse=True)[0]
    return {
        "question_id": row["question_id"],
        "category": row["category"],
        "question": row["question"],
        "label": "supported_sentence",
        "source_support_label": row["annotation"]["support_label"],
        "source_bucket": row["auto_bucket"],
        "sentence_index": idx,
        "snippet_text": sent["sentence"],
        "correct_score": float(sent.get("correct_score", 0.0)),
        "incorrect_score": float(sent.get("incorrect_score", 0.0)),
        "matched_correct": sent.get("matched_correct"),
        "matched_incorrect": sent.get("matched_incorrect"),
        "first_unsupported_span": "",
        "generated_answer": row["generated_answer"],
    }


def pick_unsupported_snippet(row, onset_row, args):
    ann = row["annotation"]
    idx = ann.get("first_unsupported_sentence_index")
    if idx is None:
        return None
    sentences = onset_row.get("sentences", [])
    if idx < 0 or idx >= len(sentences):
        return None
    sent = sentences[idx]
    text = sent.get("sentence", "")
    span_text = ann.get("first_unsupported_span") or ""
    if args.use_span_for_unsupported and word_count(span_text) >= args.min_words:
        text = span_text
    if word_count(text) < args.min_words:
        return None
    return {
        "question_id": row["question_id"],
        "category": row["category"],
        "question": row["question"],
        "label": "unsupported_onset_snippet",
        "source_support_label": ann["support_label"],
        "source_bucket": row["auto_bucket"],
        "sentence_index": idx,
        "snippet_text": text,
        "onset_sentence_text": sent.get("sentence", ""),
        "correct_score": float(sent.get("correct_score", 0.0)),
        "incorrect_score": float(sent.get("incorrect_score", 0.0)),
        "matched_correct": sent.get("matched_correct"),
        "matched_incorrect": sent.get("matched_incorrect"),
        "first_unsupported_span": ann.get("first_unsupported_span", ""),
        "generated_answer": row["generated_answer"],
    }


def confidence(row):
    return max(float(row.get("correct_score", 0.0)), float(row.get("incorrect_score", 0.0)))


def balance_labels(rows):
    counts = Counter(row["label"] for row in rows)
    if not counts:
        return rows
    target = min(counts.values())
    kept = Counter()
    out = []
    for row in rows:
        label = row["label"]
        if kept[label] >= target:
            continue
        out.append(row)
        kept[label] += 1
    return out


def main():
    args = parse_args()
    reviewed_rows = load_jsonl(Path(args.reviewed_jsonl))
    onset_rows = load_jsonl(Path(args.onset_candidates_jsonl))
    onset_by_id = {row["question_id"]: row for row in onset_rows}

    selected = []
    for row in reviewed_rows:
        onset_row = onset_by_id.get(row["question_id"])
        if not onset_row:
            continue
        label = row["annotation"]["support_label"]
        if label == "supported":
            item = pick_supported_sentence(row, onset_row, args)
            if item is not None:
                selected.append(item)
        elif label in {"unsupported", "mixed"}:
            item = pick_unsupported_snippet(row, onset_row, args)
            if item is not None:
                selected.append(item)

    selected = sorted(
        selected,
        key=lambda row: (
            row["label"],
            -confidence(row),
            row["question_id"],
        ),
    )
    if args.balance_labels:
        selected = balance_labels(selected)

    summary = {
        "reviewed_jsonl": args.reviewed_jsonl,
        "onset_candidates_jsonl": args.onset_candidates_jsonl,
        "min_supported_score": args.min_supported_score,
        "max_supported_incorrect_score": args.max_supported_incorrect_score,
        "min_supported_margin": args.min_supported_margin,
        "min_words": args.min_words,
        "balance_labels": args.balance_labels,
        "use_span_for_unsupported": args.use_span_for_unsupported,
        "n_rows": len(selected),
        "label_counts": dict(sorted(Counter(row["label"] for row in selected).items())),
        "source_support_label_counts": dict(sorted(Counter(row["source_support_label"] for row in selected).items())),
    }
    save_jsonl(Path(args.output_jsonl), selected)
    save_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved reviewed dataset to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
