import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build silver snippet-level dataset for mechanistic hallucination analysis")
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
        "--supported-threshold",
        type=float,
        default=0.78,
        help="Minimum correct score for supported snippets",
    )
    parser.add_argument(
        "--unsupported-threshold",
        type=float,
        default=0.72,
        help="Minimum incorrect score for unsupported snippets",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Optional cap per label (0 means no cap)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=3,
        help="Minimum number of whitespace-delimited words in a snippet",
    )
    parser.add_argument(
        "--balance-labels",
        action="store_true",
        help="Downsample labels to the same count after filtering",
    )
    parser.add_argument(
        "--supported-margin",
        type=float,
        default=0.18,
        help="Minimum correct-minus-incorrect score gap for supported snippets",
    )
    parser.add_argument(
        "--unsupported-margin",
        type=float,
        default=0.18,
        help="Minimum incorrect-minus-correct score gap for unsupported snippets",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/artifacts/mechanistic_silver_dataset.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary-json",
        default="experiments/artifacts/mechanistic_silver_dataset_summary.json",
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


def confidence(row):
    return max(float(row.get("correct_score", 0.0)), float(row.get("incorrect_score", 0.0)))


def has_min_words(text: str, min_words: int) -> bool:
    return len([part for part in (text or "").split() if part.strip()]) >= min_words


def select_supported_sentences(pack_row, onset_row, supported_threshold: float, min_words: int, supported_margin: float):
    out = []
    for idx, sent in enumerate(onset_row.get("sentences", [])):
        if sent["bucket"] != "supported":
            continue
        if float(sent.get("correct_score", 0.0)) < supported_threshold:
            continue
        incorrect_score = float(sent.get("incorrect_score", 0.0))
        correct_score = float(sent.get("correct_score", 0.0))
        if incorrect_score > 0.65:
            continue
        if correct_score - incorrect_score < supported_margin:
            continue
        if not has_min_words(sent["sentence"], min_words):
            continue
        out.append(
            {
                "question_id": pack_row["question_id"],
                "category": pack_row["category"],
                "question": pack_row["question"],
                "label": "supported_sentence",
                "source_bucket": pack_row["auto_bucket"],
                "sentence_index": idx,
                "snippet_text": sent["sentence"],
                "correct_score": float(sent.get("correct_score", 0.0)),
                "incorrect_score": float(sent.get("incorrect_score", 0.0)),
                "matched_correct": sent.get("matched_correct"),
                "matched_incorrect": sent.get("matched_incorrect"),
                "generated_answer": pack_row["generated_answer"],
            }
        )
    return out


def select_unsupported_onset(pack_row, onset_row, unsupported_threshold: float, min_words: int, unsupported_margin: float):
    idx = onset_row.get("proposed_onset_sentence_index")
    if idx is None:
        return None
    sentences = onset_row.get("sentences", [])
    if idx < 0 or idx >= len(sentences):
        return None
    sent = sentences[idx]
    if sent["bucket"] not in {"contradicted", "mixed"}:
        return None
    incorrect_score = float(sent.get("incorrect_score", 0.0))
    correct_score = float(sent.get("correct_score", 0.0))
    if incorrect_score < unsupported_threshold:
        return None
    if incorrect_score - correct_score < unsupported_margin:
        return None
    if pack_row["auto_bucket"] not in {"contradicted_reference", "mixed_or_self_contradictory"}:
        return None
    if not has_min_words(sent["sentence"], min_words):
        return None
    return {
        "question_id": pack_row["question_id"],
        "category": pack_row["category"],
        "question": pack_row["question"],
        "label": "unsupported_onset_sentence",
        "source_bucket": pack_row["auto_bucket"],
        "sentence_index": idx,
        "snippet_text": sent["sentence"],
        "correct_score": float(sent.get("correct_score", 0.0)),
        "incorrect_score": float(sent.get("incorrect_score", 0.0)),
        "matched_correct": sent.get("matched_correct"),
        "matched_incorrect": sent.get("matched_incorrect"),
        "generated_answer": pack_row["generated_answer"],
    }


def take_per_label(rows, max_per_label: int):
    if max_per_label <= 0:
        return rows
    out = []
    counts = Counter()
    for row in rows:
        label = row["label"]
        if counts[label] >= max_per_label:
            continue
        out.append(row)
        counts[label] += 1
    return out


def balance_labels(rows):
    counts = Counter(row["label"] for row in rows)
    if not counts:
        return rows
    target = min(counts.values())
    out = []
    kept = Counter()
    for row in rows:
        label = row["label"]
        if kept[label] >= target:
            continue
        out.append(row)
        kept[label] += 1
    return out


def main():
    args = parse_args()
    pack_rows = load_jsonl(Path(args.annotation_pack_jsonl))
    onset_rows = load_jsonl(Path(args.onset_candidates_jsonl))
    onset_by_id = {row["question_id"]: row for row in onset_rows}

    selected = []
    for pack_row in pack_rows:
        onset_row = onset_by_id.get(pack_row["question_id"])
        if not onset_row:
            continue
        selected.extend(
            select_supported_sentences(
                pack_row,
                onset_row,
                args.supported_threshold,
                args.min_words,
                args.supported_margin,
            )
        )
        unsupported = select_unsupported_onset(
            pack_row,
            onset_row,
            args.unsupported_threshold,
            args.min_words,
            args.unsupported_margin,
        )
        if unsupported is not None:
            selected.append(unsupported)

    selected = sorted(
        selected,
        key=lambda row: (
            row["label"],
            -confidence(row),
            row["question_id"],
            row["sentence_index"],
        ),
    )
    selected = take_per_label(selected, args.max_per_label)
    if args.balance_labels:
        selected = balance_labels(selected)

    label_counts = Counter(row["label"] for row in selected)
    bucket_counts = Counter(row["source_bucket"] for row in selected)
    summary = {
        "annotation_pack_jsonl": args.annotation_pack_jsonl,
        "onset_candidates_jsonl": args.onset_candidates_jsonl,
        "supported_threshold": args.supported_threshold,
        "unsupported_threshold": args.unsupported_threshold,
        "min_words": args.min_words,
        "balance_labels": args.balance_labels,
        "supported_margin": args.supported_margin,
        "unsupported_margin": args.unsupported_margin,
        "n_rows": len(selected),
        "label_counts": dict(sorted(label_counts.items())),
        "source_bucket_counts": dict(sorted(bucket_counts.items())),
    }

    save_jsonl(Path(args.output_jsonl), selected)
    save_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved silver dataset to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
