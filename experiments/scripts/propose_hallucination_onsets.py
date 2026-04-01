import argparse
import json
import re
from pathlib import Path

from truthfulqa_open_generation_eval import best_variant_match


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def parse_args():
    parser = argparse.ArgumentParser(description="Propose sentence-level hallucination onset candidates")
    parser.add_argument(
        "--annotation-pack-jsonl",
        required=True,
        help="Annotation pack JSONL",
    )
    parser.add_argument(
        "--output-jsonl",
        default="experiments/artifacts/mechanistic_onset_candidates.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--summary-json",
        default="experiments/artifacts/mechanistic_onset_candidates_summary.json",
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


def split_sentences(text: str):
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text or "") if part.strip()]
    return parts or [text.strip()]


def sentence_bucket(sentence: str, correct_answers, incorrect_answers):
    correct_match, correct_score, matched_correct = best_variant_match(sentence, correct_answers)
    incorrect_match, incorrect_score, matched_incorrect = best_variant_match(sentence, incorrect_answers)
    if correct_match and incorrect_match:
        bucket = "mixed"
    elif incorrect_match and not correct_match:
        bucket = "contradicted"
    elif correct_match and not incorrect_match:
        bucket = "supported"
    else:
        bucket = "uncertain"
    return {
        "sentence": sentence,
        "bucket": bucket,
        "correct_score": correct_score,
        "incorrect_score": incorrect_score,
        "matched_correct": matched_correct,
        "matched_incorrect": matched_incorrect,
    }


def propose_onset(sentences):
    for idx, row in enumerate(sentences):
        if row["bucket"] in {"contradicted", "mixed"}:
            return idx
    for idx, row in enumerate(sentences):
        if row["bucket"] == "uncertain" and row["correct_score"] < 0.45:
            return idx
    return None


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.annotation_pack_jsonl))
    out_rows = []
    with_onset = 0
    for row in rows:
        sentences = [
            sentence_bucket(sentence, row.get("correct_answers", []), row.get("incorrect_answers", []))
            for sentence in split_sentences(row.get("generated_answer", ""))
        ]
        onset_index = propose_onset(sentences)
        if onset_index is not None:
            with_onset += 1
        out_rows.append(
            {
                "question_id": row["question_id"],
                "category": row["category"],
                "question": row["question"],
                "auto_bucket": row["auto_bucket"],
                "generated_answer": row["generated_answer"],
                "proposed_onset_sentence_index": onset_index,
                "proposed_onset_sentence": None if onset_index is None else sentences[onset_index]["sentence"],
                "sentences": sentences,
            }
        )

    summary = {
        "annotation_pack_jsonl": args.annotation_pack_jsonl,
        "n_rows": len(out_rows),
        "rows_with_proposed_onset": with_onset,
        "rows_without_proposed_onset": len(out_rows) - with_onset,
    }
    save_jsonl(Path(args.output_jsonl), out_rows)
    save_json(Path(args.summary_json), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved onset candidates to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
