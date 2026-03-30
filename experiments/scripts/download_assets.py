import argparse
import json
from pathlib import Path

import requests


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_TRUTHFULQA_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Download model and TruthfulQA assets")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--model-dir",
        default="artifacts/models/Qwen3-4B-Instruct-2507",
        help="Local model snapshot directory",
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download full model snapshot",
    )
    parser.add_argument(
        "--truthfulqa-url",
        default=DEFAULT_TRUTHFULQA_URL,
        help="Official TruthfulQA CSV URL",
    )
    parser.add_argument(
        "--truthfulqa-csv",
        default="experiments/data/TruthfulQA.csv",
        help="Local CSV output path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--allow-hf-fallback",
        action="store_true",
        help=(
            "If official CSV download fails, save EleutherAI/truthful_qa_binary "
            "validation split as JSONL fallback"
        ),
    )
    parser.add_argument(
        "--report-json",
        default="experiments/artifacts/download_report.json",
        help="Where to save download summary",
    )
    return parser.parse_args()


def download_truthfulqa_csv(url: str, out_path: Path, timeout: int, force: bool) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return {
            "status": "skipped",
            "reason": "already_exists",
            "path": str(out_path),
        }

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return {
        "status": "downloaded",
        "path": str(out_path),
        "bytes": len(response.content),
        "url": url,
    }


def download_model_snapshot(model_id: str, model_dir: Path, force: bool) -> dict:
    if model_dir.exists() and any(model_dir.iterdir()) and not force:
        return {
            "status": "skipped",
            "reason": "already_exists",
            "path": str(model_dir),
        }

    model_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
    )
    return {
        "status": "downloaded",
        "repo_id": model_id,
        "path": str(local_path),
    }


def write_hf_fallback_jsonl(path: Path) -> dict:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/truthful_qa_binary", "multiple_choice")
    val = ds["validation"]
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "status": "downloaded",
        "path": str(path),
        "rows": len(val),
        "dataset": "EleutherAI/truthful_qa_binary:multiple_choice/validation",
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    report = {
        "model": {
            "requested": args.download_model,
            "result": None,
        },
        "truthfulqa": {
            "result": None,
            "fallback": None,
        },
    }

    if args.download_model:
        report["model"]["result"] = download_model_snapshot(
            args.model_id,
            Path(args.model_dir),
            force=args.force,
        )
    else:
        report["model"]["result"] = {"status": "skipped", "reason": "not_requested"}

    csv_error = None
    try:
        report["truthfulqa"]["result"] = download_truthfulqa_csv(
            args.truthfulqa_url,
            Path(args.truthfulqa_csv),
            timeout=args.timeout,
            force=args.force,
        )
    except Exception as exc:
        csv_error = str(exc)
        report["truthfulqa"]["result"] = {
            "status": "failed",
            "error": csv_error,
        }

    if csv_error and args.allow_hf_fallback:
        fallback_path = Path("experiments/data/truthful_qa_binary_validation.jsonl")
        report["truthfulqa"]["fallback"] = write_hf_fallback_jsonl(fallback_path)
    elif csv_error:
        report["truthfulqa"]["fallback"] = {
            "status": "skipped",
            "reason": "csv_failed_and_fallback_not_enabled",
        }
    else:
        report["truthfulqa"]["fallback"] = {
            "status": "skipped",
            "reason": "csv_succeeded",
        }

    save_report(Path(args.report_json), report)

    print("Download summary:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report to: {args.report_json}")


if __name__ == "__main__":
    main()
