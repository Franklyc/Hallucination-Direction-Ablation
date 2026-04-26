import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from common import build_chat_prompt, get_primary_device, load_jsonl, load_model_and_tokenizer, save_json


REFUSAL_MARKERS = [
    "i can't",
    "i cannot",
    "i'm unable",
    "i am unable",
    "sorry, but",
    "cannot help with that",
    "can't help with that",
    "i won’t",
    "i won't",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate benign output drift on a prompt set.")
    parser.add_argument("--model", required=True, help="HF model id or local model directory")
    parser.add_argument("--drift-jsonl", required=True, help="Path to the benign drift prompt set")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Generation budget")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model for similarity",
    )
    parser.add_argument("--output-json", required=True, help="Where to save drift rows and summary")
    return parser.parse_args()


def generate_text(model, tokenizer, device: torch.device, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def safe_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    return int(len(tokenizer(text, add_special_tokens=False)["input_ids"]))


def bullet_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("-", "*")):
            count += 1
        elif len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in {". ", ") "}:
            count += 1
    return count


def parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def format_pass(row: dict, text: str):
    fmt = row.get("format_type", "paragraph")
    parsed = None
    if fmt == "paragraph":
        return bool(text.strip())
    if fmt == "bullet_count":
        expected = int(row.get("expected_bullet_count", 0))
        count = bullet_count(text)
        return count == expected if expected > 0 else count > 0
    if fmt == "json_object":
        parsed = parse_json(text)
        if not isinstance(parsed, dict):
            return False
        expected_keys = row.get("expected_keys") or []
        return all(key in parsed for key in expected_keys)
    if fmt == "json_array":
        parsed = parse_json(text)
        if not isinstance(parsed, list):
            return False
        expected_length = int(row.get("expected_list_length", 0))
        return len(parsed) == expected_length if expected_length > 0 else len(parsed) > 0
    if fmt == "markdown_table":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) < 3:
            return False
        pipe_lines = [line for line in lines if "|" in line]
        if len(pipe_lines) < 3:
            return False
        return any("---" in line or ":--" in line for line in pipe_lines[:2])
    if fmt == "title_only":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) != 1:
            return False
        return len(lines[0].split()) <= 12
    return bool(text.strip())


def exact_field_match(row: dict, text: str):
    expected = row.get("exact_fields")
    if not expected:
        return None
    parsed = parse_json(text)
    if not isinstance(parsed, dict):
        return False
    for key, value in expected.items():
        if str(parsed.get(key, "")).strip() != str(value).strip():
            return False
    return True


def is_refusal(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.drift_jsonl))
    if not rows:
        raise ValueError("No drift prompts loaded.")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    generated = []
    for row in tqdm(rows, desc="Benign drift generation"):
        prompt = build_chat_prompt(
            tokenizer=tokenizer,
            system_message="You are a helpful and concise assistant.",
            user_message=row["prompt_text"],
        )
        text = generate_text(model, tokenizer, device, prompt, max_new_tokens=args.max_new_tokens)
        out = dict(row)
        out["response_text"] = text
        out["token_count"] = safe_token_count(tokenizer, text)
        out["format_pass"] = bool(format_pass(row, text))
        out["exact_field_match"] = exact_field_match(row, text)
        out["is_refusal"] = bool(is_refusal(text))
        generated.append(out)

    encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(args.embedding_model, device=encoder_device)
    embeddings = encoder.encode(
        [row["response_text"] or "" for row in generated],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "drift_jsonl": args.drift_jsonl,
        "embedding_model": args.embedding_model,
        "n": len(generated),
        "family_counts": {},
        "responses": generated,
        "embeddings": embeddings.tolist(),
    }
    for row in generated:
        family = row["task_family"]
        summary["family_counts"][family] = summary["family_counts"].get(family, 0) + 1

    save_json(Path(args.output_json), summary)
    print(f"Saved benign drift generations to: {args.output_json}")


if __name__ == "__main__":
    main()
