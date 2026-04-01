import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import load_jsonl, load_model_and_tokenizer, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline generation and capture early-answer residuals")
    parser.add_argument(
        "--data-jsonl",
        required=True,
        help="HERETIC-simple split JSONL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
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
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation budget per prompt",
    )
    parser.add_argument(
        "--capture-answer-tokens",
        type=int,
        default=5,
        help="How many generated answer tokens to summarize",
    )
    parser.add_argument("--seed", type=int, default=7, help="Generation seed")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional row cap")
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Row-wise generation output JSONL",
    )
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Residual NPZ output path",
    )
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Capture metadata JSON path",
    )
    return parser.parse_args()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def entropy_from_scores(scores: torch.Tensor) -> float:
    probs = torch.softmax(scores, dim=-1)
    log_probs = torch.log_softmax(scores, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return float(entropy.item())


def build_prompt(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": row["system_message"]},
            {"role": "user", "content": row["user_message"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate_one(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]

    do_sample = temperature > 0.0
    generate_kwargs = dict(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs["temperature"] = max(temperature, 1e-5)
        generate_kwargs["top_p"] = top_p
    output = model.generate(**generate_kwargs)
    full_ids = output.sequences[0]
    gen_ids = full_ids[prompt_len:]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    first_entropy = float("nan")
    mean_entropy = float("nan")
    if output.scores:
        entropies = [entropy_from_scores(score) for score in output.scores]
        first_entropy = entropies[0]
        mean_entropy = float(np.mean(entropies))

    return {
        "prompt_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "full_ids": full_ids.detach().cpu().tolist(),
        "output_ids": gen_ids.detach().cpu().tolist(),
        "prompt_len": prompt_len,
        "answer_text": answer_text,
        "first_step_entropy": first_entropy,
        "mean_step_entropy": mean_entropy,
    }


@torch.no_grad()
def capture_answer_states(model, input_ids: list[int], prompt_len: int, capture_answer_tokens: int):
    device = next(model.parameters()).device
    if len(input_ids) <= prompt_len:
        return None, None, 0

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    outputs = model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)

    start = prompt_len
    end = min(len(input_ids), prompt_len + capture_answer_tokens)
    token_count = max(0, end - start)
    if token_count <= 0:
        return None, None, 0

    token1_vectors = []
    token1_to_k_vectors = []
    for hs in outputs.hidden_states[1:]:
        token1_vectors.append(hs[0, start, :].float().cpu().numpy())
        token1_to_k_vectors.append(hs[0, start:end, :].mean(dim=0).float().cpu().numpy())

    return (
        np.stack(token1_vectors, axis=0).astype(np.float16),
        np.stack(token1_to_k_vectors, axis=0).astype(np.float16),
        token_count,
    )


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.data_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No HERETIC-simple rows loaded.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )

    generated_rows = []
    answer_token_1 = []
    answer_token_1_to_5_mean = []
    answer_token_counts = []

    for row in tqdm(rows, desc="HERETIC-simple generation capture"):
        prompt = build_prompt(tokenizer, row)
        generated = generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        token1_state, token1_to_k_state, token_count = capture_answer_states(
            model,
            generated["full_ids"],
            generated["prompt_len"],
            args.capture_answer_tokens,
        )

        if token1_state is None:
            n_layers = len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else None
            if n_layers is None:
                raise ValueError("Unable to infer layer count for empty generation fallback.")
            hidden_size = int(model.config.hidden_size)
            token1_state = np.zeros((n_layers, hidden_size), dtype=np.float16)
            token1_to_k_state = np.zeros((n_layers, hidden_size), dtype=np.float16)

        answer_token_1.append(token1_state)
        answer_token_1_to_5_mean.append(token1_to_k_state)
        answer_token_counts.append(token_count)

        generated_rows.append(
            {
                "question_id": row["question_id"],
                "prompt_id": row["prompt_id"],
                "pair_group": row["pair_group"],
                "split": row["split"],
                "bucket": row["bucket"],
                "binary_bucket": row["binary_bucket"],
                "expected_behavior": row["expected_behavior"],
                "category": row["category"],
                "question": row["question"],
                "context": row["context"],
                "reference_answer": row["reference_answer"],
                "reference_notes": row["reference_notes"],
                "prompt_text": prompt,
                "prompt_ids": generated["prompt_ids"],
                "output_text": generated["answer_text"],
                "output_ids": generated["output_ids"],
                "first_answer_token_index": generated["prompt_len"],
                "captured_answer_token_count": token_count,
                "first_step_entropy": generated["first_step_entropy"],
                "mean_step_entropy": generated["mean_step_entropy"],
                "generation_config": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": args.temperature > 0.0,
                    "seed": args.seed,
                },
            }
        )

    answer_token_1 = np.stack(answer_token_1, axis=0)
    answer_token_1_to_5_mean = np.stack(answer_token_1_to_5_mean, axis=0)
    answer_token_counts = np.asarray(answer_token_counts, dtype=np.int32)

    output_jsonl = Path(args.output_jsonl)
    ensure_parent(output_jsonl)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in generated_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_npz = Path(args.output_npz)
    ensure_parent(output_npz)
    np.savez_compressed(
        output_npz,
        answer_token_1=answer_token_1,
        answer_token_1_to_5_mean=answer_token_1_to_5_mean,
        answer_token_count=answer_token_counts,
    )

    metadata = {
        "data_jsonl": args.data_jsonl,
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "n_rows": len(rows),
        "capture_answer_tokens": args.capture_answer_tokens,
        "generation_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "outputs": {
            "jsonl": str(output_jsonl),
            "npz": str(output_npz),
        },
        "summary": {
            "mean_captured_answer_token_count": float(answer_token_counts.mean()) if len(answer_token_counts) > 0 else 0.0,
            "rows_with_zero_generated_tokens": int((answer_token_counts == 0).sum()),
            "mean_first_step_entropy": float(
                np.nanmean([row["first_step_entropy"] for row in generated_rows])
            ),
            "mean_generation_entropy": float(
                np.nanmean([row["mean_step_entropy"] for row in generated_rows])
            ),
        },
    }
    save_json(Path(args.metadata_json), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
