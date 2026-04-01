import argparse
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_open_answer_prompt,
    get_decoder_layers,
    get_primary_device,
    load_jsonl,
    load_model_and_tokenizer,
    parse_int_list,
    save_json,
)
from truthfulqa_open_generation_eval import classify_bucket, summarize_bucket_rows


class HookContext:
    def __init__(self, edit_generation_only: bool, max_edited_tokens: int):
        self.edit_generation_only = edit_generation_only
        self.max_edited_tokens = max_edited_tokens
        self.generated_token_edits = 0

    def reset(self):
        self.generated_token_edits = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Open-generation mechanistic intervention eval")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument(
        "--questions-jsonl",
        default="experiments/data/mechanistic_open/annotation_seed.jsonl",
        help="Prepared question JSONL",
    )
    parser.add_argument(
        "--directions-npz",
        required=True,
        help="NPZ with layerwise direction arrays",
    )
    parser.add_argument(
        "--direction-key",
        default="unsupported_minus_supported",
        help="Array key inside the NPZ",
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer list")
    parser.add_argument("--beta", type=float, default=0.5, help="Projection removal strength")
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
        "--max-new-tokens",
        type=int,
        default=48,
        help="Generation budget per prompt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional row cap",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--edit-generation-only",
        action="store_true",
        help="Only edit generation-time single-token forward passes, not the prompt prefill",
    )
    parser.add_argument(
        "--max-edited-tokens",
        type=int,
        default=0,
        help="Limit edits to the first N generated tokens (0 means no limit)",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/mechanistic_open_intervention_eval.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def build_projection_hook(basis: torch.Tensor, beta: float, ctx: HookContext):
    def _hook(_module, _inputs, output):
        def _edit(hidden: torch.Tensor) -> torch.Tensor:
            if hidden.ndim != 3 or hidden.size(1) <= 0:
                return hidden
            if ctx.edit_generation_only:
                if hidden.size(1) != 1:
                    return hidden
                if ctx.max_edited_tokens > 0 and ctx.generated_token_edits >= ctx.max_edited_tokens:
                    return hidden
            edited = hidden.clone()
            token_vec = edited[:, -1, :]
            basis_local = basis.to(device=token_vec.device, dtype=token_vec.dtype)
            if basis_local.ndim == 1:
                basis_local = basis_local.unsqueeze(0)
            coeffs = torch.matmul(token_vec, basis_local.transpose(0, 1))
            projected = torch.matmul(coeffs, basis_local)
            edited[:, -1, :] = token_vec - beta * projected
            if ctx.edit_generation_only and hidden.size(1) == 1:
                ctx.generated_token_edits += 1
            return edited

        if isinstance(output, tuple):
            first = _edit(output[0])
            return (first,) + output[1:]
        return _edit(output)

    return _hook


def generate_answer(model, tokenizer, device, prompt: str, max_new_tokens: int, ctx: HookContext) -> str:
    ctx.reset()
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    output_ids = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def make_random_directions(direction_matrix: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal(direction_matrix.shape).astype(np.float32)
    return random_matrix


def normalize_direction(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def attach_hooks(layers, layer_indices, direction_matrix: np.ndarray, beta: float, ctx: HookContext):
    handles = []
    for layer_idx in layer_indices:
        layer_obj = direction_matrix[layer_idx]
        if layer_obj.ndim == 1:
            vec = normalize_direction(layer_obj)
            if float(np.linalg.norm(vec)) <= 1e-12:
                continue
            basis = torch.tensor(vec, dtype=torch.float32)
        else:
            rows = []
            for row in layer_obj:
                row = normalize_direction(row)
                if float(np.linalg.norm(row)) <= 1e-12:
                    continue
                rows.append(row)
            if not rows:
                continue
            basis = torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)
        handle = layers[layer_idx].register_forward_hook(
            build_projection_hook(basis, beta, ctx)
        )
        handles.append(handle)
    return handles


def summarize_condition(rows):
    bucket_summary = summarize_bucket_rows(rows)
    n = max(1, len(rows))
    hard_bad = sum(
        1 for row in rows if row["bucket"] in {"contradicted_reference", "mixed_or_self_contradictory"}
    )
    unresolved = sum(1 for row in rows if row["bucket"] == "unresolved_needs_annotation")
    supported = sum(1 for row in rows if row["bucket"] in {"supported_answer", "supported_abstention"})
    abstention = sum(1 for row in rows if row["abstains"])
    return {
        "n": len(rows),
        "bucket_summary": bucket_summary,
        "hard_bad_rate": float(hard_bad / n),
        "unresolved_rate": float(unresolved / n),
        "supported_rate": float(supported / n),
        "abstention_rate": float(abstention / n),
    }


def run_condition(name, model, tokenizer, device, questions, max_new_tokens, hook_ctx, edit_ctx: HookContext):
    rows = []
    with hook_ctx:
        for row in tqdm(questions, desc=f"Open eval [{name}]"):
            prompt = build_open_answer_prompt(tokenizer, row["question"])
            answer_text = generate_answer(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max_new_tokens,
                ctx=edit_ctx,
            )
            (
                bucket,
                abstains,
                correct_match,
                incorrect_match,
                correct_score,
                incorrect_score,
                matched_correct,
                matched_incorrect,
            ) = classify_bucket(row, answer_text)
            rows.append(
                {
                    "question_id": row["question_id"],
                    "category": row["category"],
                    "question": row["question"],
                    "answer_text": answer_text,
                    "bucket": bucket,
                    "abstains": abstains,
                    "correct_match": correct_match,
                    "incorrect_match": incorrect_match,
                    "correct_score": correct_score,
                    "incorrect_score": incorrect_score,
                    "matched_correct": matched_correct,
                    "matched_incorrect": matched_incorrect,
                }
            )
    return rows, summarize_condition(rows)


def main():
    args = parse_args()
    layer_indices = parse_int_list(args.layers)
    if not layer_indices:
        raise ValueError("No layers selected.")

    questions = load_jsonl(Path(args.questions_jsonl))
    if args.max_samples > 0:
        questions = questions[: args.max_samples]
    if not questions:
        raise ValueError("No questions loaded.")

    direction_npz = np.load(args.directions_npz)
    if args.direction_key not in direction_npz:
        raise KeyError(f"Direction key '{args.direction_key}' not found in {args.directions_npz}")
    direction_matrix = np.asarray(direction_npz[args.direction_key], dtype=np.float32)
    random_matrix = make_random_directions(direction_matrix, args.seed)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)
    target_ctx = HookContext(
        edit_generation_only=args.edit_generation_only,
        max_edited_tokens=args.max_edited_tokens,
    )
    random_ctx = HookContext(
        edit_generation_only=args.edit_generation_only,
        max_edited_tokens=args.max_edited_tokens,
    )
    base_ctx = HookContext(edit_generation_only=False, max_edited_tokens=0)

    for layer_idx in layer_indices:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= direction_matrix.shape[0]:
            raise ValueError(f"Direction file has only {direction_matrix.shape[0]} layers.")

    base_rows, base_summary = run_condition(
        "base",
        model,
        tokenizer,
        device,
        questions,
        max_new_tokens=args.max_new_tokens,
        hook_ctx=nullcontext(),
        edit_ctx=base_ctx,
    )

    target_handles = attach_hooks(layers, layer_indices, direction_matrix, args.beta, target_ctx)
    try:
        target_rows, target_summary = run_condition(
            "target",
            model,
            tokenizer,
            device,
            questions,
            max_new_tokens=args.max_new_tokens,
            hook_ctx=nullcontext(),
            edit_ctx=target_ctx,
        )
    finally:
        for handle in target_handles:
            handle.remove()

    random_handles = attach_hooks(layers, layer_indices, random_matrix, args.beta, random_ctx)
    try:
        random_rows, random_summary = run_condition(
            "random",
            model,
            tokenizer,
            device,
            questions,
            max_new_tokens=args.max_new_tokens,
            hook_ctx=nullcontext(),
            edit_ctx=random_ctx,
        )
    finally:
        for handle in random_handles:
            handle.remove()

    result = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "questions_jsonl": args.questions_jsonl,
        "n_eval": len(questions),
        "directions_npz": args.directions_npz,
        "direction_key": args.direction_key,
        "layers": layer_indices,
        "beta": args.beta,
        "seed": args.seed,
        "edit_generation_only": args.edit_generation_only,
        "max_edited_tokens": args.max_edited_tokens,
        "base": base_summary,
        "target": target_summary,
        "random": random_summary,
        "delta": {
            "target_hard_bad_rate": target_summary["hard_bad_rate"] - base_summary["hard_bad_rate"],
            "target_supported_rate": target_summary["supported_rate"] - base_summary["supported_rate"],
            "target_unresolved_rate": target_summary["unresolved_rate"] - base_summary["unresolved_rate"],
            "random_hard_bad_rate": random_summary["hard_bad_rate"] - base_summary["hard_bad_rate"],
            "random_supported_rate": random_summary["supported_rate"] - base_summary["supported_rate"],
            "random_unresolved_rate": random_summary["unresolved_rate"] - base_summary["unresolved_rate"],
        },
        "rows": {
            "base": base_rows,
            "target": target_rows,
            "random": random_rows,
        },
    }
    save_json(Path(args.output_json), result)
    print(json.dumps(result["delta"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
