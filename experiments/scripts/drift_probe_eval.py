import argparse
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from common import (
    build_chat_prompt,
    get_decoder_layers,
    get_primary_device,
    load_model_and_tokenizer,
    parse_int_list,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Drift proxy eval for activation probe")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HF model id or local path",
    )
    parser.add_argument("--directions", required=True, help="Path to directions npz")
    parser.add_argument(
        "--drift-jsonl",
        default="experiments/data/prepared/drift_benign.jsonl",
        help="Path to benign drift prompts jsonl",
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
        "--layers",
        required=True,
        help="Comma-separated layer indices (e.g., 20,21,22)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Projection removal strength",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation budget per prompt",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional prompt cap (0 means all)",
    )
    parser.add_argument(
        "--output-json",
        default="experiments/artifacts/drift_probe_eval.json",
        help="Where to save drift metrics",
    )
    return parser.parse_args()


@dataclass
class ProbeContext:
    prompt_len: int = 0
    edited_once: bool = False


def load_prompts(path: Path, max_samples: int) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt_text" not in row:
                continue
            rows.append(row)

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def make_projection_hook(v: torch.Tensor, beta: float, ctx: ProbeContext):
    def _hook(_module, _inputs, output):
        if ctx.prompt_len <= 0:
            return output
        if ctx.edited_once:
            return output

        def _edit(hidden: torch.Tensor) -> torch.Tensor:
            if hidden.size(1) <= 0:
                return hidden
            # Only edit when full prompt context is present.
            if hidden.size(1) < ctx.prompt_len:
                return hidden
            idx = min(ctx.prompt_len - 1, hidden.size(1) - 1)
            edited = hidden.clone()
            token_vec = edited[:, idx, :]

            v_local = v.to(device=token_vec.device, dtype=token_vec.dtype)
            proj = (token_vec * v_local.unsqueeze(0)).sum(dim=-1, keepdim=True)
            edited[:, idx, :] = token_vec - beta * proj * v_local.unsqueeze(0)
            ctx.edited_once = True
            return edited

        if isinstance(output, tuple):
            first = _edit(output[0])
            return (first,) + output[1:]
        return _edit(output)

    return _hook


def generate_text(model, tokenizer, device: torch.device, prompt: str, max_new_tokens: int, ctx: ProbeContext) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    ctx.prompt_len = int(prompt_len)
    ctx.edited_once = False

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    ctx.prompt_len = 0
    ctx.edited_once = False
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def safe_token_count(tokenizer, text: str) -> int:
    if not text:
        return 0
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(len(ids))


def main():
    args = parse_args()

    prompt_rows = load_prompts(Path(args.drift_jsonl), args.max_samples)
    if not prompt_rows:
        raise ValueError("No valid drift prompts loaded.")

    selected_layers = parse_int_list(args.layers)
    if not selected_layers:
        raise ValueError("No layers selected.")

    direction_data = np.load(args.directions)
    directions = direction_data["directions"]

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)
    layers = get_decoder_layers(model)

    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(
                f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}"
            )

    context = ProbeContext(prompt_len=0)
    hook_handles = []
    for layer_idx in selected_layers:
        v_np = directions[layer_idx]
        norm = np.linalg.norm(v_np)
        if norm <= 1e-12:
            continue
        v = torch.tensor(v_np / norm, dtype=torch.float32)
        handle = layers[layer_idx].register_forward_hook(
            make_projection_hook(v=v, beta=args.beta, ctx=context)
        )
        hook_handles.append(handle)

    rows = []
    base_ratios = []
    for row in tqdm(prompt_rows, desc="Drift eval"):
        prompt = build_chat_prompt(
            tokenizer,
            system_message="You are a helpful and concise assistant.",
            user_message=row["prompt_text"],
        )

        for h in hook_handles:
            h.remove()
        hook_handles = []
        base_text = generate_text(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=args.max_new_tokens,
            ctx=context,
        )

        for layer_idx in selected_layers:
            v_np = directions[layer_idx]
            norm = np.linalg.norm(v_np)
            if norm <= 1e-12:
                continue
            v = torch.tensor(v_np / norm, dtype=torch.float32)
            handle = layers[layer_idx].register_forward_hook(
                make_projection_hook(v=v, beta=args.beta, ctx=context)
            )
            hook_handles.append(handle)

        probe_text = generate_text(
            model,
            tokenizer,
            device,
            prompt,
            max_new_tokens=args.max_new_tokens,
            ctx=context,
        )

        ratio = float(SequenceMatcher(None, base_text, probe_text).ratio())
        base_tokens = safe_token_count(tokenizer, base_text)
        probe_tokens = safe_token_count(tokenizer, probe_text)

        base_ratios.append(ratio)
        rows.append(
            {
                "prompt_id": row.get("prompt_id", ""),
                "prompt_text": row["prompt_text"],
                "base_text": base_text,
                "probe_text": probe_text,
                "similarity_ratio": ratio,
                "exact_match": base_text == probe_text,
                "base_token_count": base_tokens,
                "probe_token_count": probe_tokens,
                "token_count_delta": probe_tokens - base_tokens,
            }
        )

    for h in hook_handles:
        h.remove()

    ratios = np.asarray([r["similarity_ratio"] for r in rows], dtype=np.float64)
    token_deltas = np.asarray([r["token_count_delta"] for r in rows], dtype=np.float64)
    exact_match_rate = float(np.mean([1.0 if r["exact_match"] else 0.0 for r in rows]))

    sorted_rows = sorted(rows, key=lambda x: x["similarity_ratio"])
    summary = {
        "n": len(rows),
        "mean_similarity_ratio": float(ratios.mean()),
        "median_similarity_ratio": float(np.median(ratios)),
        "min_similarity_ratio": float(ratios.min()),
        "exact_match_rate": exact_match_rate,
        "mean_token_count_delta": float(token_deltas.mean()),
        "median_token_count_delta": float(np.median(token_deltas)),
        "max_abs_token_count_delta": float(np.max(np.abs(token_deltas))),
    }

    out = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "directions": args.directions,
        "layers": selected_layers,
        "beta": args.beta,
        "drift_jsonl": args.drift_jsonl,
        "max_new_tokens": args.max_new_tokens,
        "summary": summary,
        "lowest_similarity_examples": sorted_rows[:10],
        "rows": rows,
    }
    save_json(Path(args.output_json), out)

    print(
        "drift summary: "
        f"mean_similarity={summary['mean_similarity_ratio']:.4f} "
        f"exact_match_rate={summary['exact_match_rate']:.4f} "
        f"mean_token_delta={summary['mean_token_count_delta']:.4f}"
    )
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
