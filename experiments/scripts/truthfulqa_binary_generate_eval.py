import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    bootstrap_accuracy_ci,
    ensure_truthfulqa_csv,
    get_decoder_layers,
    get_layer_write_modules,
    get_primary_device,
    load_model_and_tokenizer,
    load_truthfulqa_binary_items,
    make_binary_instance,
    maybe_cap_items,
    parse_int_list,
    save_json,
    split_calibration_eval,
    stable_hash,
    summarize_accuracy_line,
    summarize_category_accuracy,
)


LETTER_RE = re.compile(r"[AB]")


def parse_args():
    parser = argparse.ArgumentParser(description="TruthfulQA binary eval via greedy generation, with optional DoLa and HDA patch.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model id or local path")
    parser.add_argument("--truthfulqa-csv", default="experiments/data/TruthfulQA.csv", help="Path to TruthfulQA CSV")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--gpu-memory-gb", type=int, default=15, help="Per-GPU memory cap in GiB for model loading")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model with bitsandbytes 4-bit quantization")
    parser.add_argument("--seed", type=int, default=41, help="Random seed")
    parser.add_argument("--calibration-size", type=int, default=200, help="Calibration split size to hold out from final eval")
    parser.add_argument("--max-samples", type=int, default=0, help="Cap total rows before split (0 means no cap)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap rounds for CI")
    parser.add_argument("--max-new-tokens", type=int, default=4, help="Generation budget")
    parser.add_argument("--disable-thinking", action="store_true", help="Pass enable_thinking=False to tokenizer.apply_chat_template when supported")
    parser.add_argument("--dola", action="store_true", help="Enable DoLa custom_generate decoding")
    parser.add_argument("--dola-layers", default="high", help='DoLa layer selector: "high", "low", or comma-separated integers')
    parser.add_argument("--repetition-penalty", type=float, default=1.2, help="Repetition penalty for DoLa decoding")
    parser.add_argument("--directions", help="Optional directions npz for HDA patch")
    parser.add_argument("--layers", default="", help="Comma-separated layer indices for HDA patch")
    parser.add_argument("--alpha", type=float, default=1.0, help="HDA patch alpha")
    parser.add_argument("--modules", default="mlp", choices=["attn", "mlp", "both"], help="Which modules to patch")
    parser.add_argument("--output-json", default="experiments/artifacts/truthfulqa_binary_generate_eval.json", help="Where to save metrics")
    return parser.parse_args()


def parse_dola_layers(raw: str):
    raw = (raw or "").strip()
    if raw.lower() in {"high", "low"}:
        return raw.lower()
    values = parse_int_list(raw)
    if not values:
        raise ValueError(f"Invalid DoLa layer selector: {raw}")
    return values


@torch.no_grad()
def apply_rank_one_patch(weight: torch.Tensor, v_hat: torch.Tensor, alpha: float):
    v = v_hat.to(weight.device, dtype=weight.dtype)
    proj_row = torch.matmul(v.unsqueeze(0), weight)
    delta = alpha * torch.matmul(v.unsqueeze(1), proj_row)
    weight.sub_(delta)


def apply_patch_to_model(model, directions: np.ndarray, selected_layers, modules: str, alpha: float, device: torch.device):
    layers = get_decoder_layers(model)
    patch_log = []
    for layer_idx in selected_layers:
        if layer_idx < 0 or layer_idx >= len(layers):
            raise ValueError(f"Layer index out of bounds: {layer_idx}")
        if layer_idx >= directions.shape[0]:
            raise ValueError(f"Direction file has {directions.shape[0]} layers, cannot use {layer_idx}")

        v = directions[layer_idx]
        norm = float(np.linalg.norm(v))
        if norm <= 1e-12:
            patch_log.append({"layer": layer_idx, "patched_modules": [], "skipped": True, "reason": "near-zero direction norm"})
            continue

        v_hat = torch.tensor(v / norm, dtype=torch.float32, device=device)
        patched_names = []
        for module_name, module in get_layer_write_modules(layers[layer_idx], modules):
            apply_rank_one_patch(module.weight.data, v_hat, alpha)
            patched_names.append(module_name)
        patch_log.append({"layer": layer_idx, "patched_modules": patched_names, "skipped": len(patched_names) == 0})
    return patch_log


def build_generate_kwargs(args):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": None,
        "eos_token_id": None,
    }
    if args.dola:
        kwargs.update(
            {
                "custom_generate": "transformers-community/dola",
                "trust_remote_code": True,
                "dola_layers": parse_dola_layers(args.dola_layers),
                "repetition_penalty": args.repetition_penalty,
                "output_hidden_states": True,
                "return_dict_in_generate": False,
            }
        )
    return kwargs


def extract_letter(text: str):
    match = LETTER_RE.search((text or "").upper())
    if match:
        return match.group(0)
    return "?"


def generate_letter(model, tokenizer, device, prompt: str, generate_kwargs: dict):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    kwargs = dict(generate_kwargs)
    kwargs["pad_token_id"] = tokenizer.eos_token_id
    kwargs["eos_token_id"] = tokenizer.eos_token_id
    with torch.no_grad():
        output_ids = model.generate(**encoded, **kwargs)
    gen_ids = output_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return extract_letter(text), text


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    csv_path = ensure_truthfulqa_csv(Path(args.truthfulqa_csv), download_if_missing=True)
    items = load_truthfulqa_binary_items(csv_path)
    items = maybe_cap_items(items, args.max_samples)
    _, eval_items = split_calibration_eval(items, args.calibration_size, args.seed)
    if not eval_items:
        eval_items = items

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=args.load_in_4bit,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    device = get_primary_device(model)

    patch_log = []
    selected_layers = parse_int_list(args.layers) if args.layers else []
    if args.directions:
        if not selected_layers:
            raise ValueError("--layers is required when --directions is provided")
        direction_data = np.load(args.directions)
        directions = direction_data["directions"]
        patch_log = apply_patch_to_model(
            model=model,
            directions=directions,
            selected_layers=selected_layers,
            modules=args.modules,
            alpha=args.alpha,
            device=device,
        )

    generate_kwargs = build_generate_kwargs(args)
    rows = []
    y_true = []
    y_pred = []
    for item in tqdm(eval_items, desc="Binary generate eval"):
        row_rng = random.Random(args.seed + stable_hash(item.question))
        prompt, correct, a_text, b_text = make_binary_instance(
            item,
            row_rng,
            tokenizer,
            enable_thinking=False if args.disable_thinking else None,
        )
        pred, raw_text = generate_letter(model, tokenizer, device, prompt, generate_kwargs)
        y_true.append(1 if correct == "A" else 0)
        y_pred.append(1 if pred == "A" else 0)
        rows.append(
            {
                "category": item.category,
                "question": item.question,
                "correct": correct,
                "pred": pred,
                "raw_generation": raw_text,
                "choice_A": a_text,
                "choice_B": b_text,
            }
        )

    acc, lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=args.bootstrap, seed=args.seed)
    out = {
        "model": args.model,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "gpu_memory_gb": args.gpu_memory_gb,
        "seed": args.seed,
        "n_eval": len(eval_items),
        "calibration_size": args.calibration_size,
        "max_new_tokens": args.max_new_tokens,
        "disable_thinking": args.disable_thinking,
        "dola": args.dola,
        "dola_layers": parse_dola_layers(args.dola_layers) if args.dola else None,
        "repetition_penalty": args.repetition_penalty if args.dola else None,
        "patch": {
            "enabled": bool(args.directions),
            "directions": args.directions,
            "layers": selected_layers,
            "alpha": args.alpha if args.directions else None,
            "modules": args.modules if args.directions else None,
            "patch_log": patch_log,
        },
        "accuracy": acc,
        "ci95": [lo, hi],
        "category_accuracy": summarize_category_accuracy(rows),
        "rows": rows,
    }
    save_json(Path(args.output_json), out)
    label = "binary_generate_dola" if args.dola else "binary_generate"
    if args.directions:
        label += "_patched"
    print(summarize_accuracy_line(label, acc, lo, hi, len(eval_items)))
    print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
