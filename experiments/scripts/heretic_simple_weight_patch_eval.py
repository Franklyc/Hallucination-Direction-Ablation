import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import get_decoder_layers, get_layer_write_modules, load_jsonl, load_model_and_tokenizer, save_json
from heretic_simple_runtime_eval import build_prompt, classify_output, summarize_rows


def parse_args():
    parser = argparse.ArgumentParser(description="HERETIC-simple weight patch evaluation")
    parser.add_argument(
        "--data-jsonl",
        required=True,
        help="HERETIC-simple dev/test JSONL",
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
        "--directions-npz",
        required=True,
        help="Direction or control NPZ",
    )
    parser.add_argument(
        "--direction-key",
        default="non_direct_minus_direct__normalized",
        help="Array key inside NPZ",
    )
    parser.add_argument(
        "--control-direction-key",
        default="",
        help="Optional control direction key for side-by-side patch eval",
    )
    parser.add_argument("--layers", required=True, help="Comma-separated layer indices")
    parser.add_argument("--alpha", type=float, default=0.2, help="Patch strength")
    parser.add_argument(
        "--modules",
        default="attn",
        choices=["attn", "mlp", "both"],
        help="Which write modules to patch",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Generation budget per prompt",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Optional row cap")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output JSON path",
    )
    return parser.parse_args()


def parse_layers(raw: str):
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def normalize_basis(layer_obj: np.ndarray) -> np.ndarray:
    if layer_obj.ndim == 1:
        norm = float(np.linalg.norm(layer_obj))
        if norm <= 1e-12:
            return np.zeros((0, layer_obj.shape[0]), dtype=np.float32)
        return (layer_obj[None, :] / norm).astype(np.float32)
    norms = np.linalg.norm(layer_obj, axis=-1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return (layer_obj / norms).astype(np.float32)


@torch.no_grad()
def apply_subspace_patch(weight: torch.Tensor, basis: torch.Tensor, alpha: float):
    local_basis = basis.to(device=weight.device, dtype=weight.dtype)
    proj_rows = torch.matmul(local_basis, weight)
    delta = torch.matmul(local_basis.transpose(0, 1), proj_rows)
    weight.sub_(alpha * delta)


def snapshot_weights_and_norms(layers, layer_indices, modules_mode: str):
    snapshot = {}
    row_norms = {}
    for layer_idx in layer_indices:
        for module_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            key = (layer_idx, module_name)
            tensor = module.weight.data.detach().clone()
            snapshot[key] = tensor
            row_norms[key] = tensor.norm(dim=1).float().cpu().numpy()
    return snapshot, row_norms


def restore_weights(layers, modules_mode: str, snapshot):
    for (layer_idx, module_name), tensor in snapshot.items():
        for current_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            if current_name == module_name:
                module.weight.data.copy_(tensor)
                break


def row_norm_delta(before, after):
    rows = []
    for key, before_vals in before.items():
        after_vals = after[key]
        rows.append(
            {
                "layer": key[0],
                "module": key[1],
                "mean_row_norm_before": float(before_vals.mean()),
                "mean_row_norm_after": float(after_vals.mean()),
                "mean_row_norm_delta": float(after_vals.mean() - before_vals.mean()),
                "max_abs_row_norm_delta": float(np.abs(after_vals - before_vals).max()),
            }
        )
    return rows


@torch.no_grad()
def prompt_final_hidden_norm(model, tokenizer, prompt: str) -> float:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded, output_hidden_states=True, use_cache=False)
    last_hidden = outputs.hidden_states[-1][0, -1, :]
    return float(last_hidden.norm().item())


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_len = encoded["input_ids"].shape[1]
    output = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    full_ids = output.sequences[0]
    gen_ids = full_ids[prompt_len:]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    entropies = []
    for score in output.scores:
        probs = torch.softmax(score, dim=-1)
        log_probs = torch.log_softmax(score, dim=-1)
        entropies.append(float((-(probs * log_probs).sum(dim=-1).mean()).item()))

    return {
        "answer_text": answer_text,
        "output_ids": gen_ids.detach().cpu().tolist(),
        "output_length": int(len(gen_ids)),
        "first_step_entropy": float(entropies[0]) if entropies else float("nan"),
        "mean_generation_entropy": float(np.mean(entropies)) if entropies else float("nan"),
        "prompt_final_hidden_norm": prompt_final_hidden_norm(model, tokenizer, prompt),
    }


def evaluate_condition(name, model, tokenizer, rows, max_new_tokens):
    evaluated = []
    for row in tqdm(rows, desc=f"HERETIC-simple patch eval [{name}]"):
        prompt = build_prompt(tokenizer, row)
        generated = generate_answer(model, tokenizer, prompt, max_new_tokens)
        classified = classify_output(row, generated["answer_text"])
        evaluated.append(
            {
                "question_id": row["question_id"],
                "prompt_id": row["prompt_id"],
                "bucket": row["bucket"],
                "binary_bucket": row["binary_bucket"],
                "expected_behavior": row["expected_behavior"],
                "question": row["question"],
                "output_ids": generated["output_ids"],
                "output_length": generated["output_length"],
                "first_step_entropy": generated["first_step_entropy"],
                "mean_generation_entropy": generated["mean_generation_entropy"],
                "prompt_final_hidden_norm": generated["prompt_final_hidden_norm"],
                **classified,
            }
        )
    return evaluated, summarize_rows(evaluated)


def changed_rate(base_rows, other_rows):
    if not base_rows:
        return 0.0
    changed = 0
    for base_row, other_row in zip(base_rows, other_rows):
        if base_row["answer_text"] != other_row["answer_text"]:
            changed += 1
    return float(changed / len(base_rows))


def patch_layers(layers, layer_indices, basis_matrix: np.ndarray, alpha: float, modules_mode: str):
    patch_log = []
    for layer_idx in layer_indices:
        basis_np = normalize_basis(np.asarray(basis_matrix[layer_idx], dtype=np.float32))
        if basis_np.shape[0] == 0:
            patch_log.append({"layer": layer_idx, "patched_modules": [], "skipped": True})
            continue
        basis = torch.tensor(basis_np, dtype=torch.float32, device=next(layers[layer_idx].parameters()).device)
        patched_modules = []
        for module_name, module in get_layer_write_modules(layers[layer_idx], modules_mode):
            apply_subspace_patch(module.weight.data, basis, alpha)
            patched_modules.append(module_name)
        patch_log.append(
            {
                "layer": layer_idx,
                "basis_rank": int(basis.shape[0]),
                "patched_modules": patched_modules,
                "skipped": len(patched_modules) == 0,
            }
        )
    return patch_log


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.data_jsonl))
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("No HERETIC-simple eval rows loaded.")

    layer_indices = parse_layers(args.layers)
    if not layer_indices:
        raise ValueError("No layers selected.")

    direction_npz = np.load(args.directions_npz)
    if args.direction_key not in direction_npz:
        raise KeyError(f"Direction key '{args.direction_key}' not found in {args.directions_npz}")
    target_basis = np.asarray(direction_npz[args.direction_key], dtype=np.float32)
    control_basis = None
    if args.control_direction_key:
        if args.control_direction_key not in direction_npz:
            raise KeyError(
                f"Control direction key '{args.control_direction_key}' not found in {args.directions_npz}"
            )
        control_basis = np.asarray(direction_npz[args.control_direction_key], dtype=np.float32)

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        args.dtype,
        load_in_4bit=False,
        max_gpu_memory_gb=args.gpu_memory_gb,
    )
    layers = get_decoder_layers(model)

    originals, row_norms_before = snapshot_weights_and_norms(layers, layer_indices, args.modules)
    base_rows, base_summary = evaluate_condition("base", model, tokenizer, rows, args.max_new_tokens)

    target_patch_log = patch_layers(layers, layer_indices, target_basis, args.alpha, args.modules)
    _, row_norms_target = snapshot_weights_and_norms(layers, layer_indices, args.modules)
    target_rows, target_summary = evaluate_condition("target", model, tokenizer, rows, args.max_new_tokens)
    restore_weights(layers, args.modules, originals)

    control_rows = []
    control_summary = None
    control_patch_log = []
    control_row_norms = []
    if control_basis is not None:
        control_patch_log = patch_layers(layers, layer_indices, control_basis, args.alpha, args.modules)
        _, row_norms_control = snapshot_weights_and_norms(layers, layer_indices, args.modules)
        control_rows, control_summary = evaluate_condition("control", model, tokenizer, rows, args.max_new_tokens)
        control_row_norms = row_norm_delta(row_norms_before, row_norms_control)
        restore_weights(layers, args.modules, originals)

    result = {
        "data_jsonl": args.data_jsonl,
        "model": args.model,
        "dtype": args.dtype,
        "gpu_memory_gb": args.gpu_memory_gb,
        "directions_npz": args.directions_npz,
        "direction_key": args.direction_key,
        "control_direction_key": args.control_direction_key or None,
        "layers": layer_indices,
        "alpha": args.alpha,
        "modules": args.modules,
        "base": base_summary,
        "target": target_summary,
        "control": control_summary,
        "delta": {
            "target_contradicted_rate": target_summary["contradicted_rate"] - base_summary["contradicted_rate"],
            "target_supported_answer_rate": target_summary["supported_answer_rate"] - base_summary["supported_answer_rate"],
            "target_good_non_direct_rate": target_summary["good_non_direct_rate"] - base_summary["good_non_direct_rate"],
            "target_bad_abstention_rate": target_summary["bad_abstention_rate"] - base_summary["bad_abstention_rate"],
            "target_mean_output_length": target_summary["mean_output_length"] - base_summary["mean_output_length"],
            "target_mean_first_step_entropy": target_summary["mean_first_step_entropy"] - base_summary["mean_first_step_entropy"],
            "target_mean_generation_entropy": target_summary["mean_generation_entropy"] - base_summary["mean_generation_entropy"],
            "target_mean_prompt_final_hidden_norm": target_summary["mean_prompt_final_hidden_norm"] - base_summary["mean_prompt_final_hidden_norm"],
            "target_output_changed_rate": changed_rate(base_rows, target_rows),
        },
        "patch_log": {
            "target": target_patch_log,
            "control": control_patch_log,
        },
        "row_norm_change": {
            "target": row_norm_delta(row_norms_before, row_norms_target),
            "control": control_row_norms,
        },
        "rows": {
            "base": base_rows,
            "target": target_rows,
            "control": control_rows,
        },
    }
    if control_summary is not None:
        result["delta"].update(
            {
                "control_contradicted_rate": control_summary["contradicted_rate"] - base_summary["contradicted_rate"],
                "control_supported_answer_rate": control_summary["supported_answer_rate"] - base_summary["supported_answer_rate"],
                "control_good_non_direct_rate": control_summary["good_non_direct_rate"] - base_summary["good_non_direct_rate"],
                "control_bad_abstention_rate": control_summary["bad_abstention_rate"] - base_summary["bad_abstention_rate"],
                "control_mean_prompt_final_hidden_norm": control_summary["mean_prompt_final_hidden_norm"] - base_summary["mean_prompt_final_hidden_norm"],
                "control_output_changed_rate": changed_rate(base_rows, control_rows),
            }
        )

    save_json(Path(args.output_json), result)
    print(json.dumps(result["delta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
