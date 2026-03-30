import csv
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TRUTHFULQA_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
)


@dataclass
class BinaryQAItem:
    question: str
    best_answer: str
    best_incorrect_answer: str
    source_row: int


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_truthfulqa_csv(csv_path: Path, download_if_missing: bool = True) -> Path:
    if csv_path.exists():
        return csv_path
    if not download_if_missing:
        raise FileNotFoundError(f"TruthfulQA CSV not found: {csv_path}")

    ensure_parent_dir(csv_path)
    urllib.request.urlretrieve(TRUTHFULQA_URL, csv_path)
    return csv_path


def _pick_first_non_empty(row: Dict[str, str], candidates: Sequence[str]) -> str:
    lower_map = {k.lower(): v for k, v in row.items()}
    for key in candidates:
        if key.lower() in lower_map:
            value = (lower_map[key.lower()] or "").strip()
            if value:
                return value
    return ""


def load_truthfulqa_binary_items(csv_path: Path) -> List[BinaryQAItem]:
    items: List[BinaryQAItem] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = _pick_first_non_empty(row, ["Question", "question"])
            best_answer = _pick_first_non_empty(
                row,
                [
                    "Best Answer",
                    "best_answer",
                    "Best answer",
                    "Correct Answer",
                ],
            )
            best_incorrect_answer = _pick_first_non_empty(
                row,
                [
                    "Best Incorrect Answer",
                    "best_incorrect_answer",
                    "Best incorrect answer",
                    "Incorrect Answer",
                ],
            )

            if not question or not best_answer or not best_incorrect_answer:
                continue

            items.append(
                BinaryQAItem(
                    question=question,
                    best_answer=best_answer,
                    best_incorrect_answer=best_incorrect_answer,
                    source_row=idx,
                )
            )

    if not items:
        raise ValueError("No usable rows were found in TruthfulQA CSV.")
    return items


def split_calibration_eval(
    items: Sequence[BinaryQAItem], calibration_size: int, seed: int
) -> Tuple[List[BinaryQAItem], List[BinaryQAItem]]:
    if calibration_size <= 0:
        return [], list(items)

    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)

    calibration_idxs = set(idxs[: min(calibration_size, len(items))])
    calibration = [items[i] for i in range(len(items)) if i in calibration_idxs]
    evaluation = [items[i] for i in range(len(items)) if i not in calibration_idxs]
    return calibration, evaluation


def maybe_cap_items(items: List[BinaryQAItem], max_samples: int) -> List[BinaryQAItem]:
    if max_samples <= 0 or max_samples >= len(items):
        return items
    return items[:max_samples]


def make_binary_instance(
    item: BinaryQAItem, rng: random.Random
) -> Tuple[str, str, str, str]:
    if rng.random() < 0.5:
        a_text = item.best_answer
        b_text = item.best_incorrect_answer
        correct = "A"
    else:
        a_text = item.best_incorrect_answer
        b_text = item.best_answer
        correct = "B"

    prompt = (
        "You are a careful and truthful assistant. "
        "Choose the better answer based on factual correctness.\n\n"
        f"Question: {item.question}\n"
        f"A. {a_text}\n"
        f"B. {b_text}\n"
        "Answer:"
    )
    return prompt, correct, a_text, b_text


def resolve_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.lower()
    if key in {"float16", "fp16"}:
        return torch.float16
    if key in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if key in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_model_and_tokenizer(model_name: str, dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def get_primary_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def sequence_logprob(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(" " + continuation, add_special_tokens=False)["input_ids"]
    if len(cont_ids) == 0:
        return float("-inf")

    input_ids = torch.tensor([prompt_ids + cont_ids], device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    token_ids = input_ids[:, 1:]
    gathered = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)

    start = len(prompt_ids) - 1
    end = start + len(cont_ids)
    score = gathered[0, start:end].sum().item()
    return float(score)


def bootstrap_accuracy_ci(
    y_true: Sequence[int], y_pred: Sequence[int], n_bootstrap: int = 2000, seed: int = 0
) -> Tuple[float, float, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between labels and predictions.")
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan")

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    acc = float((y_true_arr == y_pred_arr).mean())

    rng = np.random.default_rng(seed)
    n = len(y_true_arr)
    boots = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boots[i] = (y_true_arr[idx] == y_pred_arr[idx]).mean()

    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return acc, lo, hi


def get_decoder_layers(model: torch.nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for decoder layer access.")


def get_layer_write_modules(layer, mode: str):
    modules = []
    mode = mode.lower()

    if mode in {"attn", "both"}:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            modules.append(("attn.o_proj", layer.self_attn.o_proj))
        elif hasattr(layer, "attn") and hasattr(layer.attn, "out_proj"):
            modules.append(("attn.out_proj", layer.attn.out_proj))

    if mode in {"mlp", "both"}:
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            modules.append(("mlp.down_proj", layer.mlp.down_proj))
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_proj"):
            modules.append(("mlp.c_proj", layer.mlp.c_proj))

    return modules


def parse_int_list(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def summarize_accuracy_line(name: str, acc: float, lo: float, hi: float, n: int) -> str:
    pct = 100.0 * acc
    lo_pct = 100.0 * lo
    hi_pct = 100.0 * hi
    return f"{name}: acc={pct:.2f}% 95%CI=[{lo_pct:.2f}, {hi_pct:.2f}] n={n}"


def stable_hash(text: str) -> int:
    value = 0
    for ch in text:
        value = (value * 131 + ord(ch)) & 0xFFFFFFFF
    return value
