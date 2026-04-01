import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import load_jsonl, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Extract HERETIC-simple early-answer directions with diagnostics")
    parser.add_argument(
        "--capture-jsonl",
        required=True,
        help="Generation capture JSONL",
    )
    parser.add_argument(
        "--capture-npz",
        required=True,
        help="Capture NPZ with early-answer residuals",
    )
    parser.add_argument(
        "--state-key",
        default="answer_token_1",
        choices=["answer_token_1", "answer_token_1_to_5_mean"],
        help="Which captured state to use for extraction",
    )
    parser.add_argument(
        "--winsorization-quantile",
        type=float,
        default=0.0,
        help="Optional per-feature winsorization quantile",
    )
    parser.add_argument(
        "--trim-fraction",
        type=float,
        default=0.0,
        help="Optional symmetric trimmed-mean fraction",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=100,
        help="Bootstrap stability rounds",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--output-npz",
        required=True,
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        required=True,
        help="Output metadata JSON path",
    )
    return parser.parse_args()


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.where(norms <= 1e-12, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def orthogonalize_per_layer(direction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    result = np.zeros_like(direction)
    for layer_idx in range(direction.shape[0]):
        vec = direction[layer_idx]
        ref = reference[layer_idx]
        denom = float(np.dot(ref, ref))
        if denom <= 1e-12:
            result[layer_idx] = vec
        else:
            result[layer_idx] = vec - (float(np.dot(vec, ref)) / denom) * ref
    return result.astype(np.float32)


def winsorize_matrix(data: np.ndarray, quantile: float) -> np.ndarray:
    if quantile <= 0.0:
        return data
    lo = np.quantile(data, quantile, axis=0, keepdims=True)
    hi = np.quantile(data, 1.0 - quantile, axis=0, keepdims=True)
    return np.clip(data, lo, hi)


def trimmed_mean(data: np.ndarray, trim_fraction: float) -> np.ndarray:
    if trim_fraction <= 0.0:
        return data.mean(axis=0).astype(np.float32)
    n = data.shape[0]
    trim = int(n * trim_fraction)
    if trim <= 0 or trim * 2 >= n:
        return data.mean(axis=0).astype(np.float32)
    sorted_data = np.sort(data, axis=0)
    return sorted_data[trim : n - trim].mean(axis=0).astype(np.float32)


def mean_direction(pos: np.ndarray, neg: np.ndarray, winsor_q: float, trim_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_proc = winsorize_matrix(pos, winsor_q)
    neg_proc = winsorize_matrix(neg, winsor_q)
    pos_mean = trimmed_mean(pos_proc, trim_fraction)
    neg_mean = trimmed_mean(neg_proc, trim_fraction)
    return (pos_mean - neg_mean).astype(np.float32), pos_mean.astype(np.float32), neg_mean.astype(np.float32)


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    equal = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * equal)


def diagonal_mahalanobis(mean_pos: np.ndarray, mean_neg: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    pos_var = pos.var(axis=0, ddof=1) if len(pos) > 1 else np.zeros_like(mean_pos)
    neg_var = neg.var(axis=0, ddof=1) if len(neg) > 1 else np.zeros_like(mean_neg)
    pooled = 0.5 * (pos_var + neg_var) + 1e-6
    diff = mean_pos - mean_neg
    return float(np.sqrt(((diff * diff) / pooled).sum()))


def top_component(data: np.ndarray) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0].astype(np.float32)


def bootstrap_stability(pos: np.ndarray, neg: np.ndarray, full_direction: np.ndarray, rounds: int, rng: np.random.Generator) -> float:
    if rounds <= 0 or len(pos) < 2 or len(neg) < 2:
        return float("nan")
    values = []
    for _ in range(rounds):
        pos_idx = rng.integers(0, len(pos), size=max(2, int(0.8 * len(pos))))
        neg_idx = rng.integers(0, len(neg), size=max(2, int(0.8 * len(neg))))
        boot_direction = pos[pos_idx].mean(axis=0) - neg[neg_idx].mean(axis=0)
        values.append(cosine_similarity(boot_direction, full_direction))
    return float(np.mean(values))


def top_layers_from_values(values: list[float], top_k: int = 8) -> list[dict]:
    rows = [{"layer": idx, "value": float(value)} for idx, value in enumerate(values)]
    return sorted(rows, key=lambda row: row["value"], reverse=True)[:top_k]


def collect_pairwise_group_indices(rows: list[dict], target_buckets: set[str]) -> list[tuple[int, list[int]]]:
    grouped = defaultdict(lambda: {"direct": [], "target": []})
    for idx, row in enumerate(rows):
        pair_group = row.get("pair_group") or row.get("metadata", {}).get("pair_group") or row["prompt_id"]
        bucket = row["bucket"]
        if row["binary_bucket"] == "direct_answer_ok":
            grouped[pair_group]["direct"].append(idx)
        if bucket in target_buckets:
            grouped[pair_group]["target"].append(idx)

    pairs = []
    for value in grouped.values():
        if len(value["direct"]) == 1 and len(value["target"]) >= 1:
            pairs.append((value["direct"][0], list(value["target"])))
    return pairs


def pairwise_mean_difference(states: np.ndarray, pairs: list[tuple[int, list[int]]]) -> np.ndarray:
    if not pairs:
        return np.zeros(states.shape[1:], dtype=np.float32)
    diffs = []
    for direct_idx, target_indices in pairs:
        direct_state = states[direct_idx]
        target_mean = states[target_indices].mean(axis=0)
        diffs.append((target_mean - direct_state).astype(np.float32))
    return np.mean(np.stack(diffs, axis=0), axis=0).astype(np.float32)


def main():
    args = parse_args()
    rows = load_jsonl(Path(args.capture_jsonl))
    data = np.load(args.capture_npz)
    states = np.asarray(data[args.state_key], dtype=np.float32)

    if len(rows) != states.shape[0]:
        raise ValueError("Capture JSONL row count does not match NPZ sample count.")

    labels = np.asarray(
        [1 if row["binary_bucket"] == "do_not_confidently_continue" else 0 for row in rows],
        dtype=np.int64,
    )
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both direct and non-direct samples for extraction.")

    n_samples, n_layers, hidden_size = states.shape
    raw_direction = np.zeros((n_layers, hidden_size), dtype=np.float32)
    pos_mean = np.zeros_like(raw_direction)
    neg_mean = np.zeros_like(raw_direction)
    shuffled_direction = np.zeros_like(raw_direction)
    pca_direction = np.zeros_like(raw_direction)

    cosine_distances = []
    within_class_variance = []
    probe_auc = []
    mahalanobis = []
    bootstrap_mean_cosine = []
    direction_norms = []

    rng = np.random.default_rng(args.seed)
    shuffled_labels = rng.permutation(labels)
    shuf_pos_idx = np.where(shuffled_labels == 1)[0]
    shuf_neg_idx = np.where(shuffled_labels == 0)[0]

    for layer_idx in range(n_layers):
        layer_states = states[:, layer_idx, :]
        pos = layer_states[pos_idx]
        neg = layer_states[neg_idx]
        direction, pos_layer_mean, neg_layer_mean = mean_direction(
            pos,
            neg,
            args.winsorization_quantile,
            args.trim_fraction,
        )
        raw_direction[layer_idx] = direction
        pos_mean[layer_idx] = pos_layer_mean
        neg_mean[layer_idx] = neg_layer_mean
        direction_norms.append(float(np.linalg.norm(direction)))
        cosine_distances.append(cosine_distance(pos_layer_mean, neg_layer_mean))

        pos_var = ((pos - pos_layer_mean) ** 2).sum(axis=1).mean() if len(pos) > 0 else 0.0
        neg_var = ((neg - neg_layer_mean) ** 2).sum(axis=1).mean() if len(neg) > 0 else 0.0
        within_class_variance.append(float(0.5 * (pos_var + neg_var)))

        scores = layer_states @ l2_normalize_rows(direction[None, :])[0]
        probe_auc.append(auc_from_scores(scores, labels))
        mahalanobis.append(diagonal_mahalanobis(pos_layer_mean, neg_layer_mean, pos, neg))
        bootstrap_mean_cosine.append(
            bootstrap_stability(pos, neg, direction, args.bootstrap_rounds, np.random.default_rng(args.seed + layer_idx))
        )

        shuffled_direction[layer_idx] = (
            layer_states[shuf_pos_idx].mean(axis=0) - layer_states[shuf_neg_idx].mean(axis=0)
        ).astype(np.float32)
        pca_direction[layer_idx] = top_component(layer_states)

    normalized_direction = l2_normalize_rows(raw_direction)
    projected_direction = orthogonalize_per_layer(raw_direction, neg_mean)
    projected_normalized_direction = l2_normalize_rows(projected_direction)
    shuffled_normalized = l2_normalize_rows(shuffled_direction)
    pca_normalized = l2_normalize_rows(pca_direction)

    pairwise_all_direction = pairwise_mean_difference(
        states,
        collect_pairwise_group_indices(
            rows,
            {
                "insufficient_should_abstain",
                "fabricated_premise_should_reject",
                "ambiguous_should_clarify",
            },
        ),
    )
    pairwise_abstain_direction = pairwise_mean_difference(
        states,
        collect_pairwise_group_indices(rows, {"insufficient_should_abstain"}),
    )
    pairwise_reject_direction = pairwise_mean_difference(
        states,
        collect_pairwise_group_indices(rows, {"fabricated_premise_should_reject"}),
    )
    pairwise_clarify_direction = pairwise_mean_difference(
        states,
        collect_pairwise_group_indices(rows, {"ambiguous_should_clarify"}),
    )

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        non_direct_minus_direct__raw=raw_direction.astype(np.float32),
        non_direct_minus_direct__normalized=normalized_direction.astype(np.float32),
        non_direct_minus_direct__projected_orth_to_direct_mean=projected_direction.astype(np.float32),
        non_direct_minus_direct__projected_orth_to_direct_mean__normalized=projected_normalized_direction.astype(np.float32),
        direct_answer_ok__mean=neg_mean.astype(np.float32),
        do_not_confidently_continue__mean=pos_mean.astype(np.float32),
        shuffled_non_direct_minus_direct__raw=shuffled_direction.astype(np.float32),
        shuffled_non_direct_minus_direct__normalized=shuffled_normalized.astype(np.float32),
        pca_1__raw=pca_direction.astype(np.float32),
        pca_1__normalized=pca_normalized.astype(np.float32),
        pairmean_non_direct_minus_direct__raw=pairwise_all_direction.astype(np.float32),
        pairmean_non_direct_minus_direct__normalized=l2_normalize_rows(pairwise_all_direction).astype(np.float32),
        pairmean_abstain_minus_direct__raw=pairwise_abstain_direction.astype(np.float32),
        pairmean_abstain_minus_direct__normalized=l2_normalize_rows(pairwise_abstain_direction).astype(np.float32),
        pairmean_reject_minus_direct__raw=pairwise_reject_direction.astype(np.float32),
        pairmean_reject_minus_direct__normalized=l2_normalize_rows(pairwise_reject_direction).astype(np.float32),
        pairmean_clarify_minus_direct__raw=pairwise_clarify_direction.astype(np.float32),
        pairmean_clarify_minus_direct__normalized=l2_normalize_rows(pairwise_clarify_direction).astype(np.float32),
    )

    metadata = {
        "capture_jsonl": args.capture_jsonl,
        "capture_npz": args.capture_npz,
        "state_key": args.state_key,
        "n_samples": n_samples,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "label_counts": {
            "direct_answer_ok": int((labels == 0).sum()),
            "do_not_confidently_continue": int((labels == 1).sum()),
        },
        "winsorization_quantile": args.winsorization_quantile,
        "trim_fraction": args.trim_fraction,
        "bootstrap_rounds": args.bootstrap_rounds,
        "diagnostics": {
            "direction_norm": direction_norms,
            "cosine_distance_between_class_means": cosine_distances,
            "within_class_variance": within_class_variance,
            "linear_probe_auc": probe_auc,
            "mahalanobis_separation": mahalanobis,
            "bootstrap_mean_cosine": bootstrap_mean_cosine,
        },
        "top_layers": {
            "by_direction_norm": top_layers_from_values(direction_norms),
            "by_probe_auc": top_layers_from_values(probe_auc),
            "by_mahalanobis": top_layers_from_values(mahalanobis),
            "by_bootstrap_stability": top_layers_from_values(bootstrap_mean_cosine),
            "by_pairmean_non_direct_norm": top_layers_from_values(
                [float(np.linalg.norm(pairwise_all_direction[layer_idx])) for layer_idx in range(n_layers)]
            ),
            "by_pairmean_abstain_norm": top_layers_from_values(
                [float(np.linalg.norm(pairwise_abstain_direction[layer_idx])) for layer_idx in range(n_layers)]
            ),
            "by_pairmean_reject_norm": top_layers_from_values(
                [float(np.linalg.norm(pairwise_reject_direction[layer_idx])) for layer_idx in range(n_layers)]
            ),
            "by_pairmean_clarify_norm": top_layers_from_values(
                [float(np.linalg.norm(pairwise_clarify_direction[layer_idx])) for layer_idx in range(n_layers)]
            ),
        },
        "direction_semantics": "non_direct_minus_direct; runtime subtraction suppresses overconfident continuation side",
        "pairwise_groups": {
            "non_direct_vs_direct": len(
                collect_pairwise_group_indices(
                    rows,
                    {
                        "insufficient_should_abstain",
                        "fabricated_premise_should_reject",
                        "ambiguous_should_clarify",
                    },
                )
            ),
            "abstain_vs_direct": len(collect_pairwise_group_indices(rows, {"insufficient_should_abstain"})),
            "reject_vs_direct": len(
                collect_pairwise_group_indices(rows, {"fabricated_premise_should_reject"})
            ),
            "clarify_vs_direct": len(collect_pairwise_group_indices(rows, {"ambiguous_should_clarify"})),
        },
        "saved_keys": [
            "non_direct_minus_direct__raw",
            "non_direct_minus_direct__normalized",
            "non_direct_minus_direct__projected_orth_to_direct_mean",
            "non_direct_minus_direct__projected_orth_to_direct_mean__normalized",
            "direct_answer_ok__mean",
            "do_not_confidently_continue__mean",
            "shuffled_non_direct_minus_direct__raw",
            "shuffled_non_direct_minus_direct__normalized",
            "pca_1__raw",
            "pca_1__normalized",
            "pairmean_non_direct_minus_direct__raw",
            "pairmean_non_direct_minus_direct__normalized",
            "pairmean_abstain_minus_direct__raw",
            "pairmean_abstain_minus_direct__normalized",
            "pairmean_reject_minus_direct__raw",
            "pairmean_reject_minus_direct__normalized",
            "pairmean_clarify_minus_direct__raw",
            "pairmean_clarify_minus_direct__normalized",
        ],
    }
    save_json(Path(args.metadata_json), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
