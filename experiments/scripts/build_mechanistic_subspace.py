import argparse
import json
from pathlib import Path

import numpy as np

from common import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build a low-rank mechanistic subspace from reviewed snippet states")
    parser.add_argument(
        "--states-npz",
        required=True,
        help="Snippet-state NPZ with per-label sample arrays",
    )
    parser.add_argument(
        "--supported-key",
        default="supported_sentence__samples",
        help="Sample key for supported rows",
    )
    parser.add_argument(
        "--unsupported-key",
        default="unsupported_onset_snippet__samples",
        help="Sample key for unsupported rows",
    )
    parser.add_argument("--rank", type=int, default=4, help="Target subspace rank per layer")
    parser.add_argument(
        "--output-npz",
        default="experiments/artifacts/mechanistic_reviewed_subspace.npz",
        help="Output NPZ path",
    )
    parser.add_argument(
        "--metadata-json",
        default="experiments/artifacts/mechanistic_reviewed_subspace_meta.json",
        help="Output metadata path",
    )
    return parser.parse_args()


def orthonormal_basis(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    q, _ = np.linalg.qr(mat.T)
    return q.T


def main():
    args = parse_args()
    data = np.load(args.states_npz)
    supported = np.asarray(data[args.supported_key], dtype=np.float32)
    unsupported = np.asarray(data[args.unsupported_key], dtype=np.float32)
    if supported.ndim != 3 or unsupported.ndim != 3:
        raise ValueError("Expected sample arrays with shape [n, layers, dim].")
    if supported.shape[1:] != unsupported.shape[1:]:
        raise ValueError("Supported and unsupported sample arrays must share [layers, dim].")

    n_layers = supported.shape[1]
    dim = supported.shape[2]
    rank = max(1, int(args.rank))
    basis = np.zeros((n_layers, rank, dim), dtype=np.float32)
    mean_diff = np.zeros((n_layers, dim), dtype=np.float32)
    explained = []

    for layer_idx in range(n_layers):
        s = supported[:, layer_idx, :]
        u = unsupported[:, layer_idx, :]
        s_mean = s.mean(axis=0)
        u_mean = u.mean(axis=0)
        diff = u_mean - s_mean
        mean_diff[layer_idx] = diff

        residuals = np.concatenate([u - u_mean, s - s_mean], axis=0)
        u_svd, sing_vals, vt = np.linalg.svd(residuals, full_matrices=False)
        _ = u_svd

        vectors = []
        diff_norm = float(np.linalg.norm(diff))
        if diff_norm > 1e-12:
            vectors.append(diff / diff_norm)
        for vec in vt:
            if len(vectors) >= rank:
                break
            vectors.append(vec.astype(np.float32))
        mat = np.stack(vectors[:rank], axis=0)
        mat = orthonormal_basis(mat)
        if mat.shape[0] < rank:
            padded = np.zeros((rank, dim), dtype=np.float32)
            padded[: mat.shape[0], :] = mat
            mat = padded
        basis[layer_idx] = mat[:rank]
        total_var = float((sing_vals**2).sum()) if sing_vals.size else 0.0
        top_var = float((sing_vals[: max(0, rank - 1)] ** 2).sum()) if sing_vals.size else 0.0
        explained.append(
            {
                "layer": layer_idx,
                "mean_diff_norm": diff_norm,
                "residual_topk_explained_ratio": 0.0 if total_var <= 0 else top_var / total_var,
            }
        )

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, basis=basis, mean_diff=mean_diff)

    meta = {
        "states_npz": args.states_npz,
        "supported_key": args.supported_key,
        "unsupported_key": args.unsupported_key,
        "rank": rank,
        "n_layers": int(n_layers),
        "dim": int(dim),
        "layer_stats_top8_mean_diff_norm": sorted(explained, key=lambda x: x["mean_diff_norm"], reverse=True)[:8],
    }
    save_json(Path(args.metadata_json), meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Saved subspace to: {args.output_npz}")


if __name__ == "__main__":
    main()
