import argparse
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from libs.critics import set_critic
from libs.bounds import estimate_mutual_information


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch cosine similarity analysis: for each dataset subdir, load npz + saved separable critic, "
            "compute positive-pair cosines, and plot vs PMI."
        )
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Root directory containing subdirectories with npz files (one per dataset).",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Root results directory containing subdir/<estimator>/critic.pth checkpoints.",
    )
    parser.add_argument(
        "--estimator",
        default="infonce",
        help="Estimator subfolder name under results_dir to locate critic.pth (e.g., infonce, js, etc.).",
    )
    parser.add_argument("--data_key", default="X", help="Key for paired data inside each npz (e.g., X or Noise).")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for embedding inference.")
    parser.add_argument("--critic_depth", type=int, default=2, help="Hidden depth used when training the critic.")
    parser.add_argument("--critic_width", type=int, default=256, help="Hidden width used when training the critic.")
    parser.add_argument("--critic_embed", type=int, default=32, help="Embedding dim used when training the critic.")
    parser.add_argument(
        "--dr",
        type=int,
        default=None,
        help="Flattened input dim for x/y. If unset, inferred per dataset from npz shape.",
    )
    parser.add_argument("--activation", default="relu", help="Activation used during training (defaults to relu).")
    parser.add_argument(
        "--normalization",
        default=None,
        choices=[None, "None", "layernorm", "batchnorm"],
        help="Normalization used during training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="cuda, cpu, or auto (auto picks cuda if available).",
    )
    parser.add_argument(
        "--output_dir",
        default="pmi_cosine_plots",
        help="Directory to write plots (and optional numpy dumps). One plot per dataset subdir.",
    )
    parser.add_argument(
        "--save_numpy",
        action="store_true",
        help="If set, also save cosine arrays as <dataset>_cosines.npy alongside plots.",
    )
    parser.add_argument(
        "--use_final",
        action="store_true",
        help="If set, use the mean of the last 1000 MI steps for the true-vs-estimated plot; otherwise full mean.",
    )
    parser.add_argument(
        "--recompute_mi",
        action="store_true",
        help="If set, recompute MI on the saved critic over the npz data (batched InfoNCE) instead of reading mi.npy.",
    )
    return parser.parse_args()


def find_npz(subdir: str) -> str:
    candidates = sorted(glob.glob(os.path.join(subdir, "*.npz")))
    if not candidates:
        raise FileNotFoundError(f"No npz found in {subdir}")
    return candidates[0]


def load_npz(npz_path: str, data_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if data_key not in data:
        raise KeyError(f"data_key '{data_key}' not found in {npz_path}. Keys: {list(data.keys())}")
    if "PMI" not in data:
        raise KeyError(f"'PMI' not found in {npz_path}. Keys: {list(data.keys())}")
    mi_truth = None
    if "MI_results" in data:
        mi_arr = data["MI_results"]
        if len(mi_arr) > 2:
            mi_truth = float(mi_arr[2])
        else:
            mi_truth = float(mi_arr[0])
    return data[data_key], data["PMI"], mi_truth


def build_critic(args, input_dim: int, device: str):
    critic = set_critic(
        critic_type="separable",
        dim=input_dim,
        hidden_dim=args.critic_width,
        embed_dim=args.critic_embed,
        layers=args.critic_depth,
        activation=args.activation,
        normalization=args.normalization,
        device=device,
    )
    return critic


def compute_cosines(critic, data_array: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    cosines: List[np.ndarray] = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, data_array.shape[0], batch_size):
            end = min(start + batch_size, data_array.shape[0])
            z1 = torch.from_numpy(data_array[start:end, 0]).float().to(device)
            z2 = torch.from_numpy(data_array[start:end, 1]).float().to(device)
            x = z1.view(z1.size(0), -1)
            y = z2.view(z2.size(0), -1)
            gx = critic._g(x)
            hy = critic._h(y)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cos_batch = cos(gx, hy)
            cosines.append(cos_batch.cpu().numpy())
    return np.concatenate(cosines)


def plot_scatter(pmi: np.ndarray, cosines: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(7, 5))
    plt.scatter(pmi, cosines, s=8, alpha=0.5)
    plt.xlabel("PMI")
    plt.ylabel("Cosine similarity of g(x) and h(y)")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Iterate over dataset subdirectories
    subdirs = sorted([d for d in os.listdir(args.dataset_dir) if os.path.isdir(os.path.join(args.dataset_dir, d))])
    if not subdirs:
        raise RuntimeError(f"No dataset subdirectories found in {args.dataset_dir}")

    true_mis = []
    est_mis = []
    first_overlay = None  # (subdir, pmi, cosines, mi_truth)
    last_overlay = None   # (subdir, pmi, cosines, mi_truth)

    for subdir in subdirs:
        dataset_path = os.path.join(args.dataset_dir, subdir)
        npz_path = find_npz(dataset_path)
        data_array, pmi, mi_truth = load_npz(npz_path, args.data_key)
        if data_array.shape[0] != pmi.shape[0]:
            raise ValueError(f"[{subdir}] PMI length {pmi.shape[0]} does not match data size {data_array.shape[0]}.")

        input_dim = args.dr or int(np.prod(data_array.shape[2:]))
        critic = build_critic(args, input_dim, device)

        ckpt_path = os.path.join(args.results_dir, subdir, args.estimator, "critic.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[{subdir}] Critic checkpoint not found at {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device)
        critic.load_state_dict(state)

        cosines = compute_cosines(critic, data_array, args.batch_size, device)

        if mi_truth is None:
            title = "PMI vs InfoNCE Cosine Similarity (I(X1;X2) = N/A)"
        else:
            title = f"PMI vs InfoNCE Cosine Similarity (I(X1;X2) = {mi_truth:.4f})"

        plot_path = os.path.join(args.output_dir, f"{subdir}_pmi_cosine.png")
        plot_scatter(pmi, cosines, plot_path, title=title)

        if first_overlay is None:
            first_overlay = (subdir, pmi, cosines, mi_truth)
        last_overlay = (subdir, pmi, cosines, mi_truth)

        if args.save_numpy:
            np.save(os.path.join(args.output_dir, f"{subdir}_cosines.npy"), cosines)

        mean_cos = float(cosines.mean())
        print(f"[{subdir}] saved plot -> {plot_path}; mean cosine: {mean_cos:.4f}")

        # Collect true vs estimated MI if available
        if mi_truth is not None:
            if args.recompute_mi:
                # Evaluate MI on the saved critic over the dataset in batches (no gradients)
                mi_batches = []
                critic.eval()
                with torch.no_grad():
                    for start in range(0, data_array.shape[0], args.batch_size):
                        end = min(start + args.batch_size, data_array.shape[0])
                        z1 = torch.from_numpy(data_array[start:end, 0]).float().to(device)
                        z2 = torch.from_numpy(data_array[start:end, 1]).float().to(device)
                        x = z1.view(z1.size(0), -1)
                        y = z2.view(z2.size(0), -1)
                        mi_val = estimate_mutual_information(args.estimator, x, y, critic)
                        mi_batches.append(float(mi_val.item()))
                estimate = float(np.mean(mi_batches))
            else:
                mi_file = os.path.join(args.results_dir, subdir, args.estimator, "mi.npy")
                if os.path.exists(mi_file):
                    mi_estimates = np.load(mi_file)
                    estimate = (
                        np.mean(mi_estimates[-1000:])
                        if args.use_final and mi_estimates.shape[0] > 1000
                        else np.mean(mi_estimates)
                    )
                else:
                    estimate = None

            if estimate is not None:
                true_mis.append(mi_truth)
                est_mis.append(estimate)

    if first_overlay is not None and last_overlay is not None:
        first_name, first_pmi, first_cos, first_mi = first_overlay
        last_name, last_pmi, last_cos, last_mi = last_overlay

        def _mi_str(val):
            return "N/A" if val is None else f"{val:.4f}"

        plt.figure(figsize=(7, 5))
        plt.scatter(first_pmi, first_cos, s=4, alpha=0.3, label=f"I={_mi_str(first_mi)}")
        plt.scatter(last_pmi, last_cos, s=4, alpha=0.3, label=f"I={_mi_str(last_mi)}")
        plt.xlabel("PMI")
        plt.ylabel("Cosine similarity of g(x) and h(y)")
        plt.title("PMI vs Cosine Similarity Overlay")
        plt.grid(alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()
        overlay_path = os.path.join(args.output_dir, "first_vs_last_pmi_cosine_overlay.png")
        plt.savefig(overlay_path, dpi=200)
        plt.close()
        print(f"[overlay] saved first-vs-last overlay plot -> {overlay_path}")

    # Plot aggregated true vs estimated MI
    if true_mis and est_mis:
        true_mis_arr = np.array(true_mis)
        est_mis_arr = np.array(est_mis)
        idx = np.argsort(true_mis_arr)
        true_mis_arr = true_mis_arr[idx]
        est_mis_arr = est_mis_arr[idx]

        min_val = min(true_mis_arr.min(), est_mis_arr.min())
        max_val = max(true_mis_arr.max(), est_mis_arr.max())
        margin = (max_val - min_val) * 0.05
        min_val -= margin
        max_val += margin

        plt.figure(figsize=(8, 6))
        plt.scatter(true_mis_arr, est_mis_arr, s=60, alpha=0.7, label=args.estimator)
        plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.6, label="Perfect estimation")
        corr = np.corrcoef(true_mis_arr, est_mis_arr)[0, 1]
        rmse = np.sqrt(np.mean((true_mis_arr - est_mis_arr) ** 2))
        plt.xlabel("Ground Truth MI")
        plt.ylabel("Estimated MI")
        plt.title(f"True vs Estimated MI ({args.estimator})\nCorr: {corr:.3f}, RMSE: {rmse:.4f}")
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        tv_plot = os.path.join(args.output_dir, f"true_vs_estimated_{args.estimator}.png")
        plt.savefig(tv_plot, dpi=300)
        plt.close()
        print(f"[aggregate] saved true-vs-estimated plot -> {tv_plot}")


if __name__ == "__main__":
    main()
