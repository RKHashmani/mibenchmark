import argparse
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from libs.critics import set_critic


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
    return data[data_key], data["PMI"]


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
            gx = F.normalize(gx, dim=1)
            hy = F.normalize(hy, dim=1)
            cos_batch = (gx * hy).sum(dim=1)
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

    for subdir in subdirs:
        dataset_path = os.path.join(args.dataset_dir, subdir)
        npz_path = find_npz(dataset_path)
        data_array, pmi = load_npz(npz_path, args.data_key)
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

        plot_path = os.path.join(args.output_dir, f"{subdir}_pmi_cosine.png")
        plot_scatter(pmi, cosines, plot_path, title=f"{subdir} (estimator={args.estimator})")

        if args.save_numpy:
            np.save(os.path.join(args.output_dir, f"{subdir}_cosines.npy"), cosines)

        mean_cos = float(cosines.mean())
        print(f"[{subdir}] saved plot -> {plot_path}; mean cosine: {mean_cos:.4f}")


if __name__ == "__main__":
    main()
