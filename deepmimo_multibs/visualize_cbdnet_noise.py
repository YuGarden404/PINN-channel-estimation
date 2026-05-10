import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_cbdnet_baseline import (
    CBDNetChannelEstimator,
    build_rss,
    complex_to_tokens,
    find_ls_file,
    make_split,
    nmse,
)


def token_energy(tokens):
    return np.sum(tokens**2, axis=-1)


def sample_energy(tokens):
    return np.mean(token_energy(tokens), axis=1)


def pearson_corr(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_arrays(data_dir, ls_file, mode, seed):
    channel = np.load(data_dir / "channel_target.npy")
    ls_path = find_ls_file(data_dir, ls_file)
    ls_channel = np.load(ls_path)
    rss_multi = np.load(data_dir / "rss_multibs_normalized.npy")

    n_samples = min(len(channel), len(ls_channel), len(rss_multi))
    channel = channel[:n_samples]
    ls_channel = ls_channel[:n_samples]
    rss_multi = rss_multi[:n_samples]

    target_tokens = complex_to_tokens(channel)
    ls_tokens = complex_to_tokens(ls_channel)
    scale = max(np.max(np.abs(target_tokens)), np.max(np.abs(ls_tokens)), 1e-12)
    target_tokens = target_tokens / scale
    ls_tokens = ls_tokens / scale
    rss = build_rss(rss_multi, mode, seed)
    return ls_tokens, target_tokens, rss, ls_path


def select_indices(n_samples, split, seed, max_samples):
    splits = make_split(n_samples, seed=seed)
    if split == "all":
        indices = np.arange(n_samples)
    else:
        indices = splits[split]
    if max_samples > 0:
        indices = indices[:max_samples]
    return indices


def run_model(model, ls_tokens, rss, target_tokens, device, batch_size):
    preds = []
    noise_maps = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(ls_tokens), batch_size):
            end = start + batch_size
            ls_batch = torch.from_numpy(ls_tokens[start:end]).float().to(device)
            rss_batch = torch.from_numpy(rss[start:end]).float().to(device)
            pred, noise_map = model(ls_batch, rss_batch, return_noise=True)
            preds.append(pred.cpu().numpy())
            noise_maps.append(noise_map.cpu().numpy())

    pred_tokens = np.concatenate(preds, axis=0)
    noise_map = np.concatenate(noise_maps, axis=0)

    with torch.no_grad():
        pred_t = torch.from_numpy(pred_tokens).float()
        target_t = torch.from_numpy(target_tokens).float()
        ls_t = torch.from_numpy(ls_tokens).float()
        pred_nmse = float(nmse(pred_t, target_t).item())
        ls_nmse = float(nmse(ls_t, target_t).item())

    return pred_tokens, noise_map, ls_nmse, pred_nmse


def save_energy_bars(metrics, out_path):
    labels = [
        "True noise",
        "Pred noise",
        "LS error",
        "Refined error",
        "Residual",
    ]
    values = [
        metrics["true_noise_energy_mean"],
        metrics["pred_noise_energy_mean"],
        metrics["ls_error_energy_mean"],
        metrics["refined_error_energy_mean"],
        metrics["residual_energy_mean"],
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Mean token energy")
    ax.set_title("Noise / Residual Energy Summary")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_noise_scatter(true_energy, pred_energy, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(true_energy, pred_energy, s=12, alpha=0.45, edgecolors="none")
    lo = min(float(np.min(true_energy)), float(np.min(pred_energy)))
    hi = max(float(np.max(true_energy)), float(np.max(pred_energy)))
    ax.plot([lo, hi], [lo, hi], color="#e45756", linewidth=1.5, label="y = x")
    ax.set_xlabel("True noise energy")
    ax.set_ylabel("Predicted noise-map energy")
    ax.set_title("Noise Energy Correlation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_error_scatter(ls_error, refined_error, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(ls_error, refined_error, s=12, alpha=0.45, edgecolors="none")
    lo = min(float(np.min(ls_error)), float(np.min(refined_error)))
    hi = max(float(np.max(ls_error)), float(np.max(refined_error)))
    ax.plot([lo, hi], [lo, hi], color="#e45756", linewidth=1.5, label="y = x")
    ax.set_xlabel("LS error energy")
    ax.set_ylabel("Refined error energy")
    ax.set_title("Per-sample Error Before / After Refinement")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_token_heatmaps(true_noise, pred_noise, residual, refined_error, out_path, max_rows):
    rows = min(max_rows, true_noise.shape[0])
    panels = [
        ("True noise", token_energy(true_noise[:rows])),
        ("Predicted noise map", token_energy(pred_noise[:rows])),
        ("Predicted residual", token_energy(residual[:rows])),
        ("Refined error", token_energy(refined_error[:rows])),
    ]

    vmax = max(float(np.percentile(values, 98)) for _, values in panels)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (title, values) in zip(axes.ravel(), panels):
        image = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Channel token")
        ax.set_ylabel("Sample")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize CBDNet-style noise maps and residuals.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--mode", choices=["ls_only", "single_rss", "multibs_rss", "zero_rss", "shuffled_rss"], default="ls_only")
    parser.add_argument("--ls-file", default=None)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--heatmap-samples", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--no-rss-context", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint.parent / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    ls_tokens_all, target_tokens_all, rss_all, ls_path = load_arrays(
        data_dir=data_dir,
        ls_file=args.ls_file,
        mode=args.mode,
        seed=args.seed,
    )
    indices = select_indices(len(ls_tokens_all), args.split, args.seed, args.max_samples)
    ls_tokens = ls_tokens_all[indices]
    target_tokens = target_tokens_all[indices]
    rss = rss_all[indices]

    model = CBDNetChannelEstimator(
        num_channel_tokens=ls_tokens.shape[1],
        rss_dim=rss.shape[1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
        use_rss_context=not args.no_rss_context,
    ).to(args.device)
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state)

    pred_tokens, noise_map, ls_nmse, pred_nmse = run_model(
        model=model,
        ls_tokens=ls_tokens,
        rss=rss,
        target_tokens=target_tokens,
        device=args.device,
        batch_size=args.batch_size,
    )

    true_noise = target_tokens - ls_tokens
    residual = pred_tokens - ls_tokens
    ls_error = ls_tokens - target_tokens
    refined_error = pred_tokens - target_tokens

    true_noise_sample_energy = sample_energy(true_noise)
    pred_noise_sample_energy = sample_energy(noise_map)
    ls_error_sample_energy = sample_energy(ls_error)
    refined_error_sample_energy = sample_energy(refined_error)

    metrics = {
        "data_dir": str(data_dir),
        "checkpoint": str(checkpoint),
        "mode": args.mode,
        "split": args.split,
        "n_samples_visualized": int(len(indices)),
        "ls_file": str(ls_path),
        "ls_nmse": ls_nmse,
        "refined_nmse": pred_nmse,
        "nmse_gain": float(ls_nmse - pred_nmse),
        "true_noise_energy_mean": float(np.mean(true_noise_sample_energy)),
        "pred_noise_energy_mean": float(np.mean(pred_noise_sample_energy)),
        "ls_error_energy_mean": float(np.mean(ls_error_sample_energy)),
        "refined_error_energy_mean": float(np.mean(refined_error_sample_energy)),
        "residual_energy_mean": float(np.mean(sample_energy(residual))),
        "noise_energy_corr": pearson_corr(true_noise_sample_energy, pred_noise_sample_energy),
        "error_energy_corr_before_after": pearson_corr(ls_error_sample_energy, refined_error_sample_energy),
    }

    save_energy_bars(metrics, out_dir / "energy_summary.png")
    save_noise_scatter(true_noise_sample_energy, pred_noise_sample_energy, out_dir / "noise_energy_scatter.png")
    save_error_scatter(ls_error_sample_energy, refined_error_sample_energy, out_dir / "error_before_after_scatter.png")
    save_token_heatmaps(
        true_noise=true_noise,
        pred_noise=noise_map,
        residual=residual,
        refined_error=refined_error,
        out_path=out_dir / "token_heatmaps.png",
        max_rows=args.heatmap_samples,
    )

    with open(out_dir / "visualization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
