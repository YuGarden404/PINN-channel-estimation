import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ChannelRSSDataset(Dataset):
    def __init__(self, ls_real, target_real, rss, indices):
        self.ls_real = ls_real
        self.target_real = target_real
        self.rss = rss
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            torch.from_numpy(self.ls_real[real_idx]).float(),
            torch.from_numpy(self.rss[real_idx]).float(),
            torch.from_numpy(self.target_real[real_idx]).float(),
        )


class ResidualMLP(nn.Module):
    def __init__(self, channel_dim, rss_dim, hidden_dim=256, depth=3):
        super().__init__()
        input_dim = channel_dim + rss_dim
        layers = []
        dim = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)])
            dim = hidden_dim
        layers.append(nn.Linear(dim, channel_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, ls_real, rss):
        residual = self.net(torch.cat([ls_real, rss], dim=1))
        return ls_real + residual


def complex_to_real_flat(arr):
    arr = arr.reshape(arr.shape[0], -1)
    return np.concatenate([arr.real, arr.imag], axis=1).astype(np.float32)


def nmse(pred_real, target_real):
    n = pred_real.shape[1] // 2
    pred = torch.complex(pred_real[:, :n], pred_real[:, n:])
    target = torch.complex(target_real[:, :n], target_real[:, n:])
    err = torch.sum(torch.abs(pred - target) ** 2, dim=1)
    power = torch.sum(torch.abs(target) ** 2, dim=1).clamp_min(1e-12)
    return torch.mean(err / power)


def make_split(n_samples, seed=42, train_ratio=0.8, val_ratio=0.1):
    np.random.seed(seed)
    indices = np.random.permutation(n_samples)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def build_rss(rss_multi, mode, seed):
    rss_multi = np.nan_to_num(rss_multi, nan=0.0, neginf=0.0, posinf=0.0).astype(np.float32)
    if mode == "ls_only":
        return np.zeros((len(rss_multi), 0), dtype=np.float32)
    if mode == "single_rss":
        return rss_multi[:, :1].astype(np.float32)
    if mode == "multibs_rss":
        return rss_multi.astype(np.float32)
    if mode == "zero_rss":
        return np.zeros_like(rss_multi, dtype=np.float32)
    if mode == "shuffled_rss":
        rng = np.random.default_rng(seed)
        return rss_multi[rng.permutation(len(rss_multi))].astype(np.float32)
    raise ValueError(f"Unknown RSS mode: {mode}")


def run_epoch(model, loader, optimizer, device, train):
    model.train(train)
    total_nmse = 0.0
    for ls_real, rss, target_real in loader:
        ls_real = ls_real.to(device)
        rss = rss.to(device)
        target_real = target_real.to(device)
        with torch.set_grad_enabled(train):
            pred = model(ls_real, rss)
            loss = nmse(pred, target_real)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_nmse += loss.item()
    return total_nmse / len(loader)


def evaluate(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for ls_real, rss, target_real in loader:
            ls_real = ls_real.to(device)
            rss = rss.to(device)
            target_real = target_real.to(device)
            values.append(nmse(model(ls_real, rss), target_real).item())
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description="Train DeepMIMO RSS ablation baselines.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["ls_only", "single_rss", "multibs_rss", "zero_rss", "shuffled_rss"],
        required=True,
    )
    parser.add_argument("--ls-file", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    channel = np.load(data_dir / "channel_target.npy")
    if args.ls_file is None:
        ls_candidates = sorted(data_dir.glob("ls_target_snr*.npy"))
        if not ls_candidates:
            raise FileNotFoundError("No ls_target_snr*.npy file found.")
        ls_path = ls_candidates[0]
    else:
        ls_path = data_dir / args.ls_file
    ls_channel = np.load(ls_path)
    rss_multi = np.load(data_dir / "rss_multibs_normalized.npy")

    n_samples = min(len(channel), len(ls_channel), len(rss_multi))
    channel = channel[:n_samples]
    ls_channel = ls_channel[:n_samples]
    rss_multi = rss_multi[:n_samples]

    target_real = complex_to_real_flat(channel)
    ls_real = complex_to_real_flat(ls_channel)
    scale = max(np.max(np.abs(target_real)), np.max(np.abs(ls_real)), 1e-12)
    target_real = target_real / scale
    ls_real = ls_real / scale
    rss = build_rss(rss_multi, args.mode, args.seed)

    splits = make_split(n_samples, seed=args.seed)
    train_ds = ChannelRSSDataset(ls_real, target_real, rss, splits["train"])
    val_ds = ChannelRSSDataset(ls_real, target_real, rss, splits["val"])
    test_ds = ChannelRSSDataset(ls_real, target_real, rss, splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = ResidualMLP(
        channel_dim=ls_real.shape[1],
        rss_dim=rss.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "runs" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / "best_model.pth"

    print(json.dumps({
        "mode": args.mode,
        "data_dir": str(data_dir),
        "n_samples": n_samples,
        "channel_dim": int(ls_real.shape[1]),
        "rss_dim": int(rss.shape[1]),
        "device": args.device,
        "ls_file": str(ls_path),
    }, indent=2))

    for epoch in range(args.epochs):
        train_nmse = run_epoch(model, train_loader, optimizer, args.device, train=True)
        val_nmse = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch + 1}/{args.epochs}: Train NMSE={train_nmse:.6f}, Val NMSE={val_nmse:.6f}")
        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with Val NMSE={best_val:.6f}")

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_nmse = evaluate(model, test_loader, args.device)
    result = {
        "mode": args.mode,
        "best_val_nmse": best_val,
        "test_nmse": test_nmse,
        "test_db": float(10 * np.log10(test_nmse)),
        "best_model": str(best_path),
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
