import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from train_cross_attention_baseline import CrossAttentionResidualNet


MODES = ["ls_only", "single_rss", "multibs_rss", "zero_rss", "shuffled_rss"]


class ChannelRSSDataset(Dataset):
    def __init__(self, ls_tokens, target_tokens, rss, indices):
        self.ls_tokens = ls_tokens
        self.target_tokens = target_tokens
        self.rss = rss
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            torch.from_numpy(self.ls_tokens[real_idx]).float(),
            torch.from_numpy(self.rss[real_idx]).float(),
            torch.from_numpy(self.target_tokens[real_idx]).float(),
        )


class NoiseConditionedAdapter(nn.Module):
    """Small noise-conditioned adapter on top of a trained cross-attention model."""

    def __init__(
        self,
        base_model,
        num_channel_tokens,
        hidden_dim=64,
        num_heads=4,
        depth=1,
        dropout=0.05,
        adapter_scale=0.1,
        freeze_base=True,
    ):
        super().__init__()
        self.base_model = base_model
        self.freeze_base = freeze_base
        self.adapter_scale = adapter_scale

        self.noise_in = nn.Linear(2, hidden_dim)
        self.noise_pos = nn.Parameter(torch.zeros(1, num_channel_tokens, hidden_dim))
        noise_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.noise_encoder = nn.TransformerEncoder(noise_layer, num_layers=1)
        self.noise_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        self.adapter_in = nn.Linear(8, hidden_dim)
        self.adapter_pos = nn.Parameter(torch.zeros(1, num_channel_tokens, hidden_dim))
        adapter_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.adapter_encoder = nn.TransformerEncoder(adapter_layer, num_layers=max(1, depth))
        self.correction_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )
        self.correction_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

        nn.init.normal_(self.noise_pos, std=0.02)
        nn.init.normal_(self.adapter_pos, std=0.02)
        nn.init.zeros_(self.correction_head[-1].weight)
        nn.init.zeros_(self.correction_head[-1].bias)
        if self.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_base:
            self.base_model.eval()
        return self

    def forward(self, ls_tokens, rss, return_aux=False):
        if self.freeze_base:
            with torch.no_grad():
                base_pred = self.base_model(ls_tokens, rss)
        else:
            base_pred = self.base_model(ls_tokens, rss)

        noise_feat = self.noise_in(ls_tokens) + self.noise_pos
        noise_feat = self.noise_encoder(noise_feat)
        noise_map = self.noise_head(noise_feat)

        base_residual = base_pred - ls_tokens
        adapter_input = torch.cat([ls_tokens, base_pred, base_residual, noise_map], dim=-1)
        adapter_feat = self.adapter_in(adapter_input) + self.adapter_pos
        adapter_feat = self.adapter_encoder(adapter_feat)
        correction = self.correction_gate(adapter_feat) * self.correction_head(adapter_feat)
        pred = base_pred + self.adapter_scale * correction

        if return_aux:
            return pred, noise_map, correction, base_pred
        return pred


def complex_to_tokens(arr):
    arr = arr.reshape(arr.shape[0], -1)
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)


def nmse(pred_tokens, target_tokens):
    pred = torch.complex(pred_tokens[..., 0], pred_tokens[..., 1])
    target = torch.complex(target_tokens[..., 0], target_tokens[..., 1])
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


def run_epoch(model, loader, optimizer, device, train, noise_weight, correction_weight):
    model.train(train)
    total_loss = 0.0
    total_nmse = 0.0
    total_noise = 0.0
    total_correction = 0.0
    total_base = 0.0

    for ls_tokens, rss, target_tokens in loader:
        ls_tokens = ls_tokens.to(device)
        rss = rss.to(device)
        target_tokens = target_tokens.to(device)
        true_noise = target_tokens - ls_tokens

        with torch.set_grad_enabled(train):
            pred, noise_map, correction, base_pred = model(ls_tokens, rss, return_aux=True)
            nmse_loss = nmse(pred, target_tokens)
            base_nmse = nmse(base_pred, target_tokens)
            noise_loss = torch.mean((noise_map - true_noise) ** 2)
            correction_loss = torch.mean(correction**2)
            loss = nmse_loss + noise_weight * noise_loss + correction_weight * correction_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [param for param in model.parameters() if param.requires_grad],
                    1.0,
                )
                optimizer.step()

        total_loss += loss.item()
        total_nmse += nmse_loss.item()
        total_base += base_nmse.item()
        total_noise += noise_loss.item()
        total_correction += correction_loss.item()

    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "nmse": total_nmse / n_batches,
        "base_nmse": total_base / n_batches,
        "noise": total_noise / n_batches,
        "correction": total_correction / n_batches,
    }


def evaluate(model, loader, device):
    model.eval()
    values = []
    base_values = []
    with torch.no_grad():
        for ls_tokens, rss, target_tokens in loader:
            ls_tokens = ls_tokens.to(device)
            rss = rss.to(device)
            target_tokens = target_tokens.to(device)
            pred, _, _, base_pred = model(ls_tokens, rss, return_aux=True)
            values.append(nmse(pred, target_tokens).item())
            base_values.append(nmse(base_pred, target_tokens).item())
    return float(np.mean(values)), float(np.mean(base_values))


def find_ls_file(data_dir, requested):
    if requested is not None:
        return data_dir / requested
    ls_candidates = sorted(data_dir.glob("ls_target_snr*.npy"))
    if not ls_candidates:
        raise FileNotFoundError(f"No ls_target_snr*.npy file found in {data_dir}")
    return ls_candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Train a noise-conditioned adapter over cross-attention.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--mode", choices=MODES, default="ls_only")
    parser.add_argument("--ls-file", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base-hidden-dim", type=int, default=128)
    parser.add_argument("--base-num-heads", type=int, default=4)
    parser.add_argument("--base-depth", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--adapter-scale", type=float, default=0.1)
    parser.add_argument("--noise-weight", type=float, default=0.001)
    parser.add_argument("--correction-weight", type=float, default=0.001)
    parser.add_argument("--finetune-base", action="store_true")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    channel = np.load(data_dir / "channel_target.npy")
    ls_path = find_ls_file(data_dir, args.ls_file)
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
    rss = build_rss(rss_multi, args.mode, args.seed)

    splits = make_split(n_samples, seed=args.seed)
    train_ds = ChannelRSSDataset(ls_tokens, target_tokens, rss, splits["train"])
    val_ds = ChannelRSSDataset(ls_tokens, target_tokens, rss, splits["val"])
    test_ds = ChannelRSSDataset(ls_tokens, target_tokens, rss, splits["test"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    base_model = CrossAttentionResidualNet(
        num_channel_tokens=ls_tokens.shape[1],
        rss_dim=rss.shape[1],
        hidden_dim=args.base_hidden_dim,
        num_heads=args.base_num_heads,
        depth=args.base_depth,
        dropout=args.dropout,
    ).to(args.device)
    base_model.load_state_dict(torch.load(args.base_checkpoint, map_location=args.device))

    model = NoiseConditionedAdapter(
        base_model=base_model,
        num_channel_tokens=ls_tokens.shape[1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
        adapter_scale=args.adapter_scale,
        freeze_base=not args.finetune_base,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    suffix = f"nc_adapter_{args.mode}_nw{str(args.noise_weight).replace('.', 'p')}_s{str(args.adapter_scale).replace('.', 'p')}"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "runs" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pth"
    best_val = float("inf")
    best_base_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    history = []

    print(
        json.dumps(
            {
                "model": "nc_adapter",
                "mode": args.mode,
                "data_dir": str(data_dir),
                "base_checkpoint": args.base_checkpoint,
                "n_samples": n_samples,
                "channel_tokens": int(ls_tokens.shape[1]),
                "rss_dim": int(rss.shape[1]),
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "adapter_scale": args.adapter_scale,
                "noise_weight": args.noise_weight,
                "correction_weight": args.correction_weight,
                "freeze_base": not args.finetune_base,
                "device": args.device,
                "ls_file": str(ls_path),
            },
            indent=2,
        )
    )

    initial_val_nmse, initial_base_val_nmse = evaluate(model, val_loader, args.device)
    best_val = initial_val_nmse
    best_base_val = initial_base_val_nmse
    best_epoch = 0
    torch.save(model.state_dict(), best_path)
    print(
        f"Saved initial baseline-preserving adapter with "
        f"Val NMSE={best_val:.6f}, Base Val NMSE={best_base_val:.6f}"
    )

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            train=True,
            noise_weight=args.noise_weight,
            correction_weight=args.correction_weight,
        )
        val_nmse, base_val_nmse = evaluate(model, val_loader, args.device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"Train Loss={train_metrics['loss']:.6f}, "
            f"Train NMSE={train_metrics['nmse']:.6f}, "
            f"Base NMSE={train_metrics['base_nmse']:.6f}, "
            f"Train Noise={train_metrics['noise']:.6f}, "
            f"Train Correction={train_metrics['correction']:.6f}, "
            f"Val NMSE={val_nmse:.6f}, "
            f"Base Val NMSE={base_val_nmse:.6f}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                **train_metrics,
                "val_nmse": val_nmse,
                "base_val_nmse": base_val_nmse,
            }
        )

        if val_nmse < best_val - args.min_delta:
            best_val = val_nmse
            best_base_val = base_val_nmse
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"Saved best adapter at epoch {best_epoch} with Val NMSE={best_val:.6f}")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                early_stopped = True
                print(
                    f"Early stopping at epoch {epoch + 1}. "
                    f"Best epoch={best_epoch}, Best Val NMSE={best_val:.6f}"
                )
                break

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_nmse, base_test_nmse = evaluate(model, test_loader, args.device)
    result = {
        "model": "nc_adapter",
        "mode": args.mode,
        "epochs_requested": args.epochs,
        "epochs_trained": len(history),
        "early_stopped": early_stopped,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "best_epoch": best_epoch,
        "best_val_nmse": best_val,
        "base_val_nmse_at_best": best_base_val,
        "test_nmse": test_nmse,
        "base_test_nmse": base_test_nmse,
        "delta_vs_base_test_nmse": test_nmse - base_test_nmse,
        "test_db": float(10 * np.log10(test_nmse)),
        "base_test_db": float(10 * np.log10(base_test_nmse)),
        "adapter_scale": args.adapter_scale,
        "noise_weight": args.noise_weight,
        "correction_weight": args.correction_weight,
        "freeze_base": not args.finetune_base,
        "best_model": str(best_path),
    }
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
