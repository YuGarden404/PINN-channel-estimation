import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


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


class CrossAttentionResidualNet(nn.Module):
    """Channel-token cross-attention with RSS-conditioned residual refinement.

    The LS-like channel is represented as one token per complex channel element,
    with two features per token: real and imaginary parts. RSS values are encoded
    as side-information tokens. Channel tokens query RSS tokens through
    cross-attention, then predict a residual correction gated by the RSS context.
    """

    def __init__(
        self,
        num_channel_tokens,
        rss_dim,
        hidden_dim=128,
        num_heads=4,
        depth=2,
        dropout=0.05,
        max_rss_tokens=16,
    ):
        super().__init__()
        self.rss_dim = rss_dim
        self.hidden_dim = hidden_dim

        self.channel_proj = nn.Linear(2, hidden_dim)
        self.channel_pos = nn.Parameter(torch.zeros(1, num_channel_tokens, hidden_dim))

        self.rss_value_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rss_pos = nn.Parameter(torch.zeros(1, max_rss_tokens, hidden_dim))
        self.null_rss_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_refiner = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.channel_pos, std=0.02)
        nn.init.normal_(self.rss_pos, std=0.02)
        nn.init.normal_(self.null_rss_token, std=0.02)

    def encode_rss(self, rss):
        batch_size = rss.shape[0]
        if self.rss_dim == 0:
            return self.null_rss_token.expand(batch_size, -1, -1)

        rss_tokens = self.rss_value_proj(rss.unsqueeze(-1))
        rss_tokens = rss_tokens + self.rss_pos[:, : self.rss_dim, :]
        return rss_tokens

    def forward(self, ls_tokens, rss):
        channel_tokens = self.channel_proj(ls_tokens) + self.channel_pos
        rss_tokens = self.encode_rss(rss)

        attended, _ = self.cross_attention(
            query=channel_tokens,
            key=rss_tokens,
            value=rss_tokens,
            need_weights=False,
        )
        fused = self.cross_norm(channel_tokens + attended)
        refined = self.channel_refiner(fused)

        residual = self.residual_head(refined)
        rss_context = rss_tokens.mean(dim=1)
        gate = self.gate_head(rss_context).unsqueeze(1)
        return ls_tokens + gate * residual


def complex_to_tokens(arr):
    arr = arr.reshape(arr.shape[0], -1)
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)


def nmse(pred_tokens, target_tokens):
    pred = torch.complex(pred_tokens[..., 0], pred_tokens[..., 1])
    target = torch.complex(target_tokens[..., 0], target_tokens[..., 1])
    err = torch.sum(torch.abs(pred - target) ** 2, dim=1)
    power = torch.sum(torch.abs(target) ** 2, dim=1).clamp_min(1e-12)
    return torch.mean(err / power)


def channel_power_db(tokens):
    channel = torch.complex(tokens[..., 0], tokens[..., 1])
    power = torch.sum(torch.abs(channel) ** 2, dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(power)


def rss_ranking_loss(pred_tokens, rss, target_rss_col=0, margin=0.05, min_rss_gap=0.05):
    """Batch-wise ranking constraint between target-BS RSS and predicted power.

    The current DeepMIMO dataset predicts only the target-BS channel. Therefore,
    the physically meaningful ranking signal is across users within a batch:
    if user i has stronger target-BS RSS than user j, the predicted target-BS
    channel power for i should also be larger than that of j.
    """
    if rss.shape[1] <= target_rss_col or rss.shape[0] < 2:
        return pred_tokens.new_zeros(())

    rss_target = rss[:, target_rss_col]
    power_db = channel_power_db(pred_tokens)

    rss_diff = rss_target.unsqueeze(1) - rss_target.unsqueeze(0)
    power_diff = power_db.unsqueeze(1) - power_db.unsqueeze(0)
    valid = torch.abs(rss_diff) > min_rss_gap
    if not torch.any(valid):
        return pred_tokens.new_zeros(())

    desired_sign = torch.sign(rss_diff[valid])
    signed_power_diff = desired_sign * power_diff[valid]
    return torch.relu(margin - signed_power_diff).mean()


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


def run_epoch(model, loader, optimizer, device, train, rank_weight, rank_margin, rank_min_gap):
    model.train(train)
    total_nmse = 0.0
    total_rank = 0.0
    total_loss = 0.0
    for ls_tokens, rss, target_tokens in loader:
        ls_tokens = ls_tokens.to(device)
        rss = rss.to(device)
        target_tokens = target_tokens.to(device)
        with torch.set_grad_enabled(train):
            pred = model(ls_tokens, rss)
            nmse_loss = nmse(pred, target_tokens)
            rank_loss = (
                rss_ranking_loss(
                    pred,
                    rss,
                    margin=rank_margin,
                    min_rss_gap=rank_min_gap,
                )
                if rank_weight > 0
                else pred.new_zeros(())
            )
            loss = nmse_loss + rank_weight * rank_loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_nmse += nmse_loss.item()
        total_rank += rank_loss.item()
        total_loss += loss.item()
    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "nmse": total_nmse / n_batches,
        "rank": total_rank / n_batches,
    }


def evaluate(model, loader, device):
    model.eval()
    values = []
    with torch.no_grad():
        for ls_tokens, rss, target_tokens in loader:
            ls_tokens = ls_tokens.to(device)
            rss = rss.to(device)
            target_tokens = target_tokens.to(device)
            values.append(nmse(model(ls_tokens, rss), target_tokens).item())
    return float(np.mean(values))


def find_ls_file(data_dir, requested):
    if requested is not None:
        return data_dir / requested
    ls_candidates = sorted(data_dir.glob("ls_target_snr*.npy"))
    if not ls_candidates:
        raise FileNotFoundError(f"No ls_target_snr*.npy file found in {data_dir}")
    return ls_candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Train a DeepMIMO cross-attention RSS baseline."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--ls-file", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.0,
        help="Weight for batch-wise RSS ranking loss. Use >0 only for real RSS modes.",
    )
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-min-gap", type=float, default=0.05)
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

    model = CrossAttentionResidualNet(
        num_channel_tokens=ls_tokens.shape[1],
        rss_dim=rss.shape[1],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        depth=args.depth,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else data_dir / "runs" / f"cross_attention_{args.mode}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = out_dir / "best_model.pth"

    print(
        json.dumps(
            {
                "model": "cross_attention_residual",
                "mode": args.mode,
                "data_dir": str(data_dir),
                "n_samples": n_samples,
                "channel_tokens": int(ls_tokens.shape[1]),
                "rss_dim": int(rss.shape[1]),
                "hidden_dim": args.hidden_dim,
                "num_heads": args.num_heads,
                "depth": args.depth,
                "rank_weight": args.rank_weight,
                "rank_margin": args.rank_margin,
                "rank_min_gap": args.rank_min_gap,
                "device": args.device,
                "ls_file": str(ls_path),
            },
            indent=2,
        )
    )

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            train=True,
            rank_weight=args.rank_weight,
            rank_margin=args.rank_margin,
            rank_min_gap=args.rank_min_gap,
        )
        val_nmse = evaluate(model, val_loader, args.device)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"Train Loss={train_metrics['loss']:.6f}, "
            f"Train NMSE={train_metrics['nmse']:.6f}, "
            f"Train Rank={train_metrics['rank']:.6f}, "
            f"Val NMSE={val_nmse:.6f}"
        )
        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model with Val NMSE={best_val:.6f}")

    if not best_path.exists():
        raise RuntimeError("No best model was saved. Check whether training produced NaN losses.")

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    test_nmse = evaluate(model, test_loader, args.device)
    result = {
        "model": "cross_attention_residual",
        "mode": args.mode,
        "best_val_nmse": best_val,
        "test_nmse": test_nmse,
        "test_db": float(10 * np.log10(test_nmse)),
        "rank_weight": args.rank_weight,
        "rank_margin": args.rank_margin,
        "rank_min_gap": args.rank_min_gap,
        "best_model": str(best_path),
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
