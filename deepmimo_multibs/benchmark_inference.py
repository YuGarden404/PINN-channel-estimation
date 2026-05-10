import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_cbdnet_baseline import CBDNetChannelEstimator, build_rss as build_token_rss
from train_cross_attention_baseline import CrossAttentionResidualNet
from train_nc_adapter import NoiseConditionedAdapter
from train_noise_aware_baseline import NoiseAwareResidualNet
from train_rss_baseline import ResidualMLP, build_rss as build_flat_rss, complex_to_real_flat


def complex_to_tokens(arr):
    arr = arr.reshape(arr.shape[0], -1)
    return np.stack([arr.real, arr.imag], axis=-1).astype(np.float32)


def pick_ls_file(data_dir, ls_file):
    if ls_file:
        return data_dir / ls_file
    candidates = sorted(data_dir.glob("ls_target_snr*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No ls_target_snr*.npy found in {data_dir}")
    return candidates[0]


def parameter_count(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def checkpoint_size_mb(path):
    return path.stat().st_size / (1024 * 1024)


def make_inputs(args):
    data_dir = Path(args.data_dir)
    channel = np.load(data_dir / "channel_target.npy")
    ls_channel = np.load(pick_ls_file(data_dir, args.ls_file))
    rss_multi = np.load(data_dir / "rss_multibs_normalized.npy")

    n_samples = min(len(channel), len(ls_channel), len(rss_multi), args.num_samples)
    channel = channel[:n_samples]
    ls_channel = ls_channel[:n_samples]
    rss_multi = rss_multi[:n_samples]

    if args.model == "mlp":
        target_real = complex_to_real_flat(channel)
        ls_real = complex_to_real_flat(ls_channel)
        scale = max(np.max(np.abs(target_real)), np.max(np.abs(ls_real)), 1e-12)
        ls_real = ls_real / scale
        rss = build_flat_rss(rss_multi, args.mode, args.seed)
        x = torch.from_numpy(ls_real[: args.batch_size]).float()
        r = torch.from_numpy(rss[: args.batch_size]).float()
        return x, r, {"channel_dim": int(ls_real.shape[1]), "tokens": int(ls_real.shape[1] // 2)}

    target_tokens = complex_to_tokens(channel)
    ls_tokens = complex_to_tokens(ls_channel)
    scale = max(np.max(np.abs(target_tokens)), np.max(np.abs(ls_tokens)), 1e-12)
    ls_tokens = ls_tokens / scale
    rss = build_token_rss(rss_multi, args.mode, args.seed)
    x = torch.from_numpy(ls_tokens[: args.batch_size]).float()
    r = torch.from_numpy(rss[: args.batch_size]).float()
    return x, r, {"channel_tokens": int(ls_tokens.shape[1])}


def build_model(args, input_meta, rss_dim):
    if args.model == "mlp":
        return ResidualMLP(
            channel_dim=input_meta["channel_dim"],
            rss_dim=rss_dim,
            hidden_dim=args.hidden_dim_mlp,
            depth=args.depth_mlp,
        )
    if args.model == "cross_attention":
        return CrossAttentionResidualNet(
            num_channel_tokens=input_meta["channel_tokens"],
            rss_dim=rss_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth_cross,
        )
    if args.model == "noise_aware":
        return NoiseAwareResidualNet(
            num_channel_tokens=input_meta["channel_tokens"],
            rss_dim=rss_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth_cross,
        )
    if args.model == "cbdnet":
        return CBDNetChannelEstimator(
            num_channel_tokens=input_meta["channel_tokens"],
            rss_dim=rss_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth_cbdnet,
        )
    if args.model == "nc_adapter":
        if not args.base_checkpoint:
            raise ValueError("--base-checkpoint is required for --model nc_adapter")
        base_model = CrossAttentionResidualNet(
            num_channel_tokens=input_meta["channel_tokens"],
            rss_dim=rss_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            depth=args.depth_cross,
        )
        base_model.load_state_dict(torch.load(args.base_checkpoint, map_location="cpu"))
        return NoiseConditionedAdapter(
            base_model=base_model,
            num_channel_tokens=input_meta["channel_tokens"],
            hidden_dim=args.adapter_hidden_dim,
            num_heads=args.adapter_num_heads,
            depth=args.adapter_depth,
            dropout=args.dropout,
            adapter_scale=args.adapter_scale,
            freeze_base=not args.finetune_base,
        )
    raise ValueError(args.model)


def forward_model(model, args, x, rss):
    if args.model == "mlp":
        return model(x, rss)
    return model(x, rss)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model, args, x, rss, device):
    model.eval()
    x = x.to(device)
    rss = rss.to(device)

    with torch.no_grad():
        for _ in range(args.warmup):
            forward_model(model, args, x, rss)
        sync(device)

        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            forward_model(model, args, x, rss)
            sync(device)
            times.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    return {
        "latency_ms_mean_per_batch": float(arr.mean()),
        "latency_ms_std_per_batch": float(arr.std(ddof=0)),
        "latency_ms_p50_per_batch": float(np.percentile(arr, 50)),
        "latency_ms_p95_per_batch": float(np.percentile(arr, 95)),
        "latency_us_mean_per_sample": float(arr.mean() * 1000.0 / args.batch_size),
        "throughput_samples_per_sec": float(args.batch_size * 1000.0 / arr.mean()),
    }


def profile_flops(model, args, x, rss, device):
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
            + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            with_flops=True,
            record_shapes=False,
        ) as prof:
            with torch.no_grad():
                forward_model(model, args, x.to(device), rss.to(device))
                sync(device)
        total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        return {
            "flops_per_batch": int(total_flops),
            "flops_per_sample": float(total_flops / args.batch_size),
        }
    except Exception as exc:
        return {"flops_error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description="Benchmark DeepMIMO model inference latency.")
    parser.add_argument(
        "--model",
        choices=["mlp", "cross_attention", "noise_aware", "cbdnet", "nc_adapter"],
        required=True,
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--mode", default="ls_only")
    parser.add_argument("--ls-file", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--depth-cross", type=int, default=2)
    parser.add_argument("--depth-cbdnet", type=int, default=3)
    parser.add_argument("--hidden-dim-mlp", type=int, default=256)
    parser.add_argument("--depth-mlp", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--adapter-hidden-dim", type=int, default=64)
    parser.add_argument("--adapter-num-heads", type=int, default=4)
    parser.add_argument("--adapter-depth", type=int, default=1)
    parser.add_argument("--adapter-scale", type=float, default=0.1)
    parser.add_argument("--finetune-base", action="store_true")
    parser.add_argument("--profile-flops", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    x, rss, input_meta = make_inputs(args)
    model = build_model(args, input_meta, rss_dim=rss.shape[1]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    total_params, trainable_params = parameter_count(model)
    result = {
        "model": args.model,
        "mode": args.mode,
        "data_dir": args.data_dir,
        "checkpoint": str(checkpoint),
        "base_checkpoint": args.base_checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "input_meta": input_meta,
        "rss_dim": int(rss.shape[1]),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "checkpoint_size_mb": checkpoint_size_mb(checkpoint),
    }
    result.update(benchmark(model, args, x, rss, device))
    if args.profile_flops:
        result.update(profile_flops(model, args, x, rss, device))

    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
