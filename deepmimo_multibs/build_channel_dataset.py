import argparse
import json
from pathlib import Path

import numpy as np


def add_complex_awgn(channel, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.abs(channel) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (
        rng.standard_normal(channel.shape) + 1j * rng.standard_normal(channel.shape)
    )
    return (channel + noise).astype(np.complex64)


def select_channel(channel_source, target_pair_index):
    """Handle both pair-list channels and single-pair DeepMIMO channel arrays."""
    if isinstance(channel_source, list):
        channel = np.asarray(channel_source[target_pair_index])
    else:
        arr = np.asarray(channel_source)
        if arr.ndim >= 4:
            channel = arr
        elif arr.dtype == object:
            channel = np.asarray(arr.tolist()[target_pair_index])
        else:
            raise RuntimeError(f"Unsupported channel array shape: {arr.shape}")
    return channel.astype(np.complex64)


def select_valid_indices(channel, num_users, min_power_quantile, sort_by_power):
    power = np.mean(np.abs(channel) ** 2, axis=tuple(range(1, channel.ndim)))
    indices = np.arange(len(channel))
    threshold = None
    if min_power_quantile > 0:
        threshold = float(np.quantile(power, min_power_quantile))
        indices = indices[power >= threshold]
    if sort_by_power == "descending":
        order = np.argsort(power[indices])[::-1]
        indices = indices[order]
    elif sort_by_power == "ascending":
        order = np.argsort(power[indices])
        indices = indices[order]
    if num_users:
        indices = indices[:num_users]
    return indices, power, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Extract target-BS DeepMIMO channels and build noisy LS-like inputs."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--target-pair-index", type=int, required=True)
    parser.add_argument("--rss-dir", required=True, help="Directory produced by build_multibs_dataset.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-users", type=int, default=5000)
    parser.add_argument(
        "--min-power-quantile",
        type=float,
        default=0.0,
        help="Drop the weakest channel samples by average channel power before taking --num-users.",
    )
    parser.add_argument(
        "--sort-by-power",
        choices=["none", "ascending", "descending"],
        default="none",
        help="Optionally sort samples by average channel power before taking --num-users.",
    )
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        import deepmimo as dm
    except ImportError as exc:
        raise SystemExit("DeepMIMO is not installed. Run: pip install DeepMIMO") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rss_dir = Path(args.rss_dir)

    dataset = dm.load(args.scenario)
    full_channel = select_channel(dataset.channel, args.target_pair_index)
    selected_indices, channel_power, power_threshold = select_valid_indices(
        full_channel,
        num_users=args.num_users,
        min_power_quantile=args.min_power_quantile,
        sort_by_power=args.sort_by_power,
    )
    channel = full_channel[selected_indices]
    ls_channel = add_complex_awgn(channel, snr_db=args.snr, seed=args.seed)

    np.save(out_dir / "channel_target.npy", channel)
    np.save(out_dir / f"ls_target_snr{args.snr:g}.npy", ls_channel)

    copied = {}
    for name in ["rss_multibs_normalized.npy", "rss_multibs_db.npy", "user_positions.npy"]:
        src = rss_dir / name
        if src.exists():
            arr = np.load(src)[selected_indices]
            np.save(out_dir / name, arr)
            copied[name] = list(arr.shape)

    manifest = {
        "scenario": args.scenario,
        "target_pair_index": args.target_pair_index,
        "num_users": int(channel.shape[0]),
        "channel_shape": list(channel.shape),
        "source_num_users": int(full_channel.shape[0]),
        "min_power_quantile": args.min_power_quantile,
        "sort_by_power": args.sort_by_power,
        "power_threshold": power_threshold,
        "selected_power_mean": float(np.mean(channel_power[selected_indices])),
        "selected_power_min": float(np.min(channel_power[selected_indices])),
        "selected_power_max": float(np.max(channel_power[selected_indices])),
        "ls_file": f"ls_target_snr{args.snr:g}.npy",
        "snr": args.snr,
        "seed": args.seed,
        "rss_dir": str(rss_dir),
        "copied_arrays": copied,
        "notes": (
            "LS-like input is simulated by adding complex AWGN to target channel. "
            "This is a first DeepMIMO baseline, not the original LS-OFDM estimator."
        ),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
