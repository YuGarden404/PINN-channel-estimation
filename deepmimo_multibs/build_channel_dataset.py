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


def main():
    parser = argparse.ArgumentParser(
        description="Extract target-BS DeepMIMO channels and build noisy LS-like inputs."
    )
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--target-pair-index", type=int, required=True)
    parser.add_argument("--rss-dir", required=True, help="Directory produced by build_multibs_dataset.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-users", type=int, default=5000)
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
    channel = np.asarray(dataset.channel[args.target_pair_index])[: args.num_users]
    channel = channel.astype(np.complex64)
    ls_channel = add_complex_awgn(channel, snr_db=args.snr, seed=args.seed)

    np.save(out_dir / "channel_target.npy", channel)
    np.save(out_dir / f"ls_target_snr{args.snr:g}.npy", ls_channel)

    copied = {}
    for name in ["rss_multibs_normalized.npy", "rss_multibs_db.npy", "user_positions.npy"]:
        src = rss_dir / name
        if src.exists():
            arr = np.load(src)[: args.num_users]
            np.save(out_dir / name, arr)
            copied[name] = list(arr.shape)

    manifest = {
        "scenario": args.scenario,
        "target_pair_index": args.target_pair_index,
        "num_users": int(channel.shape[0]),
        "channel_shape": list(channel.shape),
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
