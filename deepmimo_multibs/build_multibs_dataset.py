import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from deepmimo_multibs.ls_ofdm import LSOFDMChannelEstimator


def _as_array(value):
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(value.tolist())
    return arr


def compute_rss_from_channels(channels_by_bs):
    """Compute a simple multi-BS RSS/pathloss proxy from channel power.

    Input shape for each BS is expected as `(n_user, n_tap, n_rx, n_tx)`.
    Output shape is `(n_user, n_bs)`, normalized in dB per BS.
    """

    rss_columns = []
    for channels in channels_by_bs:
        power = np.mean(np.abs(channels) ** 2, axis=(1, 2, 3))
        power_db = 10.0 * np.log10(np.maximum(power, 1e-30))
        rss_columns.append(power_db)

    rss = np.stack(rss_columns, axis=1)
    mean = rss.mean(axis=0, keepdims=True)
    std = rss.std(axis=0, keepdims=True) + 1e-8
    return (rss - mean) / std, rss


def load_deepmimo_channels(args):
    """Load/compute channels with DeepMIMO.

    DeepMIMO APIs differ across versions. This function intentionally fails with
    a clear message if the installed version exposes a different interface.
    """

    try:
        import deepmimo as dm
    except ImportError as exc:
        raise SystemExit("DeepMIMO is not installed. Run: pip install DeepMIMO") from exc

    if args.download and hasattr(dm, "download"):
        dm.download(args.scenario)

    dataset = dm.load(args.scenario)

    if not hasattr(dm, "ChannelParameters") or not hasattr(dataset, "compute_channels"):
        raise RuntimeError(
            "This script expects DeepMIMO v4-style APIs: dm.ChannelParameters and "
            "dataset.compute_channels(...). Run inspect_deepmimo.py and adapt "
            "load_deepmimo_channels() to your installed DeepMIMO version."
        )

    ch_params = dm.ChannelParameters()
    ch_params.num_paths = args.num_paths
    ch_params.num_time_steps = 1
    ch_params.freq_domain = False

    channels_by_bs = []
    for bs_id in args.bs_ids:
        channels = dataset.compute_channels(
            tx_ids=[bs_id],
            rx_ids="all",
            channel_params=ch_params,
        )
        channels = _as_array(channels)
        channels_by_bs.append(channels[: args.num_users] if args.num_users else channels)

    positions = None
    for candidate in ("rx_pos", "ue_pos", "rx_positions", "ue_positions"):
        if hasattr(dataset, candidate):
            positions = _as_array(getattr(dataset, candidate))
            if args.num_users:
                positions = positions[: args.num_users]
            break

    return channels_by_bs, positions


def load_deepmimo_matrices(args):
    """Load DeepMIMO fundamental matrices for a first multi-BS RSS dataset.

    The `o1_60` MacroDataset already exposes matrices such as power/pathloss and
    rx_pos. This path builds RSS fingerprints from those matrices before we
    commit to a specific channel-generation API.
    """

    try:
        import deepmimo as dm
    except ImportError as exc:
        raise SystemExit("DeepMIMO is not installed. Run: pip install DeepMIMO") from exc

    if args.download and hasattr(dm, "download"):
        dm.download(args.scenario)

    dataset = dm.load(args.scenario)
    if not hasattr(dataset, "power") and not hasattr(dataset, "pathloss"):
        raise RuntimeError("Dataset exposes neither power nor pathloss matrices.")

    source_name = "power" if hasattr(dataset, "power") else "pathloss"
    source = getattr(dataset, source_name)

    if args.pair_indices:
        matrices = [np.asarray(source[idx]) for idx in args.pair_indices]
    elif isinstance(source, list):
        matrices = [np.asarray(item) for item in source[: len(args.bs_ids)]]
    else:
        arr = np.asarray(source)
        matrices = [arr] if arr.ndim == 2 else [arr[idx] for idx in range(min(arr.shape[0], len(args.bs_ids)))]

    if len(matrices) < len(args.bs_ids):
        raise RuntimeError(f"Only found {len(matrices)} matrices for {len(args.bs_ids)} BS ids.")

    n_rx = min(mat.shape[0] for mat in matrices)
    if args.num_users:
        n_rx = min(n_rx, args.num_users)

    rss_cols = []
    matrix_shapes = []
    for mat in matrices[: len(args.bs_ids)]:
        matrix_shapes.append(list(mat.shape))
        mat = mat[:n_rx]
        if source_name == "pathloss":
            # Pathloss is positive attenuation; larger received RSS means smaller pathloss.
            if mat.ndim == 1:
                col = -mat
            elif mat.ndim == 2:
                col = -np.min(mat, axis=1)
            else:
                raise RuntimeError(f"Expected pathloss [num_rx] or [num_rx, num_paths], got {mat.shape}.")
        else:
            # power is per-path received power in dBW. Sum linear path powers.
            if mat.ndim != 2:
                raise RuntimeError(f"Expected per-pair {source_name} matrix [num_rx, num_paths], got {mat.shape}.")
            mat = np.nan_to_num(mat, nan=-300.0, neginf=-300.0, posinf=0.0)
            col = 10.0 * np.log10(np.maximum(np.sum(10.0 ** (mat / 10.0), axis=1), 1e-30))
        rss_cols.append(col)

    rss_db = np.stack(rss_cols, axis=1)

    if args.num_users:
        rss_db = rss_db[: args.num_users]

    rss_db = np.nan_to_num(rss_db, nan=-300.0, neginf=-300.0, posinf=0.0).astype(np.float32)
    rss_norm = (rss_db - rss_db.mean(axis=0, keepdims=True)) / (rss_db.std(axis=0, keepdims=True) + 1e-8)
    rss_norm = np.nan_to_num(rss_norm, nan=0.0, neginf=0.0, posinf=0.0).astype(np.float32)

    positions = None
    if hasattr(dataset, "rx_pos"):
        rx_pos = dataset.rx_pos
        if args.pair_indices and isinstance(rx_pos, list):
            positions = np.asarray(rx_pos[args.pair_indices[0]])[:n_rx]
        elif isinstance(rx_pos, list):
            positions = np.asarray(rx_pos[0])[:n_rx]
        else:
            positions = np.asarray(rx_pos)[:n_rx]

    return rss_norm, rss_db, positions, source_name, matrix_shapes


def main():
    parser = argparse.ArgumentParser(description="Build a multi-BS DeepMIMO ablation dataset.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--bs-ids", nargs="+", type=int, required=True)
    parser.add_argument("--target-bs", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-users", type=int, default=0, help="0 means all users.")
    parser.add_argument("--num-paths", type=int, default=25)
    parser.add_argument("--snr", type=float, default=0.0)
    parser.add_argument("--pilot-spacing", type=int, default=4)
    parser.add_argument("--n-subcarriers", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pair-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Indices into DeepMIMO's loaded TX/RX pair lists. Use indices with "
            "the same RX set and different TX sets to form a multi-BS fingerprint."
        ),
    )
    parser.add_argument(
        "--matrices-only",
        action="store_true",
        help="Only build multi-BS RSS/pathloss arrays from DeepMIMO matrices; skip channel/LS generation.",
    )
    args = parser.parse_args()

    if args.target_bs not in args.bs_ids:
        raise ValueError("--target-bs must be included in --bs-ids")

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.matrices_only:
        rss_norm, rss_db, positions, source_name, matrix_shapes = load_deepmimo_matrices(args)
        np.save(out_dir / "rss_multibs_normalized.npy", rss_norm)
        np.save(out_dir / "rss_multibs_db.npy", rss_db)
        if positions is not None:
            np.save(out_dir / "user_positions.npy", positions)
        manifest = {
            "scenario": args.scenario,
            "bs_ids": args.bs_ids,
            "target_bs": args.target_bs,
            "num_users": int(rss_norm.shape[0]),
            "rss_shape": list(rss_norm.shape),
            "source_matrix": source_name,
            "source_matrix_shapes": matrix_shapes,
            "pair_indices": args.pair_indices,
            "seed": args.seed,
            "matrices_only": True,
            "notes": "First-stage DeepMIMO inspection dataset: RSS/pathloss fingerprint only; no channel/LS arrays.",
        }
        with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(json.dumps(manifest, indent=2))
        return

    channels_by_bs, positions = load_deepmimo_channels(args)
    target_idx = args.bs_ids.index(args.target_bs)
    target_channels = channels_by_bs[target_idx]

    if target_channels.ndim != 4:
        raise RuntimeError(
            f"Expected target channel shape (n_user, n_tap, n_rx, n_tx), got {target_channels.shape}."
        )

    rss_norm, rss_db = compute_rss_from_channels(channels_by_bs)
    estimator = LSOFDMChannelEstimator(
        n_tap=target_channels.shape[1],
        n_rx=target_channels.shape[2],
        n_tx=target_channels.shape[3],
        n_subcarriers=args.n_subcarriers,
        pilot_spacing=args.pilot_spacing,
        snr_db=args.snr,
    )
    ls_target = estimator.estimate_batch(target_channels)

    np.save(out_dir / "channels_target.npy", target_channels)
    np.save(out_dir / "ls_target.npy", ls_target)
    np.save(out_dir / "rss_multibs_normalized.npy", rss_norm)
    np.save(out_dir / "rss_multibs_db.npy", rss_db)
    if positions is not None:
        np.save(out_dir / "user_positions.npy", positions)

    manifest = {
        "scenario": args.scenario,
        "bs_ids": args.bs_ids,
        "target_bs": args.target_bs,
        "num_users": int(target_channels.shape[0]),
        "channel_shape": list(target_channels.shape),
        "rss_shape": list(rss_norm.shape),
        "snr": args.snr,
        "pilot_spacing": args.pilot_spacing,
        "n_subcarriers": args.n_subcarriers,
        "seed": args.seed,
        "notes": "RSS is a channel-power proxy until scenario-native RSS maps are integrated.",
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
