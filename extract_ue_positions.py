# extract_ue_positions.py
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_rx_from_inter_locs(text: str):
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)]
    if len(nums) < 3 or len(nums) % 3 != 0:
        raise ValueError("inter_locs cannot be parsed into 3D points")
    pts = np.asarray(nums, dtype=float).reshape(-1, 3)
    return pts[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-clean", default="ue_positions_clean.txt")
    parser.add_argument("--out-noisy", default="ue_positions_noisy.txt")
    parser.add_argument("--noise-std", type=float, default=3.0,
                        help="Horizontal Gaussian noise std in meters. Paper says N(0, 9I), so std=3.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, usecols=["inter_locs"])
    positions = np.vstack([extract_rx_from_inter_locs(x) for x in df["inter_locs"]])

    rng = np.random.default_rng(args.seed)
    noisy = positions.copy()
    noisy[:, :2] += rng.normal(0.0, args.noise_std, size=(len(noisy), 2))

    for path, arr in [(args.out_clean, positions), (args.out_noisy, noisy)]:
        with open(path, "w", encoding="utf-8") as f:
            f.write("X Y Z\n")
            for x, y, z in arr:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"Rows: {len(positions)}")
    print(f"Clean saved to: {Path(args.out_clean).resolve()}")
    print(f"Noisy saved to: {Path(args.out_noisy).resolve()}")
    print(f"First clean UE: {positions[0]}")
    print(f"First noisy UE: {noisy[0]}")


if __name__ == "__main__":
    main()
