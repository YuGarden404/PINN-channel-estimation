"""Run 500-epoch ablation experiments.

Writes one detailed log file per run under log/ and keeps only the best-val
state_dict under a dedicated checkpoint directory.
"""

import argparse
import contextlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from train import main_train


PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "log"

RUNS = {
    "normal_alpha001": {
        "rss_mode": "normal",
        "alpha": 0.01,
        "rss_image_path": "Dataset/50_15GHz.jpg",
        "description": "normal RSS + noisy position + alpha=0.01",
    },
    "normal_alpha0": {
        "rss_mode": "normal",
        "alpha": 0.0,
        "rss_image_path": "Dataset/50_15GHz.jpg",
        "description": "normal RSS + noisy position + alpha=0",
    },
    "zero_rss_alpha001": {
        "rss_mode": "zero",
        "alpha": 0.01,
        "rss_image_path": "Dataset/50_15GHz.jpg",
        "description": "zero RSS + noisy position + alpha=0.01",
    },
    "constant_rss_alpha001": {
        "rss_mode": "constant",
        "alpha": 0.01,
        "rss_image_path": "Dataset/50_15GHz.jpg",
        "description": "constant RSS + noisy position + alpha=0.01",
    },
}


class Tee:
    def __init__(self, console_stream, log_stream):
        self.console_stream = console_stream
        self.log_stream = log_stream

    def _is_progress_line(self, data):
        return (
            '[Train]:' in data
            or '[Val]:' in data
            or 'it/s' in data
            or '\r' in data
        )

    def write(self, data):
        self.console_stream.write(data)
        self.console_stream.flush()

        if not self._is_progress_line(data):
            self.log_stream.write(data)
            self.log_stream.flush()

    def flush(self):
        self.console_stream.flush()
        self.log_stream.flush()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_ckpt_dir(args):
    if args.checkpoint_dir:
        return PROJECT_ROOT / args.checkpoint_dir
    return PROJECT_ROOT / f"checkpoints_ablation_{args.sample_tag}_ep{args.epochs}"


def build_config(run_name: str, run_cfg: dict, args, ckpt_dir: Path):
    return {
        "smomp_file": args.smomp_file,
        "accurate_file": args.accurate_file,
        "user_positions_file": args.user_positions_file,
        "rss_image_path": run_cfg["rss_image_path"],
        "bs_pixel_coords": (287, 293),
        "bs_real_coords": (71.06, 246.29),
        "image_width_meters": 527.5,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "rss_mode": run_cfg["rss_mode"],
        "alpha": run_cfg["alpha"],
        "name_val": str(ckpt_dir / f"simple_ls_0_val_{args.sample_tag}_ep{args.epochs}_{run_name}.pth"),
        "name_train": str(ckpt_dir / f"simple_ls_0_train_{args.sample_tag}_ep{args.epochs}_{run_name}.pth"),
        "save_train_checkpoint": False,
        "save_frequency": 0,
        "evaluate_train_checkpoint": False,
        "save_half_precision": args.save_half_precision,
    }


def run_one(run_name: str, args):
    run_cfg = RUNS[run_name]
    ckpt_dir = get_ckpt_dir(args)
    config = build_config(run_name, run_cfg, args, ckpt_dir)

    LOG_DIR.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)

    log_path = LOG_DIR / f"{args.log_prefix}_{run_name}.txt"
    manifest_path = ckpt_dir / f"manifest_{args.sample_tag}_ep{args.epochs}_{run_name}.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "description": run_cfg["description"],
                "seed": args.seed,
                "sample_tag": args.sample_tag,
                "config": config,
            },
            f,
            indent=2,
        )

    print(f"\n=== Running {run_name}: {run_cfg['description']} ===")
    print(f"Log: {log_path}")
    print(f"Val checkpoint: {config['name_val']}")
    print("Train checkpoint: disabled")

    if args.skip_existing and Path(config["name_val"]).exists():
        print(f"Skipping {run_name}; val checkpoint already exists.")
        return

    set_seed(args.seed)
    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(os.sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"Run name: {run_name}")
            print(f"Description: {run_cfg['description']}")
            print(f"Seed: {args.seed}")
            print(f"Sample tag: {args.sample_tag}")
            print("Config:")
            print(json.dumps(config, indent=2, default=str))
            main_train(config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation experiments.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--sample-tag", default="1000")
    parser.add_argument("--log-prefix", default="log_500")
    parser.add_argument("--checkpoint-dir", default=None)

    parser.add_argument("--smomp-file", default="initial_estimate_ls_snr0_1000.npy")
    parser.add_argument("--accurate-file", default="3D_channel_15GHz_2x2_Pt50_1000.npy")
    parser.add_argument("--user-positions-file", default="ue_positions_noisy_1000.txt")

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-half-precision", action="store_true")

    parser.add_argument(
        "--runs",
        nargs="+",
        choices=list(RUNS.keys()),
        default=list(RUNS.keys()),
        help="Subset of runs to execute in order.",
    )
    return parser.parse_args()


def main():
    os.chdir(PROJECT_ROOT)
    args = parse_args()

    for required in [args.smomp_file, args.accurate_file, args.user_positions_file]:
        if not Path(required).exists():
            raise FileNotFoundError(required)

    for run_name in args.runs:
        run_one(run_name, args)


if __name__ == "__main__":
    main()
