import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


MODES = ["ls_only", "single_rss", "multibs_rss", "zero_rss", "shuffled_rss"]


def run_one(args, mode, log_dir):
    out_dir = Path(args.data_dir) / "runs" / f"{mode}_ep{args.epochs}"
    log_path = log_dir / f"{mode}_ep{args.epochs}.log"
    cmd = [
        sys.executable,
        "deepmimo_multibs/train_rss_baseline.py",
        "--data-dir",
        args.data_dir,
        "--mode",
        mode,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--out-dir",
        str(out_dir),
    ]
    if args.device:
        cmd.extend(["--device", args.device])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.lr is not None:
        cmd.extend(["--lr", str(args.lr)])

    print(f"\n=== Running {mode} ===")
    print(" ".join(cmd))
    print(f"Log: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = proc.wait()

    if return_code != 0:
        raise RuntimeError(f"{mode} failed with exit code {return_code}. See {log_path}")

    result_path = out_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_path)
    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    result["log_file"] = str(log_path)
    result["out_dir"] = str(out_dir)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run all five DeepMIMO RSS baselines.")
    parser.add_argument(
        "--data-dir",
        default="deepmimo_multibs/processed/o1_60_rx0_tx10_11_12_channel",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    log_dir = Path(args.log_dir) if args.log_dir else data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for mode in args.modes:
        results.append(run_one(args, mode, log_dir))

    summary_json = log_dir / f"summary_ep{args.epochs}.json"
    summary_csv = log_dir / f"summary_ep{args.epochs}.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "best_val_nmse",
                "test_nmse",
                "test_db",
                "best_model",
                "log_file",
                "out_dir",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    print("\n=== All runs completed ===")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")
    for row in results:
        print(f"{row['mode']}: test_nmse={row['test_nmse']:.6f}, test_db={row['test_db']:.2f} dB")


if __name__ == "__main__":
    main()
