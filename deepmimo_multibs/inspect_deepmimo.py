import argparse
import json

import numpy as np


def summarize_value(value):
    summary = {"type": str(type(value))}
    if isinstance(value, list):
        summary["length"] = len(value)
        item_summaries = []
        for idx, item in enumerate(value[:10]):
            item_summary = {"index": idx, "type": str(type(item))}
            try:
                arr = np.asarray(item)
                item_summary.update({"shape": list(arr.shape), "dtype": str(arr.dtype)})
            except Exception as exc:
                item_summary["array_error"] = str(exc)
            item_summaries.append(item_summary)
        summary["first_items"] = item_summaries
    try:
        arr = np.asarray(value)
        summary.update({"shape": list(arr.shape), "dtype": str(arr.dtype)})
        if arr.size and arr.size <= 10:
            summary["preview"] = arr.tolist()
    except Exception as exc:
        summary["array_error"] = str(exc)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inspect a DeepMIMO scenario/dataset object.")
    parser.add_argument("--scenario", required=True, help="DeepMIMO scenario name, e.g. o1_60.")
    parser.add_argument("--download", action="store_true", help="Download the scenario before loading.")
    parser.add_argument("--out", default=None, help="Optional JSON path to save the inspection summary.")
    args = parser.parse_args()

    try:
        import deepmimo as dm
    except ImportError as exc:
        raise SystemExit("DeepMIMO is not installed. Run: pip install DeepMIMO") from exc

    if args.download and hasattr(dm, "download"):
        dm.download(args.scenario)

    dataset = dm.load(args.scenario)
    print(f"Loaded scenario: {args.scenario}")
    print(f"Dataset type: {type(dataset)}")

    public_names = [name for name in dir(dataset) if not name.startswith("_")]
    key_names = [
        "power",
        "phase",
        "delay",
        "aoa_az",
        "aoa_el",
        "aod_az",
        "aod_el",
        "pathloss",
        "rx_pos",
        "tx_pos",
        "channel",
        "num_paths",
        "los",
        "txrx_sets",
        "scene",
    ]
    summary = {
        "scenario": args.scenario,
        "dataset_type": str(type(dataset)),
        "public_names": public_names,
        "fields": {},
    }

    for name in key_names:
        if hasattr(dataset, name):
            summary["fields"][name] = summarize_value(getattr(dataset, name))

    print(json.dumps(summary["fields"], indent=2, default=str))

    if hasattr(dataset, "info"):
        print(dataset.info())
    else:
        print("No dataset.info() method found. Inspect dir(dataset) below:")
        print([name for name in dir(dataset) if not name.startswith("_")][:80])

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Saved inspection summary to {args.out}")


if __name__ == "__main__":
    main()
