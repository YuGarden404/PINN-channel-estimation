from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("deepmimo_multibs/paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name, rect=None):
    path = OUT_DIR / name
    if rect is None:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=rect)
    plt.savefig(path, dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    print(path)


def plot_main_nmse_vs_snr():
    snr = np.array([0, -5, -10, -15, -20])
    order = np.argsort(snr)
    snr = snr[order]

    data = {
        "MLP LS-only": np.array([0.418038, 0.754223, 1.070395, 1.377312, 2.014708])[order],
        "Cross-attn LS-only": np.array([0.364926, 0.645223, 0.968148, 1.014650, 1.057143])[order],
        "Noise-aware": np.array([0.378097, 0.677717, 0.981476, 1.005800, 1.050226])[order],
        "NC-CENet": np.array([0.359841, 0.644630, 0.871116, 1.010255, 1.145429])[order],
    }

    plt.figure(figsize=(7.2, 4.6))
    for label, values in data.items():
        plt.plot(snr, values, marker="o", linewidth=2.2, markersize=6, label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test NMSE")
    plt.title("Main Comparison Across SNR")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.gca().invert_xaxis()
    savefig("fig1_main_nmse_vs_snr.png")


def plot_main_db_vs_snr():
    snr = np.array([0, -5, -10, -15, -20])
    order = np.argsort(snr)
    snr = snr[order]

    data = {
        "Cross-attn LS-only": 10
        * np.log10(np.array([0.364926, 0.645223, 0.968148, 1.014650, 1.057143])[order]),
        "NC-CENet": 10
        * np.log10(np.array([0.359841, 0.644630, 0.871116, 1.010255, 1.145429])[order]),
    }

    plt.figure(figsize=(6.8, 4.4))
    for label, values in data.items():
        plt.plot(snr, values, marker="o", linewidth=2.2, markersize=6, label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Test NMSE (dB)")
    plt.title("NC-CENet vs. Cross-Attention Baseline")
    plt.grid(True, alpha=0.28)
    plt.legend(frameon=False)
    plt.gca().invert_xaxis()
    savefig("fig2_nmse_db_vs_snr.png")


def plot_noise_weight_ablation():
    weights = ["0", "0.001", "0.005", "0.01", "0.05"]
    data = {
        "-10 dB": [0.878649, 0.878225, 0.885283, 0.871116, 0.892857],
        "-15 dB": [1.047623, 0.977731, 0.991267, 1.010255, 0.992787],
        "-20 dB": [1.098779, 1.063262, 1.076384, 1.145429, 1.095953],
    }

    x = np.arange(len(weights))
    width = 0.24

    plt.figure(figsize=(8.2, 4.7))
    for idx, (label, nmse) in enumerate(data.items()):
        offset = (idx - 1) * width
        plt.bar(x + offset, nmse, width=width, label=label)
    plt.xticks(x, weights)
    plt.xlabel("Noise loss weight")
    plt.ylabel("Test NMSE")
    plt.title("Noise Weight Ablation Across Low SNR")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False, ncol=3)
    plt.ylim(0.84, 1.18)
    savefig("fig3_noise_weight_ablation.png")


def plot_rss_control():
    modes = ["LS-only", "Multi-BS RSS", "Single RSS", "Shuffled RSS"]
    nmse = [0.871116, 0.889460, 0.892777, 0.888009]
    colors = ["#f58518", "#9ecae9", "#9ecae9", "#9ecae9"]

    plt.figure(figsize=(7.2, 4.4))
    plt.bar(modes, nmse, color=colors)
    plt.ylabel("Test NMSE at -10 dB")
    plt.title("RSS Context Control in NC-CENet")
    plt.grid(axis="y", alpha=0.25)
    plt.ylim(0.86, 0.90)
    plt.xticks(rotation=12)
    savefig("fig4_rss_control.png")


def plot_best_epoch():
    snr_labels = ["0 dB", "-5 dB", "-10 dB", "-15 dB", "-20 dB"]
    best_epoch = [17, 32, 53, 30, 22]
    epochs_trained = [37, 52, 73, 50, 42]

    x = np.arange(len(snr_labels))
    width = 0.36

    plt.figure(figsize=(6.8, 4.4))
    plt.bar(x - width / 2, best_epoch, width, label="Best epoch", color="#54a24b")
    plt.bar(x + width / 2, epochs_trained, width, label="Epochs trained", color="#9ecae9")
    plt.xticks(x, snr_labels)
    plt.ylabel("Epoch")
    plt.title("Validation-Selected Checkpoint and Early Stopping")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    savefig("fig5_best_epoch_early_stop.png")


def plot_cross_scenario():
    scenarios = ["ASU campus", "San Diego", "New York"]
    cross_attn = [0.940095, 0.991768, 1.018754]
    noise_conditioned = [0.928213, 0.989168, 0.999144]

    x = np.arange(len(scenarios))
    width = 0.36

    plt.figure(figsize=(7.6, 4.5))
    plt.bar(x - width / 2, cross_attn, width, label="Cross-attn", color="#9ecae9")
    plt.bar(x + width / 2, noise_conditioned, width, label="Noise-conditioned", color="#f58518")
    plt.xticks(x, scenarios)
    plt.ylabel("Test NMSE at -10 dB")
    plt.title("Cross-Scenario Noise-Conditioned Refinement")
    plt.grid(axis="y", alpha=0.25)
    plt.ylim(0.90, 1.04)
    plt.legend(frameon=False)
    for idx, value in enumerate(cross_attn):
        plt.text(idx - width / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(noise_conditioned):
        plt.text(idx + width / 2, value + 0.004, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    savefig("fig6_cross_scenario_asu.png")


def plot_adapter_multiseed_fewshot():
    scenarios = ["San Diego", "New York"]
    base_mean = np.array([0.986318, 1.011900])
    base_std = np.array([0.005156, 0.006112])
    adapter_mean = np.array([0.983182, 0.994607])
    adapter_std = np.array([0.005239, 0.004225])

    fractions = ["5%", "10%", "20%", "100%"]
    samples = [200, 400, 800, 4000]
    fewshot_nmse = [1.003133, 0.999868, 1.001802, 0.999144]
    fewshot_base = 1.018754

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))

    x = np.arange(len(scenarios))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        base_mean,
        width,
        yerr=base_std,
        capsize=4,
        label="Cross-attn",
        color="#9ecae9",
    )
    axes[0].bar(
        x + width / 2,
        adapter_mean,
        width,
        yerr=adapter_std,
        capsize=4,
        label="Adapter",
        color="#f58518",
    )
    axes[0].set_xticks(x, scenarios)
    axes[0].set_ylabel("Test NMSE at -10 dB")
    axes[0].set_title("Adapter multi-seed stability")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.96, 1.03)
    axes[0].legend(frameon=False)
    for idx, value in enumerate(base_mean):
        axes[0].text(idx - width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    for idx, value in enumerate(adapter_mean):
        axes[0].text(idx + width / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    axes[1].plot(samples, fewshot_nmse, marker="o", linewidth=2.2, color="#f58518", label="Adapter")
    axes[1].axhline(fewshot_base, linestyle="--", linewidth=1.8, color="#4c78a8", label="Cross-attn baseline")
    axes[1].set_xscale("log")
    axes[1].set_xticks(samples, fractions)
    axes[1].set_xlabel("New York train fraction")
    axes[1].set_ylabel("Test NMSE at -10 dB")
    axes[1].set_title("Few-shot target-scene calibration")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_ylim(0.992, 1.022)
    axes[1].legend(frameon=False)
    for sample, label, value in zip(samples, fractions, fewshot_nmse):
        axes[1].text(sample, value - 0.0022, f"{label}\n{value:.3f}", ha="center", va="top", fontsize=8)

    fig.suptitle("Noise-Conditioned Adapter: Stability and Data Efficiency", y=0.98, fontsize=15)
    savefig("fig8_adapter_multiseed_fewshot.png", rect=[0, 0, 1, 0.92])

def plot_ofdm_pilot_spacing_ablation():
    pilot_spacing = np.array([4, 8, 16])
    raw_nmse = np.array([0.054987, 0.105624, 0.205703])
    cross_attn = np.array([0.012720, 0.023008, 0.039937])
    adapter = np.array([0.012007, 0.022244, 0.038160])
    relative_drop = (cross_attn - adapter) / cross_attn * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))

    axes[0].plot(pilot_spacing, raw_nmse, marker="o", linewidth=2.2, label="Raw OFDM-LS", color="#4c78a8")
    axes[0].plot(pilot_spacing, cross_attn, marker="o", linewidth=2.2, label="Cross-attn", color="#9ecae9")
    axes[0].plot(pilot_spacing, adapter, marker="o", linewidth=2.2, label="Adapter", color="#f58518")
    axes[0].set_xlabel("Pilot spacing")
    axes[0].set_ylabel("Test NMSE at -10 dB")
    axes[0].set_title("OFDM-LS Pilot Sparsity")
    axes[0].set_xticks(pilot_spacing)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    for x, y in zip(pilot_spacing, adapter):
        axes[0].text(x, y + 0.006, f"{y:.3f}", ha="center", fontsize=8)

    axes[1].bar([str(x) for x in pilot_spacing], relative_drop, color="#f58518")
    axes[1].set_xlabel("Pilot spacing")
    axes[1].set_ylabel("Adapter relative drop vs. base (%)")
    axes[1].set_title("Residual Calibration Gain")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0.0, 6.5)
    for idx, value in enumerate(relative_drop):
        axes[1].text(idx, value + 0.18, f"{value:.2f}%", ha="center", fontsize=9)

    fig.suptitle("Pilot-Limited OFDM-LS Ablation", y=0.98, fontsize=15)
    savefig("fig9_ofdm_pilot_spacing_ablation.png", rect=[0, 0, 1, 0.92])

def plot_efficiency_latency():
    models = ["MLP", "Cross-attn", "Noise-aware", "NC-CENet"]
    test_nmse = [1.070395, 0.968148, 0.981476, 0.871116]
    latency_us = [0.932516, 6.405111, 7.275176, 9.147615]
    params_m = [0.141584, 0.492612, 0.527046, 0.905926]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=False)

    axes[0].bar(x, test_nmse, color=colors)
    axes[0].set_xticks(x, models, rotation=18)
    axes[0].set_ylabel("Test NMSE at -10 dB")
    axes[0].set_title("Accuracy")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.0, 1.18)
    for idx, value in enumerate(test_nmse):
        axes[0].text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, latency_us, color=colors)
    axes[1].set_xticks(x, models, rotation=18)
    axes[1].set_ylabel("Latency per sample (us)")
    axes[1].set_title("Inference Latency")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_ylim(0.0, 10.5)
    for idx, (latency, params) in enumerate(zip(latency_us, params_m)):
        axes[1].text(
            idx,
            latency + 0.25,
            f"{latency:.2f} us\n{params:.2f}M",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Accuracy and Low-Latency Inference, batch=256", y=0.98, fontsize=15)
    savefig("fig7_accuracy_latency_tradeoff.png", rect=[0, 0, 1, 0.92])


def main():
    plot_main_nmse_vs_snr()
    plot_main_db_vs_snr()
    plot_noise_weight_ablation()
    plot_rss_control()
    plot_best_epoch()
    plot_cross_scenario()
    plot_adapter_multiseed_fewshot()
    plot_ofdm_pilot_spacing_ablation()
    plot_efficiency_latency()


if __name__ == "__main__":
    main()


