"""Render the benchmark comparison (fwd+bwd time + peak VRAM) as a grouped bar chart.
Numbers are the measured A40 fp32 k=7 results from bench.py with the fused backward
(CUDA-event median/40, paired inputs, TZ=8; run-to-run CV < 1% at 128³)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

shapes = ["(1,256,16³)", "(2,64,64³)", "(2,16,128³)"]
impls = ["fused_lncc (ours)", "FireANTs (ICLR'26)", "sep+compile", "MONAI", "naive (PyTorch)"]
time_ms = {  # fwd+bwd, ms
    "fused_lncc (ours)":  [0.39, 12.19, 24.48],
    "FireANTs (ICLR'26)": [1.56, 41.74, 85.67],
    "sep+compile":        [1.38, 40.57, 81.09],
    "MONAI":              [4.14, 86.14, 162.49],
    "naive (PyTorch)":    [6.66, 215.44, 432.46],
}
mem_gb = {  # peak VRAM, GB
    "fused_lncc (ours)":  [0.02, 0.54, 1.07],
    "FireANTs (ICLR'26)": [0.04, 1.61, 3.22],
    "sep+compile":        [0.06, 2.01, 4.03],
    "MONAI":              [0.15, 3.74, 7.21],
    "naive (PyTorch)":    [0.06, 1.75, 3.49],
}
colors = ["#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#7f7f7f"]
x = np.arange(len(shapes)); w = 0.16
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for i, im in enumerate(impls):
    ax[0].bar(x + (i - 2) * w, time_ms[im], w, label=im, color=colors[i])
    ax[1].bar(x + (i - 2) * w, mem_gb[im], w, label=im, color=colors[i])
ax[0].set_yscale("log"); ax[0].set_ylabel("fwd+bwd time (ms, log)"); ax[0].set_title("Speed (lower is better)")
ax[1].set_ylabel("peak VRAM (GB)"); ax[1].set_title("Memory (lower is better)")
for a in ax:
    a.set_xticks(x); a.set_xticklabels(shapes); a.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Fused LNCC vs baselines — A40, fp32, k=7  (ours 3.5× faster / 3.0× less VRAM than ICLR'26 SOTA)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.png"), dpi=130, bbox_inches="tight")
print("wrote assets/benchmark.png")
