"""Blackwell (RTX PRO 6000, sm_120) benchmark bar chart — fwd+bwd time + peak VRAM.
Measured live on the GPU via bench.py (CUDA-event median/40, paired inputs, fp32, k=7;
two runs agreed to <1%). FireANTs was rebuilt from source for sm_120 for this head-to-head."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

shapes = ["(1,256,16³)", "(2,64,64³)", "(2,16,128³)"]
impls = ["fused_lncc (ours)", "FireANTs (ICLR'26)", "sep+compile", "MONAI", "naive (PyTorch)"]
time_ms = {  # fwd+bwd, ms
    "fused_lncc (ours)":  [0.13, 4.25, 8.45],
    "FireANTs (ICLR'26)": [0.47, 14.91, 29.12],
    "sep+compile":        [0.46, 14.06, 27.60],
    "MONAI":              [1.27, 32.59, 60.97],
    "naive (PyTorch)":    [2.11, 67.67, 135.68],
}
mem_gb = {  # peak VRAM, GB (shape-determined, matches A40)
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
fig.suptitle("Fused LNCC vs baselines — RTX PRO 6000 (Blackwell, sm_120), fp32, k=7  "
             "(ours 3.4× faster / 3.0× less VRAM than ICLR'26 SOTA)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_blackwell.png"), dpi=130, bbox_inches="tight")
print("wrote assets/benchmark_blackwell.png")
