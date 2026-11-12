#!/usr/bin/env python3
"""Render Figure 3: signed paired DeLong contrasts with adjusted p values.

Panel A compares each FM with Radiomics for the three tasks. Panel B compares
the five FMs and Radiomics with the task-specific simple imaging baseline.
Colour encodes signed delta AUC (comparator minus baseline), centred at zero.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable

MODELS = ["Merlin", "RADAR", "TotalFM", "Jolia", "OrganLens", "Radiomics"]
FM_MODELS = MODELS[:-1]
TASKS = ["Steatosis", "NAS ≥ 3", "Sinusoidal\ndilatation"]
BASELINES = {
    "Steatosis": "Mean liver parenchyma HU",
    "Sinusoidal dilatation": "Spleen volume",
}


def fmt_p(x):
    if pd.isna(x):
        return "adj.p=NA"
    return "adj.p<0.001" if x < 0.001 else f"adj.p={x:.3f}"


def text_color(rgba):
    r, g, b, _ = rgba
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.58 else "#1F2933"


def draw_cells(ax, values, pvalues, row_labels, col_labels, norm, cmap,
               delta_label, panel_letter, title=None, fontsize=8.6):
    arr = np.asarray(values, dtype=float)
    ax.imshow(arr, cmap=cmap, norm=norm, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=38,
                  ha="right", rotation_mode="anchor", fontsize=8.0)
    ax.set_yticks(np.arange(len(row_labels)), row_labels, fontsize=8.4)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            rgba = cmap(norm(arr[i, j]))
            color = text_color(rgba)
            sign = "+" if arr[i, j] >= 0 else ""
            ax.text(j, i - 0.08, f"{sign}{arr[i, j]:.3f}", ha="center",
                    va="center", color=color, fontsize=fontsize, fontweight="bold")
            ax.text(j, i + 0.22, fmt_p(pvalues[i][j]), ha="center",
                    va="center", color=color, fontsize=6.7)
    ax.text(-0.18, 1.06, panel_letter, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom", ha="left", color="#111827")
    if title:
        ax.set_title(title, loc="left", fontsize=9.3, fontweight="bold", pad=8)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm-rad", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fm = pd.read_csv(args.fm_rad)
    if "Task" in fm.columns:
        fm = fm[fm["Family"].eq("FM_vs_Radiomics")].copy()
        fm["plot_task"] = fm["Task"].replace({"NAS >= 3": "NAS ≥ 3"})
        fm["signed_delta_auc"] = fm["Signed_delta_AUC"]
        fm["holm_adjusted_p"] = fm["Holm_p_within_5_FM_vs_Radiomics"]
    delta_col = "signed_delta_auc"
    adj_col = "holm_adjusted_p"
    a_values, a_p = [], []
    for task in TASKS:
        task_key = task.replace("\n", " ")
        # Accept both Unicode ≥ and ASCII >= task labels in fresh CSVs.
        task_norm = fm["plot_task"].astype(str).str.replace("≥", ">=", regex=False)
        key_norm = task_key.replace("≥", ">=")
        d = fm[task_norm == key_norm].set_index("Comparator").reindex(FM_MODELS)
        a_values.append(d[delta_col].to_numpy(float))
        a_p.append(d[adj_col].to_numpy(float))

    base = pd.read_csv(args.baseline)
    base = base[base["Baseline"].isin(BASELINES.values())].copy()
    b_values, b_p, b_rows = [], [], []
    for task, baseline_name in BASELINES.items():
        d = base[(base["Task"] == task) & (base["Baseline"] == baseline_name)]
        d = d.set_index("Comparator").reindex(MODELS)
        b_values.append(d["Signed_delta_AUC_model_minus_baseline"].to_numpy(float))
        b_p.append(d["Holm_p_within_baseline_task_six_models"].to_numpy(float))
        short_baseline = {
            "Mean liver parenchyma HU": "mean liver HU",
            "Spleen volume": "spleen volume",
        }[baseline_name]
        task_label = task.replace("Sinusoidal dilatation", "Sinusoidal\ndilatation")
        b_rows.append(f"{task_label}\n({short_baseline})")

    all_delta = np.concatenate([np.asarray(a_values).ravel(), np.asarray(b_values).ravel()])
    vmax = max(0.18, float(np.nanmax(np.abs(all_delta))) * 1.05)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure(figsize=(180 / 25.4, 170 / 25.4), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.22, 0.78], hspace=0.40,
                          left=0.25, right=0.875, top=0.94, bottom=0.16)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    draw_cells(ax_a, a_values, a_p, TASKS, FM_MODELS, norm, cmap, delta_col,
               "A")
    draw_cells(ax_b, b_values, b_p, b_rows, MODELS, norm, cmap,
               "Signed ΔAUC", "B")
    # Keep panel A task spacing compact and align both panels to the same right edge.
    ax_a.set_xlim(-0.5, len(FM_MODELS) - 0.5)
    ax_b.set_xlim(-0.5, len(MODELS) - 0.5)

    cax = fig.add_axes([0.90, 0.27, 0.016, 0.52])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    fig.text(0.908, 0.815, "signed\nΔAUC", ha="center", va="bottom",
             fontsize=8.0, linespacing=0.95)
    cb.ax.tick_params(labelsize=7.0, length=3)
    cb.outline.set_visible(False)

    for ext, dpi in [("pdf", None), ("png", 400)]:
        fig.savefig(args.out_dir / f"Figure_3_pairwise_AUC_DeLong.{ext}",
                    dpi=dpi, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
