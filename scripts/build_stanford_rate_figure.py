#!/usr/bin/env python3
"""Stanford P42A: rate dependence of the cold delivered-charge and delivered-energy
shortfall, paired within cells.

Panels (a) and (b) show every cell's 5/25 degC retention at C/3, C/10 and C/20,
paired. Panel (c) shows the quantity that actually tests a cold-specific rate
effect: the within-cell slope of log(delivered quantity) against C-rate, at each
temperature. A cold-specific effect requires the 5 degC slope to be steeper than
the 25 degC slope; that difference is the temperature x rate interaction.

Deliberately NOT plotted: cold penalty against rate benefit. Those two axes share
the same C/3 measurement with the same sign -- with y = R(C/20) - R(C/3) and
x = 1 - R(C/3) we have y = x - (1 - R(C/20)) -- so a positive correlation arises
partly from algebra. A within-cell permutation null on these data returns median
r ~ 0.16-0.24 with a 95 percent interval reaching 0.85, against an observed 0.88,
so that correlation is not clean evidence and is not used.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import COL_FULL, INK, INK_MUTED, use_style, style_axis, panel_label, save_figure
import matplotlib.pyplot as plt

TABLE = ROOT / "results/tables/table_stanford_rate_dependence.csv"
RATES = ["C/3", "C/10", "C/20"]
CUR = {"C/3": 1/3, "C/10": 1/10, "C/20": 1/20}
C_COLD, C_WARM, C_CELL = "#2a78d6", "#eb6834", "#9a9894"


def main() -> int:
    df = pd.read_csv(TABLE)
    ok = df.groupby("cell").filter(lambda g: set(g.temperature_deg_c) == {5, 25} and len(g) == 6)
    piv = ok.pivot_table(index=["cell", "rate"], columns="temperature_deg_c", values=["Q_ah", "E_wh"])
    rows = []
    for (cell, rate), r in piv.iterrows():
        rows.append(dict(cell=cell, rate=rate,
                         Q_ret=r[("Q_ah", 5)] / r[("Q_ah", 25)],
                         E_ret=r[("E_wh", 5)] / r[("E_wh", 25)]))
    p = pd.DataFrame(rows)
    cells = sorted(p.cell.unique())

    use_style()
    fig, axes = plt.subplots(1, 3, figsize=(COL_FULL, 3.0))
    x = np.arange(3)

    for ax, key, lab, letter in ((axes[0], "Q_ret", "charge", "a"),
                                 (axes[1], "E_ret", "energy", "b")):
        w = p.pivot(index="cell", columns="rate", values=key)[RATES]
        for c in cells:
            ax.plot(x, 100 * w.loc[c].values, color=C_CELL, lw=0.9,
                    marker="o", ms=3.0, alpha=0.75, zorder=2)
        m = 100 * w.mean().values
        se = 100 * w.std(ddof=1).values / np.sqrt(len(w))
        ax.errorbar(x, m, yerr=se, color=C_COLD, lw=2.2, marker="s", ms=6.5,
                    capsize=3.5, zorder=4, label="mean $\\pm$ s.e.")
        ax.set_xticks(x); ax.set_xticklabels(RATES)
        ax.set_xlabel("discharge rate")
        ax.set_ylabel(f"5/25 °C {lab} retention (%)")
        ax.set_xlim(-0.35, 2.35)
        style_axis(ax, grid="y"); panel_label(ax, letter, dy=1.16)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False)

    # (c) the interaction: rate sensitivity at each temperature
    ax = axes[2]
    width = 0.34
    for i, (var, lab) in enumerate((("Q_ah", "charge"), ("E_wh", "energy"))):
        for j, (T, col) in enumerate(((25, C_WARM), (5, C_COLD))):
            sl = []
            for c in cells:
                g = ok[(ok.cell == c) & (ok.temperature_deg_c == T)]
                sl.append(np.polyfit(g.rate.map(CUR).values, np.log(g[var].values), 1)[0])
            sl = np.array(sl)
            ax.bar(i + (j - 0.5) * width, -sl.mean(), width * 0.9,
                   yerr=sl.std(ddof=1) / np.sqrt(len(sl)), color=col, alpha=0.9,
                   capsize=3.5, label=f"{T} °C" if i == 0 else None,
                   edgecolor="white", lw=0.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["charge", "energy"])
    ax.set_ylabel("rate sensitivity  $-\\mathrm{d}\\ln y/\\mathrm{d}C$")
    ax.set_xlabel("delivered quantity")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False)
    ax.set_ylim(0, 0.105)
    ax.annotate("interaction\np = 0.012", xy=(1.0, 0.086), fontsize=7.5,
                color=INK, ha="center", va="bottom")
    ax.annotate("n.s.\np = 0.09", xy=(0.0, 0.050), fontsize=7.5,
                color=INK_MUTED, ha="center", va="bottom")
    style_axis(ax, grid="y"); panel_label(ax, "c", dy=1.16)

    fig.subplots_adjust(wspace=0.58)
    save_figure(fig, "figure_stanford_rate_dependence.png",
                folders=["results/figures", "manuscript/figures"])
    print("figure written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
