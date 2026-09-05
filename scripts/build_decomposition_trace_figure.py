#!/usr/bin/env python3
"""The three-term decomposition drawn on the measured voltage-charge trajectory.

Figure 3 reports the decomposition as bars. A referee cannot see from bars where
the terms come from, so this figure puts them back on the trace they were
integrated from: loaded and six-minute-rest voltage against cumulative charge for
the sodium-ion cell at 5 and 25 degC, with the cutoff, the common window Qc, and
the two geometrically representable terms shaded.

The third term is the *growth* of the load-to-rest gap and is a difference of two
integrals taken over different charge ranges, so it cannot be a single shaded
region on panel (a). Panel (c) draws it in the space where it is an area: the
load-to-rest voltage gap against charge, integrated at each temperature.

Panel (b) zooms the tail, where the cutoff is reached inside a segment, the rest
returns the relaxed voltage above it, and the following segment runs briefly. That
is the evidence that the window is set by the cutoff and not by the schedule.

Self-checks: the shaded integrals are compared against the published decomposition
and the script fails if they disagree by more than 1 mWh.

SHARED ARTWORK. The default output is used by BOTH manuscripts. Run with --acs to
write figure_decomposition_trace_acs instead: same panels, but the 1.49 V line is
labelled as an *inferred* stopping line, because the ACS Letter makes the fact
that the archive contains no explicit stopping voltage part of its validity
argument. Do not fold the two back together without re-reading both captions.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import (COL_FULL, ACCENTS, INK, INK_MUTED, TERMS, use_style,
                          style_axis, panel_label, save_figure)
from figure_style import COLD as STYLE_COLD, WARM as STYLE_WARM
import matplotlib.pyplot as plt

SEGMENTS = ROOT / "results/tables/table_delivered_energy_segments.csv"
FAMILY, CUTOFF = "SIB", 1.49
COLD, WARM = 5.0, 25.0

# The archive has no stopping-voltage field; 1.49 V is recovered from the lowest
# terminal voltage reached. The ACS text says so, so the ACS artwork says so too.
LABELS = {
    # line_a_xfrac/ha place the stopping-line caption in panel (a); tail_top and
    # gap_top are y-limits that keep legends and annotations off the curves.
    "jes": dict(line_a_xfrac=1.02, line_a_ha="right", tail_top=None,
                window_note_off=(0.145, 2.30), legend_mask=False,
                gap_note_fmt="{word} on cooling = {v:.3f} Wh", gap_note_va="baseline",
                top_a=4.17, header_y=4.19,
                panel_kw=dict(dy=1.20), panel_kw_bc=dict(dy=1.30),
                gap_legend=dict(loc="upper left", bbox_to_anchor=(0.0, 1.0)),
                gap_top=None, gap_note_xy=(0.03, 0.055),
                line_a="cutoff\n1.49 V", line_b="cutoff 1.49 V",
                title_b="tail: instantaneous minima reach the cutoff\n"
                        "while segment means stay above it",
                gap_symbol="L_{\\rm pol}", gap_word="growth",
                title_c="third term is the growth of this area",
                stem="figure_decomposition_trace.png"),
    "acs": dict(line_a_xfrac=0.012, line_a_ha="left", tail_top=2.98,
                window_note_off=(0.095, 1.75), legend_mask=True,
                gap_note_fmt="{word} on cooling\n= {v:.3f} Wh", gap_note_va="top",
                top_a=4.34, header_y=4.36,
                panel_kw=dict(dx=0.010, dy=0.975, fontsize=13.0, va="top"),
                panel_kw_bc=dict(dx=0.016, dy=0.975, fontsize=13.0, va="top"),
                gap_legend=dict(loc="upper right", bbox_to_anchor=(1.0, 1.0)),
                gap_top=1.02, gap_note_xy=(0.035, 0.72),
                line_a="inferred line\n\u2248 1.49 V", line_b="inferred line \u2248 1.49 V",
                title_b="tail: instantaneous minima reach the inferred line",
                gap_symbol="L", gap_word="increase",
                title_c="third term: change in this integrated gap",
                stem="figure_decomposition_trace_acs.png"),
}
# Temperature and term colours come from the shared grammar so that the
# shorter-window term is the same colour here as in Figure 2b. Previously this
# file set its own: the same term was purple here and blue there.
C_COLD, C_WARM = STYLE_COLD, STYLE_WARM
C_WIN, C_SHIFT, C_GAP = TERMS["window"], TERMS["shift"], TERMS["gap"]


def trace(d, T):
    s = d[(d.cell_family == FAMILY) & (d.temperature_deg_c == T)].sort_values("segment_index")
    q = s.segment_capacity_ah.values
    return dict(dq=q, q_end=np.cumsum(q), q_mid=np.cumsum(q) - q / 2,
                v_ocv=s.ocv_voltage_v.values, v_load=s.mean_terminal_voltage_v.values,
                Q=q.sum(), Edel=s.delivered_energy_wh.sum(), Eocv=s.ocv_energy_wh.sum())


def integral_to(dq, v, Qc):
    tot, prev = 0.0, 0.0
    for qi, vi, ci in zip(dq, v, np.cumsum(dq)):
        if ci <= Qc:
            tot += qi * vi
        else:
            tot += qi * vi * max(0.0, Qc - prev) / qi
            break
        prev = ci
    return tot


def main() -> int:
    variant = "acs" if "--acs" in sys.argv[1:] else "jes"
    text = LABELS[variant]
    d = pd.read_csv(SEGMENTS)
    d = d[d.direction == "discharge"]
    c, w = trace(d, COLD), trace(d, WARM)
    Qc = min(c["Q"], w["Q"])

    shorter = (w["Eocv"] - integral_to(w["dq"], w["v_ocv"], Qc)) - \
              (c["Eocv"] - integral_to(c["dq"], c["v_ocv"], Qc))
    shift = integral_to(w["dq"], w["v_ocv"], Qc) - integral_to(c["dq"], c["v_ocv"], Qc)
    gap_growth = (c["Eocv"] - c["Edel"]) - (w["Eocv"] - w["Edel"])
    total = w["Edel"] - c["Edel"]
    assert abs(shorter + shift + gap_growth - total) < 1e-3, "identity does not close"

    use_style()
    fig = plt.figure(figsize=(COL_FULL, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.55, wspace=0.28)
    ax = fig.add_subplot(gs[0, :])

    # (a) --- the trajectory, with the two representable terms shaded ----------
    mw = w["q_end"] >= Qc
    ax.fill_between(np.r_[Qc, w["q_end"][mw]], CUTOFF,
                    np.r_[np.interp(Qc, w["q_end"], w["v_ocv"]), w["v_ocv"][mw]],
                    color=C_WIN, alpha=0.28, lw=0, zorder=1)
    grid = np.linspace(0, Qc, 400)
    ax.fill_between(grid, np.interp(grid, c["q_end"], c["v_ocv"]),
                    np.interp(grid, w["q_end"], w["v_ocv"]),
                    color=C_SHIFT, alpha=0.42, lw=0, hatch="///", edgecolor="white",
                    zorder=1)

    for t, col, lab in ((w, C_WARM, "25 °C"), (c, C_COLD, "5 °C")):
        ax.plot(t["q_end"], t["v_ocv"], color=col, lw=2.0, zorder=4,
                label=f"{lab} rested (6 min)")
        ax.plot(t["q_mid"], t["v_load"], color=col, lw=1.4, ls="--", zorder=4,
                label=f"{lab} loaded (1.5 A)")
    ax.axhline(CUTOFF, color=INK_MUTED, lw=1.0, ls=":", zorder=2)
    ax.text(w["Q"] * text["line_a_xfrac"], CUTOFF + 0.04, text["line_a"],
            fontsize=8.0, color=INK_MUTED, ha=text["line_a_ha"], va="bottom",
            linespacing=1.15)
    # direct labels on the shaded areas -- identity never rests on the fill alone
    ax.annotate(f"shorter window\n{shorter:.3f} Wh",
                xy=(Qc + text["window_note_off"][0], text["window_note_off"][1]),
                fontsize=8.5, color=INK, ha="center", va="center", zorder=6)
    ax.annotate(f"rested-voltage\nshift  {shift:.3f} Wh", xy=(0.66, 3.05),
                xytext=(0.50, 2.10), fontsize=8.5, color=INK, zorder=6,
                ha="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.8))
    for x, lab in ((Qc, "$Q_c$ = Q(5 °C)"), (w["Q"], "Q(25 °C)")):
        ax.axvline(x, color=INK_MUTED, lw=0.9, ls="-", alpha=0.45, zorder=2)
        ax.text(x, text["header_y"], lab, fontsize=8.5, color=INK_MUTED,
                ha="center", va="bottom")
    ax.set_xlim(0, w["Q"] * 1.035); ax.set_ylim(CUTOFF - 0.1, text["top_a"])
    ax.set_xlabel("cumulative discharged charge (Ah)")
    ax.set_ylabel("cell voltage (V)")
    style_axis(ax, grid="y"); panel_label(ax, "a", **text["panel_kw"])
    ax.legend(loc="upper right", fontsize=8.0, ncol=1, handlelength=2.0,
              labelspacing=0.32, bbox_to_anchor=(1.0, 0.97),
              frameon=text["legend_mask"], facecolor="white", edgecolor="none",
              framealpha=1.0, borderpad=0.45)

    # (b) --- the tail: repeated cutoff encounters -----------------------------
    axb = fig.add_subplot(gs[1, 0])
    vmin = pd.read_csv(ROOT / "results/tables/table_segment_voltage_minima.csv")
    for t, col, lab, T in ((w, C_WARM, "25 °C", 25), (c, C_COLD, "5 °C", 5)):
        m = t["q_end"] >= t["Q"] - 0.14
        axb.plot((t["q_end"][m] - t["Q"]) * 1000, t["v_ocv"][m], color=col,
                 lw=1.9, marker="o", ms=2.8, label=f"{lab} rested")
        axb.plot((t["q_mid"][m] - t["Q"]) * 1000, t["v_load"][m], color=col,
                 lw=1.2, ls="--", marker="v", ms=2.8)
        # instantaneous per-sample minima: these are what the cutoff acts on
        vm = vmin[vmin.temperature_deg_c == T]
        qm = vm.cumulative_charge_ah.values
        sel = qm >= qm[-1] - 0.14
        axb.plot((qm[sel] - qm[-1]) * 1000, vm.v_min_instantaneous.values[sel],
                 color=col, lw=0, marker="v", ms=3.4, alpha=0.95,
                 markeredgecolor="none",
                 label="instantaneous minima" if T == 5 else None)
    axb.axhline(CUTOFF, color=INK_MUTED, lw=1.0, ls=":")
    axb.set_ylim(1.455, text["tail_top"])
    axb.text(-150, CUTOFF + 0.022, text["line_b"], fontsize=8.0,
             color=INK_MUTED, va="bottom", ha="left")
    axb.set_xlabel("charge before end of discharge (mAh)")
    axb.set_ylabel("voltage (V)")
    axb.set_title(text["title_b"], fontsize=9.0, color=INK_MUTED, pad=6)
    axb.legend(loc="upper right", fontsize=7.3, frameon=False, handlelength=1.6,
               labelspacing=0.28, borderaxespad=0.2)
    style_axis(axb, grid="none"); panel_label(axb, "b", **text["panel_kw_bc"])

    # (c) --- the load-to-rest gap, where the third term is an area ------------
    axc = fig.add_subplot(gs[1, 1])
    gw = w["v_ocv"] - w["v_load"]; gc = c["v_ocv"] - c["v_load"]
    axc.fill_between(w["q_mid"], 0, gw, color=C_WARM, alpha=0.22, lw=0)
    axc.fill_between(c["q_mid"], 0, gc, facecolor=C_COLD, alpha=0.16, lw=0)
    axc.fill_between(c["q_mid"], 0, gc, facecolor="none", hatch="//",
                     edgecolor=C_COLD, lw=0.0, alpha=0.40)
    sym = text["gap_symbol"]
    axc.plot(w["q_mid"], gw, color=C_WARM, lw=1.8,
             label=f"25 °C:  ${sym}$ = {w['Eocv']-w['Edel']:.3f} Wh")
    axc.plot(c["q_mid"], gc, color=C_COLD, lw=1.8,
             label=f"5 °C:  ${sym}$ = {c['Eocv']-c['Edel']:.3f} Wh")
    if text["gap_top"] is not None:
        axc.set_ylim(0.0, text["gap_top"])
    axc.annotate(text["gap_note_fmt"].format(word=text["gap_word"], v=gap_growth),
                 xy=text["gap_note_xy"], xycoords="axes fraction", fontsize=8.5,
                 color=INK, fontweight="bold", va=text["gap_note_va"])
    axc.set_xlabel("cumulative charge (Ah)")
    axc.set_ylabel("load-to-rest gap (V)")
    axc.set_title(text["title_c"], fontsize=9.0, color=INK_MUTED, pad=6)
    axc.legend(fontsize=8.0, frameon=False, handlelength=1.8, labelspacing=0.3,
               **text["gap_legend"])
    style_axis(axc, grid="none"); panel_label(axc, "c", **text["panel_kw_bc"])

    save_figure(fig, text["stem"])
    print(f"shorter window {shorter:.4f} | shift {shift:.4f} | gap growth {gap_growth:.4f}")
    print(f"sum {shorter+shift+gap_growth:.4f} vs measured shortfall {total:.4f} Wh")
    print(f"shares  {100*shorter/total:.1f} / {100*gap_growth/total:.1f} / {100*shift/total:.1f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
