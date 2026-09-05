#!/usr/bin/env python3
"""Graphical abstract: where the cold energy goes, on the measured trajectory.

Two targets. The default is the Elsevier/JES version. Run with --acs for the
ACS Energy Letters TOC graphic, which differs in two ways that matter:

  * size. ACS Energy Letters asks for 3 x 2 in (7.5 x 5 cm) with sans-serif
    text no smaller than 6 pt. The Elsevier pane is two and a half times as
    wide and cannot be scaled into that box without dropping below the type
    floor.
  * colour. The Elsevier version carries its own palette, in which purple is the
    shorter-window term. In Figures 1 and 2 purple is the rested-voltage shift
    and the shorter-window term is dark red. A reader meeting the TOC and then
    Figure 2 would see one colour used for two concepts -- the exact collision
    the shared grammar in figure_style.py was written to remove. The ACS variant
    therefore takes COLD, WARM and TERMS from that module rather than redefining
    them here.

Elsevier asks for a single pane, no panel letters, at least 1328 x 531 px. This
draws the same two quantities the paper rests on -- the measured voltage-charge
trajectory at 5 and 25 degC, and the three-term split of the difference between
them -- side by side, with the terms named on the figure rather than in a key.

--acs writes two compositions of the same content. graphical_abstract_acs.pdf
is the submitted one: it plots the cold-warm displacement itself, and is the
file the Letter includes after its abstract. graphical_abstract_acs_alt.pdf
draws the two rested trajectories instead and is kept as the alternative.

The numbers are recomputed here from the released segment table, and the script
fails if the three terms do not sum to the measured difference, so the abstract
cannot drift away from the paper.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from figure_style import (ACCENTS, INK, INK_MUTED, TERMS, style_axis,  # noqa: E402
                          use_style)
from figure_style import COLD as STYLE_COLD, WARM as STYLE_WARM  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

SEGMENTS = ROOT / "results/tables/table_delivered_energy_segments.csv"
FAMILY, CUTOFF = "SIB", 1.49
# Median matched-throughput displacement, from
# results/tables/table_common_window_ocv_comparison.csv.
MEDIAN_SHIFT = -131.1
COLD, WARM = 5.0, 25.0
C_COLD, C_WARM = "#2a78d6", "#eb6834"
C_WIN, C_SHIFT, C_GAP = "#4a3aa7", "#1baf7a", "#eda100"

# ACS Energy Letters asks for a 3 x 2 in TOC graphic (7.5 x 5 cm) placed in the
# manuscript after the abstract. Emit exactly that page: the shared style sets
# savefig.bbox = "tight", which crops back to the ink and would hand ACS a
# 3.00 x 1.69 in file, so both ACS writers turn it off for the save.
ACS_W, ACS_H = 3.0, 2.0
# The composition was laid out against a tight crop, which left no margin to
# balance: it hangs 0.19 in off the left edge and 0.06 in off the right. On a
# fixed page that reads as a rightward drift, so the whole block -- panel, bar
# and the centred headline -- moves left by ACS_DX to sit centred in the box.
ACS_DX = -0.022
ACS_LEFT, ACS_WIDTH = 0.105 + ACS_DX, 0.875


def acs_layout() -> dict:
    """Vertical anchors for the ACS TOC canvas, as figure fractions.

    Written in inches and divided by the canvas height, so the page size lives
    in one place. The design was tuned at 3.25 x 1.75 in; on the narrower,
    taller ACS page the panel keeps that aspect ratio -- its height scales with
    its width, 0.400 * 1.75 * (3.0 / 3.25) in -- so the trajectory is a uniform
    reduction rather than a vertical stretch. The height the taller page adds
    goes into the margins and into the gaps between the three blocks.
    """
    top = ACS_H - 0.12                     # first headline row, va="top"
    row2 = top - 0.091                     # the two retention numbers
    row3 = top - 0.266                     # what they are numbers of
    panel_h = 0.400 * 1.75 * (ACS_W / 3.25)
    panel_top = row3 - 0.086 - 0.15        # row-3 type height, then the gap
    panel_bottom = panel_top - panel_h
    lead = panel_bottom - 0.10 - 0.121     # clear the x label, then centre
    share = lead - 0.156                   # per-term percentages, va="bottom"
    strip_top = share - 0.0225
    strip_h = 0.05
    names = strip_top - strip_h - 0.105
    return {
        "row1": top / ACS_H,
        "row2": row2 / ACS_H,
        "row3": row3 / ACS_H,
        "panel": [ACS_LEFT, panel_bottom / ACS_H, ACS_WIDTH, panel_h / ACS_H],
        "lead": lead / ACS_H,
        "share": share / ACS_H,
        "strip": [ACS_LEFT, (strip_top - strip_h) / ACS_H, ACS_WIDTH,
                  strip_h / ACS_H],
        "names": names / ACS_H,
    }


def save_acs(fig, stem: str) -> None:
    """Write the TOC pair at the full canvas size, overriding the tight bbox."""
    with plt.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for folder in ("results/figures", "manuscript/figures"):
            directory = ROOT / folder
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(directory / f"{stem}.png", dpi=600)
            fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)


def trace(frame: pd.DataFrame, temperature: float) -> dict:
    run = frame[
        (frame.cell_family == FAMILY) & (frame.temperature_deg_c == temperature)
    ].sort_values("segment_index")
    dq = run.segment_capacity_ah.values
    return dict(
        dq=dq,
        q_end=np.cumsum(dq),
        q_mid=np.cumsum(dq) - dq / 2,
        v_ocv=run.ocv_voltage_v.values,
        v_load=run.mean_terminal_voltage_v.values,
        Q=dq.sum(),
        Edel=run.delivered_energy_wh.sum(),
        Eocv=run.ocv_energy_wh.sum(),
    )


def integral_to(dq: np.ndarray, v: np.ndarray, q_c: float) -> float:
    total, previous = 0.0, 0.0
    for step, voltage, cumulative in zip(dq, v, np.cumsum(dq)):
        if cumulative <= q_c:
            total += step * voltage
        else:
            total += step * voltage * max(0.0, q_c - previous) / step
            break
        previous = cumulative
    return total


def acs_toc(cold: dict, warm: dict, q_c: float, shorter: float, shift: float,
            gap: float, total: float) -> None:
    """ACS TOC, variation A: the two rested trajectories. Not the submitted file.

    Kept as graphical_abstract_acs_alt.pdf. The submitted composition is
    acs_toc_delta below, which plots the displacement rather than the two
    curves; that docstring gives the reason.

    Composed around a single vertical spine. The approximately-equal sign
    between the two retention numbers and the -131 mV displacement sit on the
    same figure centreline, so the composition itself carries the argument: the
    scalar summaries agree, and directly beneath them the trajectory does not.
    No divider rule, and no word such as conceal -- alignment does that work.

    Two constraints the layout has to respect. The displacement is 131 mV in a
    2.7 V span, so at this size the widest in-window gap is under 4 pt and no
    bracket can be drawn across it; the filled violet band carries it instead,
    in the same hue as the 15 % segment of the footer strip. For the same
    reason the curves cannot be direct-labelled where they are closest, so the
    labels sit above and below them at about two thirds of the window, where
    grey is unambiguously the upper trace.
    """
    use_style()
    plt.rcParams.update({"font.size": 6.4, "axes.labelsize": 6.4})

    spine = 0.500 + ACS_DX   # figure-x carrying the approx sign and the shift
    left, width = ACS_LEFT, ACS_WIDTH
    geo = acs_layout()

    fig = plt.figure(figsize=(ACS_W, ACS_H))

    # --- the scalar summaries -------------------------------------------
    fig.text(spine, geo["row1"], "5 °C retention", fontsize=6.4, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.330 + ACS_DX, geo["row2"], "0.816", fontsize=11.0, color=INK, ha="center",
             va="top", fontweight="bold")
    fig.text(spine, geo["row2"], "\u2248", fontsize=11.0, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.670 + ACS_DX, geo["row2"], "0.825", fontsize=11.0, color=INK, ha="center",
             va="top", fontweight="bold")
    fig.text(0.330 + ACS_DX, geo["row3"], "capacity", fontsize=6.2, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.670 + ACS_DX, geo["row3"], "rested-voltage energy", fontsize=6.0,
             color=INK_MUTED, ha="center", va="top")

    # --- the trajectory the summaries average over ------------------------
    ax = fig.add_axes(geo["panel"])
    beyond = warm["q_end"] >= q_c
    ax.fill_between(
        np.r_[q_c, warm["q_end"][beyond]], CUTOFF,
        np.r_[np.interp(q_c, warm["q_end"], warm["v_ocv"]), warm["v_ocv"][beyond]],
        color=TERMS["window"], alpha=0.34, lw=0, zorder=1)
    band = np.linspace(0, q_c, 400)
    ax.fill_between(band, np.interp(band, cold["q_end"], cold["v_ocv"]),
                    np.interp(band, warm["q_end"], warm["v_ocv"]),
                    color=TERMS["shift"], alpha=0.45, lw=0, zorder=2)
    for run, colour, lw in ((warm, STYLE_WARM, 1.5), (cold, STYLE_COLD, 2.1)):
        ax.plot(run["q_end"], run["v_ocv"], color=colour, lw=lw, zorder=4)

    x_hi = warm["Q"] * 1.02
    q_spine = ((spine - left) / width) * x_hi
    v_band = float(np.interp(q_spine, warm["q_end"], warm["v_ocv"]))
    ax.plot([q_spine, q_spine], [3.28, v_band - 0.035], color=TERMS["shift"],
            lw=0.6, alpha=0.75, zorder=5)
    ax.text(q_spine, 4.00, "median shift", fontsize=6.2, color=INK_MUTED,
            ha="center", va="center", zorder=6)
    ax.text(q_spine, 3.46, "\u2212131 mV", fontsize=10.5, fontweight="bold",
            color=INK, ha="center", va="center", zorder=6)

    ax.text(1.16, 2.95, "25 °C", fontsize=6.4, color=STYLE_WARM,
            fontweight="bold", ha="center", va="bottom", zorder=6)
    ax.text(0.95, 2.48, "5 °C", fontsize=6.4, color=STYLE_COLD,
            fontweight="bold", ha="center", va="top", zorder=6)
    ax.text(q_c + 0.155, 1.90, "shorter\nwindow", fontsize=6.2, color=INK,
            ha="center", va="center", linespacing=1.1, zorder=6)

    ax.set_xlim(0, x_hi)
    ax.set_ylim(CUTOFF - 0.04, 4.14)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK_MUTED)
    ax.set_ylabel("rested voltage", labelpad=3, fontsize=6.2, color=INK_MUTED)
    ax.set_xlabel("discharged charge", labelpad=2, fontsize=6.2, color=INK_MUTED)

    # --- the accounting ledger -------------------------------------------
    lead = fig.text(left, geo["lead"], f"{100 * total / warm['Edel']:.1f} %",
                    fontsize=6.6, fontweight="bold", color=INK,
                    ha="left", va="center")
    fig.canvas.draw()
    span = lead.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    fig.text(left + span + 0.012, geo["lead"], "delivered-energy shortfall",
             fontsize=6.4, color=INK_MUTED, ha="left", va="center")
    strip = fig.add_axes(geo["strip"])
    start = 0.0
    for name, value, colour, opacity in (
            ("shorter window", shorter, TERMS["window"], 0.85),
            ("load\u2013rest gap", gap, TERMS["gap"], 0.72),
            ("rested shift", shift, TERMS["shift"], 0.85)):
        share = value / total
        strip.barh(0, share, left=start, height=1.0, color=colour,
                   alpha=opacity, lw=0)
        fig.text(left + width * (start + share / 2), geo["share"],
                 f"{100 * share:.0f} %", fontsize=6.4, fontweight="bold",
                 color=INK_MUTED, ha="center", va="bottom")  # above the strip
        fig.text(left + width * (start + share / 2), geo["names"], name,
                 fontsize=6.2,
                 color=INK_MUTED, ha="center", va="center")
        start += share
    strip.set_xlim(0, 1)
    strip.set_ylim(-0.5, 0.5)
    strip.set_xticks([])
    strip.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        strip.spines[side].set_visible(False)

    save_acs(fig, "graphical_abstract_acs_alt")

    print(f"ACS TOC variation A (two trajectories), not submitted: "
          f"graphical_abstract_acs_alt.pdf, {ACS_W:g} x {ACS_H:g} in")


def acs_toc_delta(cold: dict, warm: dict, q_c: float, shorter: float,
                  shift: float, gap: float, total: float) -> None:
    """ACS TOC, submitted: the displacement itself rather than two trajectories.

    Written as graphical_abstract_acs.pdf, the file the Letter includes after
    its abstract. Variation A (acs_toc above) draws both rested curves. At 3 in wide the 131 mV separation
    is 5 % of a 2.7 V axis, so the hero number is carried by a band a few points
    thick that needs 45 % opacity to survive the thumbnail -- the composition
    fights the data. Plotting the difference makes the same number 33 % of the
    panel, about nine times larger relative to the frame, and buys three things:

      * zero becomes meaningful. If the two retentions told the whole story the
        trace would sit on it. It starts just above and leaves.
      * the median is a full-width rule, not a hairline, and sits on the spine
        under the approximately-equal sign.
      * the trace simply stops where the cold discharge stopped, so the shorter
        window is the stretch of charge over which no comparison exists.
    """
    use_style()
    plt.rcParams.update({"font.size": 6.4, "axes.labelsize": 6.4})

    spine = 0.500 + ACS_DX
    left, width = ACS_LEFT, ACS_WIDTH
    geo = acs_layout()

    fig = plt.figure(figsize=(ACS_W, ACS_H))

    fig.text(spine, geo["row1"], "5 °C retention", fontsize=6.4, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.330 + ACS_DX, geo["row2"], "0.816", fontsize=11.0, color=INK, ha="center",
             va="top", fontweight="bold")
    fig.text(spine, geo["row2"], "\u2248", fontsize=11.0, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.670 + ACS_DX, geo["row2"], "0.825", fontsize=11.0, color=INK, ha="center",
             va="top", fontweight="bold")
    fig.text(0.330 + ACS_DX, geo["row3"], "capacity", fontsize=6.2, color=INK_MUTED,
             ha="center", va="top")
    fig.text(0.670 + ACS_DX, geo["row3"], "rested-voltage energy", fontsize=6.0,
             color=INK_MUTED, ha="center", va="top")

    ax = fig.add_axes(geo["panel"])
    grid = np.linspace(0.02 * q_c, q_c, 600)
    delta = 1000.0 * (np.interp(grid, cold["q_end"], cold["v_ocv"])
                      - np.interp(grid, warm["q_end"], warm["v_ocv"]))
    x_hi = warm["Q"]

    ax.axvspan(q_c, x_hi, color=TERMS["window"], alpha=0.16, lw=0, zorder=1)
    ax.axvline(q_c, color=TERMS["window"], lw=0.9, alpha=0.85, zorder=3)
    ax.plot([0, q_c], [0, 0], color=INK_MUTED, lw=0.6, ls=(0, (3, 2.8)),
            alpha=0.55, zorder=2)
    ax.plot([0, q_c], [MEDIAN_SHIFT] * 2, color=TERMS["shift"], lw=1.0, zorder=3)
    ax.plot(grid, delta, color=STYLE_COLD, lw=2.2, zorder=5,
            solid_capstyle="round")

    ax.text(0.055, 10, "0", fontsize=6.2, color=INK_MUTED, ha="left",
            va="bottom")
    word = ax.text(0.135, -219, "median", fontsize=6.4, color=TERMS["shift"],
                   ha="left", va="center", zorder=6)
    fig.canvas.draw()
    box = word.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    word_w = abs(inv.transform((box.x1, 0))[0] - inv.transform((box.x0, 0))[0])
    ax.text(0.135 + word_w + 0.022, -245, "\u2212131 mV", fontsize=10.5,
            fontweight="bold", color=INK, ha="left", va="center", zorder=6)
    ax.text((q_c + x_hi) / 2, -214, "shorter\nwindow", fontsize=6.2,
            color=INK, ha="center", va="center", linespacing=1.1, zorder=6)

    ax.set_xlim(0, x_hi)
    ax.set_ylim(-490, 62)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)
        ax.spines[side].set_color(INK_MUTED)
    ax.set_ylabel("rested-voltage shift", labelpad=3, fontsize=6.2,
                  color=INK_MUTED)
    ax.set_xlabel("discharged charge", labelpad=2, fontsize=6.2, color=INK_MUTED)

    lead = fig.text(left, geo["lead"], f"{100 * total / warm['Edel']:.1f} %",
                    fontsize=6.6, fontweight="bold", color=INK,
                    ha="left", va="center")
    fig.canvas.draw()
    span = lead.get_window_extent(fig.canvas.get_renderer()).width / fig.bbox.width
    fig.text(left + span + 0.012, geo["lead"], "delivered-energy shortfall",
             fontsize=6.4, color=INK_MUTED, ha="left", va="center")
    strip = fig.add_axes(geo["strip"])
    start = 0.0
    for name, value, colour, opacity in (
            ("shorter window", shorter, TERMS["window"], 0.85),
            ("load\u2013rest gap", gap, TERMS["gap"], 0.72),
            ("rested shift", shift, TERMS["shift"], 0.85)):
        share = value / total
        strip.barh(0, share, left=start, height=1.0, color=colour,
                   alpha=opacity, lw=0)
        fig.text(left + width * (start + share / 2), geo["share"],
                 f"{100 * share:.0f} %", fontsize=6.4, fontweight="bold",
                 color=INK_MUTED, ha="center", va="bottom")
        fig.text(left + width * (start + share / 2), geo["names"], name,
                 fontsize=6.2,
                 color=INK_MUTED, ha="center", va="center")
        start += share
    strip.set_xlim(0, 1)
    strip.set_ylim(-0.5, 0.5)
    strip.set_xticks([])
    strip.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        strip.spines[side].set_visible(False)

    save_acs(fig, "graphical_abstract_acs")
    print(f"ACS TOC graphic (submitted, difference view): "
          f"graphical_abstract_acs.pdf, {ACS_W:g} x {ACS_H:g} in, "
          "smallest type 6.0 pt (ACS floor 6 pt), colours from "
          "figure_style.TERMS/COLD/WARM")


def main() -> int:
    frame = pd.read_csv(SEGMENTS)
    frame = frame[frame.direction == "discharge"]
    cold, warm = trace(frame, COLD), trace(frame, WARM)
    q_c = min(cold["Q"], warm["Q"])

    shorter = (warm["Eocv"] - integral_to(warm["dq"], warm["v_ocv"], q_c)) - (
        cold["Eocv"] - integral_to(cold["dq"], cold["v_ocv"], q_c)
    )
    shift = integral_to(warm["dq"], warm["v_ocv"], q_c) - integral_to(
        cold["dq"], cold["v_ocv"], q_c
    )
    gap = (cold["Eocv"] - cold["Edel"]) - (warm["Eocv"] - warm["Edel"])
    total = warm["Edel"] - cold["Edel"]
    assert abs(shorter + shift + gap - total) < 1e-3, "identity does not close"

    if "--acs" in sys.argv[1:]:
        acs_toc_delta(cold, warm, q_c, shorter, shift, gap, total)
        acs_toc(cold, warm, q_c, shorter, shift, gap, total)
        return 0

    use_style()
    fig = plt.figure(figsize=(7.5, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.30)
    ax = fig.add_subplot(grid[0, 0])

    beyond = warm["q_end"] >= q_c
    ax.fill_between(
        np.r_[q_c, warm["q_end"][beyond]],
        CUTOFF,
        np.r_[np.interp(q_c, warm["q_end"], warm["v_ocv"]), warm["v_ocv"][beyond]],
        color=C_WIN, alpha=0.30, lw=0, zorder=1,
    )
    common = np.linspace(0, q_c, 400)
    ax.fill_between(
        common,
        np.interp(common, cold["q_end"], cold["v_ocv"]),
        np.interp(common, warm["q_end"], warm["v_ocv"]),
        color=C_SHIFT, alpha=0.45, lw=0, hatch="///", edgecolor="white", zorder=1,
    )
    for run, colour, label in ((warm, C_WARM, "25 °C"), (cold, C_COLD, "5 °C")):
        ax.plot(run["q_end"], run["v_ocv"], color=colour, lw=1.9, zorder=4, label=label)
        ax.plot(run["q_mid"], run["v_load"], color=colour, lw=1.1, ls="--", zorder=4)
    ax.axhline(CUTOFF, color=INK_MUTED, lw=0.9, ls=":", zorder=2)
    ax.axvline(q_c, color=INK_MUTED, lw=0.8, alpha=0.5, zorder=2)

    ax.annotate("charge never\ndelivered", xy=(q_c + 0.14, 2.28), fontsize=8.0,
                color=INK, ha="center", va="center", zorder=6, linespacing=1.15)
    ax.annotate("rested voltage\nlower by 131 mV", xy=(0.55, 2.62), fontsize=8.0,
                color=INK, ha="center", va="center", zorder=6, linespacing=1.15)
    ax.text(0.015, 0.055, "solid: rested   dashed: under 1.5 A load",
            transform=ax.transAxes, fontsize=7.2, color=INK_MUTED)
    ax.set_xlim(0, warm["Q"] * 1.03)
    ax.set_ylim(CUTOFF - 0.06, 4.16)
    ax.set_xlabel("charge delivered (Ah)")
    ax.set_ylabel("cell voltage (V)")
    ax.set_title("Commercial 18650 Na-ion cell, 25 → 5 °C", fontsize=9.5, pad=6)
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=8.0, frameon=False)

    bar = fig.add_subplot(grid[0, 1])
    terms = [
        ("shorter delivered-\ncharge window", shorter, C_WIN),
        ("load-to-rest\ngap growth", gap, C_GAP),
        ("rested-voltage\nshift", shift, C_SHIFT),
    ]
    bottom = 0.0
    for name, value, colour in terms:
        bar.bar(0, value, bottom=bottom, width=0.52, color=colour, alpha=0.85, lw=0)
        bar.text(0.34, bottom + value / 2,
                 f"{100 * value / total:.0f} %\n{name}",
                 fontsize=8.0, va="center", ha="left", color=INK, linespacing=1.2)
        bottom += value
    bar.set_xlim(-0.34, 1.55)
    bar.set_ylim(0, total * 1.06)
    bar.set_xticks([])
    bar.set_ylabel("energy shortfall (Wh)")
    bar.set_title(f"{total:.2f} Wh lost = {100 * total / warm['Edel']:.1f} %",
                  fontsize=9.5, pad=6)
    style_axis(bar)
    for side in ("bottom",):
        bar.spines[side].set_visible(True)

    fig.tight_layout(pad=0.6)
    for folder in ("results/figures", "manuscript/figures"):
        directory = ROOT / folder
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / "graphical_abstract.png", dpi=300)
        fig.savefig(directory / "graphical_abstract.pdf")
    plt.close(fig)

    width, height = fig.get_size_inches() * 300
    print(f"graphical abstract: {width:.0f} x {height:.0f} px at 300 dpi "
          f"(Elsevier minimum 1328 x 531)")
    print(f"terms: shorter window {shorter:.3f} Wh ({100 * shorter / total:.0f} %), "
          f"gap growth {gap:.3f} Wh ({100 * gap / total:.0f} %), "
          f"rested shift {shift:.3f} Wh ({100 * shift / total:.0f} %)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
