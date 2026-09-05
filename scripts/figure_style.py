#!/usr/bin/env python3
"""Shared figure style for every manuscript and supplementary figure.

The grammar below is deliberately narrow: every colour, weight and line style
has exactly one job, and the same job everywhere.

**Hierarchy, not equality.** The paper is about one cell. The SIB is therefore
the only saturated colour on any axis; LFP, NMC and LTO are graded greys,
separated additionally by marker and line style so identity never rests on
lightness alone. Weight carries the hierarchy before colour does -- the SIB
line is drawn ~1.7x the reference width -- so the ranking survives greyscale
printing and every form of colour-vision deficiency.

**One concept, one colour.** Three colour roles are reserved and must not be
reused for anything else:

    COLORS      cell identity       SIB / LFP / NMC / LTO
    TERMS       decomposition term  shorter window / load-to-rest gap / rested shift
    COLD, WARM  temperature         5 deg C (subject) vs 25 deg C (reference)

This separation fixes a real ambiguity in the previous palette. The three
decomposition terms were drawn from the *family* colours, so blue meant "SIB"
in Figure 2a,c,d and "shorter-window term" in Figure 2b; and Figure 1 gave the
shorter-window term purple while Figure 2b gave it blue -- the same concept in
two colours in adjacent figures. Muting the references frees the saturated
slots for the terms and removes both collisions.

**Printed size.** Sizes are the true printed width (COL_FULL = 7.2 in), so a
10 pt label prints at 10 pt.

**Gridlines.** Off by default. `style_axis(ax, grid="y")` opts a panel back in
where a reader must compare levels across a wide panel; nothing else gets them.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-energystorage")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/energystorage-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- printed widths (inches) -------------------------------------------------
COL_SINGLE = 3.5      # ~90 mm, single column
COL_FULL = 7.2        # ~183 mm, full text width

# --- type --------------------------------------------------------------------
# Journals ask for Arial/Helvetica. Fall back down a chain of metric-compatible
# grotesques so the figures still build on a machine without Arial installed.
SANS_STACK = ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
              "Arimo", "FreeSans", "DejaVu Sans"]

# --- cell families: one focal colour, three graded greys ---------------------
# The bracketed L* is the CIE lightness of each swatch; that spacing is what
# keeps the four separable in a greyscale print.
COLORS = {
    "SIB": "#0b4f8a",   # [L* 33] the analysed cell -- the only saturated family colour
    "LFP": "#6b6b66",   # [L* 45]
    "NMC": "#9a9a94",   # [L* 63]
    "LTO": "#c4c4be",   # [L* 79]
}
MARKERS = {"SIB": "o", "LFP": "s", "NMC": "^", "LTO": "D"}
# LTO is the outlier reference throughout the paper -- titanate anode, excluded
# from the graphite-anode mean -- and is drawn most recessively of the four.
LINESTYLES = {
    "SIB": "-",
    "LFP": "-",
    "NMC": (0, (5.0, 1.7)),
    "LTO": (0, (1.6, 1.7)),
}
LINEWIDTHS = {"SIB": 2.6, "LFP": 1.5, "NMC": 1.5, "LTO": 1.5}
MARKERSIZES = {"SIB": 7.0, "LFP": 5.0, "NMC": 5.0, "LTO": 5.0}
ZORDERS = {"SIB": 6, "LFP": 4, "NMC": 3, "LTO": 2}

# SIB first: it is the subject of every comparison.
FAMILY_ORDER = ["SIB", "LFP", "NMC", "LTO"]

# --- the three decomposition terms: permanent, never reused -------------------
# Saturation tracks size: the shorter-window term is the largest and the claim
# the paper rests on, so it carries the strongest colour.
TERMS = {
    "window": "#9e2b18",   # [L* 36] shorter delivered-charge window    (58 %)
    "gap": "#e39a2b",      # [L* 69] growth of the load-to-rest gap     (27 %)
    "shift": "#8b7cb0",    # [L* 55] common-window rested-voltage shift (15 %)
}
TERM_LABELS = {
    "window": "shorter window",
    "gap": "load-to-rest gap growth",
    "shift": "rested-voltage shift",
}

# --- temperature coding ------------------------------------------------------
# Cold is the subject state, warm the reference state, so cold takes the focal
# colour and warm the muted one -- the same rule as SIB vs references.
COLD = COLORS["SIB"]
WARM = "#9a9a94"

# Secondary categorical slots, for series that are neither families, terms, nor
# temperatures (circuit candidates, pulse time points). Ordered light -> dark so
# a reader reads them as a progression rather than four unrelated categories.
ACCENTS = ["#c7d4e2", "#8aa6c4", "#4a7ba8", "#0b4f8a", "#e39a2b", "#9e2b18"]

INK = "#1a1a19"
INK_MUTED = "#52514e"
GRID = "#e4e3df"

# Single-hue ramp anchored on the focal colour. A magnitude deserves one hue:
# multi-hue viridis introduced a fifth colour system and, once the categorical
# palette was muted, made the heat maps the most saturated thing in Figure 3 --
# outranking the panels that carry the argument.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sib_sequential", ["#f4f6f8", "#c3d3e2", "#7fa3c4", "#3d78ab", "#0b4f8a", "#062f52"])


def use_style() -> None:
    """Apply the shared rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3 -- required by most journals
        "ps.fonttype": 42,

        "font.family": "sans-serif",
        "font.sans-serif": SANS_STACK,
        # Math must use the same face as the surrounding text: matplotlib's
        # default mathtext set is DejaVu, so an axis label reading
        # "Median $R_{1s}$ (mOhm)" otherwise mixes two typefaces mid-string.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.default": "it",

        "font.size": 10.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,

        # A title is a short handle for the panel, not a restatement of the
        # caption: muted ink and left-set, so it sits below the data and the
        # bold panel letter in the reading order.
        "axes.titleweight": "regular",
        "axes.titlecolor": INK_MUTED,
        "axes.titlepad": 7,
        "axes.titlelocation": "left",
        "axes.labelpad": 4,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "text.color": INK,

        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": False,   # minor ticks are noise at this size
        "ytick.minor.visible": False,

        "lines.linewidth": 1.5,
        "lines.markersize": 5.0,
        "lines.markeredgewidth": 0.0,

        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.grid": False,
        "axes.axisbelow": True,

        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.6,
        "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.4,
        "legend.borderaxespad": 0.4,
    })


def style_axis(axis: plt.Axes, grid: str = "none") -> None:
    """Open frame; gridlines only where explicitly asked for.

    `grid` is 'none' (default), 'y', 'x', or 'both'.
    """
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid != "none":
        axis.grid(True, axis=grid, which="major",
                  color=GRID, linewidth=0.6, zorder=0)
    axis.set_axisbelow(True)


def panel_label(axis: plt.Axes, letter: str, dx: float = -0.14, dy: float = 1.03,
                fontsize: float = 11.0, va: str = "bottom") -> None:
    """Bold panel letter at the upper-left, set into the left margin by default.

    It sits left of the axes rather than above them so that a left-aligned
    panel title can start at the axis origin without colliding with it.

    Passing dx/dy inside the unit square with va="top" instead parks the letter
    in the panel's own top-left corner, which costs no layout space at all. The
    ACS figures do that; the defaults keep the JES artwork as it was.
    """
    axis.text(dx, dy, letter, transform=axis.transAxes,
              fontsize=fontsize, fontweight="bold", color=INK,
              ha="left", va=va)


def family_kwargs(family: str, **overrides) -> dict:
    """Full plot styling for a cell family: colour, marker, weight, order.

    Identity rests on four independent channels (hue, lightness, marker, line
    style) so it survives greyscale and every colour-vision deficiency.
    """
    kwargs = {
        "color": COLORS.get(family, INK_MUTED),
        "marker": MARKERS.get(family, "o"),
        "linestyle": LINESTYLES.get(family, "-"),
        "linewidth": LINEWIDTHS.get(family, 1.5),
        "markersize": MARKERSIZES.get(family, 5.0),
        "zorder": ZORDERS.get(family, 3),
        "markeredgecolor": "white",
        "markeredgewidth": 0.6,
    }
    kwargs.update(overrides)
    return kwargs


def label_line(axis: plt.Axes, x, y, text: str, family: str | None = None,
               color: str | None = None, dx: float = 6.0, dy: float = 0.0,
               ha: str = "left", va: str = "center", weight: str | None = None,
               fontsize: float = 9.0) -> None:
    """Direct label at the end of a series, in the series' own colour.

    Preferred over a legend wherever the lines separate enough to carry one:
    it removes the eye's round trip to a key.
    """
    hue = color or COLORS.get(family or "", INK)
    axis.annotate(text, xy=(x, y), xycoords="data",
                  xytext=(dx, dy), textcoords="offset points",
                  color=hue, fontsize=fontsize, ha=ha, va=va,
                  fontweight=weight or ("bold" if family == "SIB" else "regular"),
                  annotation_clip=False, zorder=8)


def save_figure(fig: plt.Figure, name: str, folders: list[str] | None = None) -> None:
    """Write PNG (raster, for the audit and quick viewing) and PDF (vector, for print)."""
    targets = folders or ["results/figures", "manuscript/figures"]
    stem = Path(name).stem
    for folder in targets:
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{stem}.png")
        fig.savefig(directory / f"{stem}.pdf")
    plt.close(fig)
