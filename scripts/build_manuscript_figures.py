#!/usr/bin/env python3
"""Build manuscript-ready Figures 1-6 from processed descriptor tables.

Styling (sizes, palette, markers, panel letters) lives in scripts/figure_style.py
so that the supplementary figures built by other scripts match exactly.

SHARED ARTWORK -- READ BEFORE EDITING A FIGURE.
Figures 1 and 2 are used by BOTH manuscripts:

    manuscript/latex/JES/Manuscript.tex                (J. Energy Storage)
    manuscript/latex/Manuscript_ACS_Energy_Letters.tex (ACS Energy Letters)

Their captions are written independently and describe panels, insets and axis
labels in words. Changing a panel, adding an inset, or relabelling an axis
therefore silently invalidates a caption in the *other* document. That has
happened twice: a retention inset added for the ACS version went undescribed in
the JES caption, and an axis relabel orphaned a symbol the JES caption
introduced. After touching these figures, grep both captions.

Axis labels deliberately name quantities in words rather than symbols, because
the two papers use different notation for the load-to-rest gap.

Where the two papers genuinely need different artwork, add an ACS variant that
writes its own file rather than editing the shared one -- see
figure2_energy_decomposition_acs() and the --acs-only flag. The panel bodies are
shared helpers, so a change to the science reaches both; only layout and wording
branch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from figure_style import (
    ACCENTS,
    COL_FULL,
    COLORS,
    FAMILY_ORDER,
    GRID,
    INK,
    INK_MUTED,
    MARKERS,
    SEQUENTIAL,
    TERMS,
    TERM_LABELS,
    family_kwargs,
    label_line,
    panel_label,
    save_figure,
    style_axis,
    use_style,
)

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

use_style()


TARGET_LABELS = {
    "target_usable_energy_retention_vs_25c": "OCV-window energy",
    "target_pulse_r_1s_growth_vs_25c": "R1s growth",
    "target_pulse_power_proxy_vs_25c": "Power proxy",
    "target_low_temperature_penalty": "Low-temp penalty",
}


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def draw_box(axis, xy, text, color, width=0.22):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), width, 0.125,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.4, edgecolor=color, facecolor="white", zorder=3,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + 0.0625, text, ha="center", va="center",
              fontsize=9.5, color=INK, zorder=4)


def draw_arrow(axis, start, end):
    arrow = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                            linewidth=1.1, color="0.45", zorder=2)
    axis.add_patch(arrow)


def figure1_workflow() -> None:
    fig, axis = plt.subplots(figsize=(COL_FULL, 4.1))
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)

    raw = [
        ("Raw checkup\ntraces", (0.03, 0.70)),
        ("HPPC", (0.03, 0.53)),
        ("EIS", (0.03, 0.36)),
        ("Entropy +\nthermal", (0.03, 0.19)),
    ]
    descriptors = [
        ("Capacity +\nOCV energy", (0.36, 0.70), ACCENTS[0]),
        ("Raw pulse R,\npolarisation", (0.36, 0.53), ACCENTS[1]),
        ("EIS spectral\nproxies", (0.36, 0.36), ACCENTS[2]),
        ("Entropy +\nthermal context", (0.36, 0.19), ACCENTS[3]),
    ]
    outputs = [
        ("Integrated\ndescriptor matrix", (0.69, 0.60), ACCENTS[0]),
        ("Sensitivity\nchecks", (0.69, 0.41), ACCENTS[1]),
        ("Evidence\nchecks", (0.69, 0.22), ACCENTS[2]),
    ]

    for label, xy in raw:
        draw_box(axis, xy, label, "0.45")
    for label, xy, color in descriptors:
        draw_box(axis, xy, label, color)
    for label, xy, color in outputs:
        draw_box(axis, xy, label, color, width=0.26)

    for _, start in raw:
        draw_arrow(axis, (0.275, start[1] + 0.0625), (0.335, start[1] + 0.0625))
    for _, start, _ in descriptors:
        draw_arrow(axis, (0.605, start[1] + 0.0625), (0.665, 0.663))
    draw_arrow(axis, (0.82, 0.575), (0.82, 0.558))
    draw_arrow(axis, (0.82, 0.385), (0.82, 0.368))

    for x, heading in [(0.14, "Archive streams"),
                       (0.47, "Descriptors"),
                       (0.80, "Evidence path")]:
        axis.text(x, 0.90, heading, fontsize=10, fontweight="bold",
                  color=INK, ha="center")
    axis.text(0.03, 0.045,
              "Equivalent-circuit parameters are screened, then excluded:\n"
              "no candidate circuit passes the reliability screen.",
              fontsize=9, color=INK_MUTED)
    save_figure(fig, "figure1_data_architecture.png")


def figure3_pulse_maps(pulse_summary: pd.DataFrame, soc_map: pd.DataFrame,
                       pulse_sens: pd.DataFrame) -> None:
    """Main-text pulse figure: two panels, two questions.

    (a) what happens -- the SIB resistance rises far more steeply on cooling;
    (b) whether that conclusion depends on the chosen pulse time point -- it
    does not.

    The pulse-block maps that used to occupy (c) and (d) now build separately
    as `figure_s2_pulse_block_maps` for the supplement. They answer a third
    question -- whether the excess is broadly distributed over protocol-block
    position -- which is a robustness diagnostic rather than a pillar of the
    argument, and at half the figure they outranked the panels that carry it.
    """
    fig, axes_row = plt.subplots(1, 2, figsize=(COL_FULL, 3.2))
    axes = np.array([axes_row, axes_row])   # keep the [0, 0] / [0, 1] indexing
    subset = pulse_summary[(pulse_summary["pulse_direction"] == "discharge")
                           & (pulse_summary["abs_c_rate"] == 1.0)]
    for family in FAMILY_ORDER:
        group = subset[subset["cell_family"] == family].sort_values("temperature_deg_c")
        if group.empty:
            continue
        axes[0, 0].plot(group["temperature_deg_c"], group["median_r_1s_mohm"],
                        label=family, **family_kwargs(family))
    axes[0, 0].set_title("1C discharge $R_{1\,\mathrm{s}}$")
    axes[0, 0].set_xlabel("Temperature (°C)")
    axes[0, 0].set_ylabel("Median $R_{1\,\mathrm{s}}$ (m$\Omega$)")
    axes[0, 0].set_xticks([5, 15, 25, 35, 45])
    for family in FAMILY_ORDER:
        group = subset[subset["cell_family"] == family].sort_values("temperature_deg_c")
        if group.empty:
            continue
        label_line(axes[0, 0], 5.0, float(group["median_r_1s_mohm"].iloc[0]),
                   family, family=family, dx=-7, ha="right")
    axes[0, 0].set_xlim(-2.0, 47.0)
    style_axis(axes[0, 0])
    panel_label(axes[0, 0], "a")

    # On discharge the pulse-end descriptor coincides exactly with the 10 s one
    # (the protocol pulse is 10 s long), so plotting both with the same style
    # hides one curve completely. "end" is drawn dashed with open markers on top
    # of the solid 10 s curve, and its legend entry says so.
    # The plotted ratio is against the most resistive lithium-ion reference at
    # each temperature, not against the graphite-anode mean: an average of
    # 83.2 and 42.9 mOhm describes neither cell, and the paper itself calls
    # that average the weakest form of the comparison. This is also the most
    # conservative per-family reading available.
    # The first post-step sample is excluded here: the source study states a
    # 100 ms equipment limit, so it is an archive-trace diagnostic only and no
    # claim rests on it. Showing it in a main-text figure invites the reader to
    # use it. It remains available in the released descriptor tables.
    order = ["100ms", "1s", "10s", "end"]
    labels = {"100ms": "100 ms", "1s": "1 s",
              "10s": "10 s", "end": "end (= 10 s)"}
    for index, timepoint in enumerate(order):
        group = pulse_sens[pulse_sens["timepoint"] == timepoint].sort_values("temperature_deg_c")
        if group.empty:
            continue
        dashed = timepoint == "end"
        axes[0, 1].plot(
            group["temperature_deg_c"], group["sib_to_max_li_resistance_ratio"],
            color=ACCENTS[index],
            marker="o", markersize=4.6,
            markerfacecolor="white" if dashed else ACCENTS[index],
            markeredgecolor=ACCENTS[index],
            markeredgewidth=1.2 if dashed else 0.6,
            linestyle=(0, (4, 2)) if dashed else "-",
            linewidth=1.6, label=labels[timepoint],
        )
    axes[0, 1].axhline(1.0, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    axes[0, 1].set_title("Time-point sensitivity")
    axes[0, 1].set_xlabel("Temperature (°C)")
    axes[0, 1].set_ylabel("SIB / most resistive Li reference")
    axes[0, 1].set_xticks([5, 15, 25, 35, 45])
    axes[0, 1].margins(y=0.20)
    # The unity line is the argument of this panel: annotate it rather than
    # leaving the reader to infer what crossing it means.
    axes[0, 1].text(0.03, 0.05, "below 1.0: SIB less resistive than\nthe most resistive Li reference",
                    transform=axes[0, 1].transAxes, ha="left", va="bottom",
                    fontsize=7.8, color=INK_MUTED, linespacing=1.25)
    style_axis(axes[0, 1])
    axes[0, 1].legend(ncol=2, fontsize=8.5, loc="upper right")
    panel_label(axes[0, 1], "b")

    fig.tight_layout(w_pad=2.6)
    save_figure(fig, "figure4_raw_pulse_bottleneck.png")


def figure_s2_pulse_block_maps(soc_map: pd.DataFrame) -> None:
    """Supplementary: is the SIB cold excess broadly distributed over the
    discharge, or does it sit in one part of the protocol?

    Moved out of main-text Figure 3, where it answered a third question at the
    cost of half the figure. LFP is the comparator because it is the closest
    reference to the SIB in cold resistance.
    """
    fig, axes_row = plt.subplots(1, 2, figsize=(COL_FULL, 3.4))
    heat_subset = soc_map[
        (soc_map["pulse_direction"] == "discharge")
        & (soc_map["abs_c_rate"] == 1.0)
        & (soc_map["cell_family"].isin(["SIB", "LFP"]))
    ]
    temperatures = sorted(heat_subset["temperature_deg_c"].dropna().unique())
    soc_values = sorted(heat_subset["nominal_soc_percent"].dropna().unique(), reverse=True)
    vmax = float(np.nanpercentile(heat_subset["median_r_1s_mohm"], 98))
    vmin = float(np.nanpercentile(heat_subset["median_r_1s_mohm"], 2))
    for position, (axis, family, letter) in enumerate(
            zip([axes_row[0], axes_row[1]], ["SIB", "LFP"], ["a", "b"])):
        view = heat_subset[heat_subset["cell_family"] == family]
        pivot = (
            view.pivot_table(index="nominal_soc_percent", columns="temperature_deg_c",
                             values="median_r_1s_mohm", aggfunc="median")
            .reindex(index=soc_values, columns=temperatures)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(pivot, aspect="auto", cmap=SEQUENTIAL, vmin=vmin, vmax=vmax)
        axis.set_title(f"{family} pulse-block map")
        axis.set_xticks(np.arange(len(temperatures)))
        axis.set_xticklabels([f"{int(temp)}" for temp in temperatures])
        axis.set_yticks(np.arange(0, len(soc_values), 2))
        axis.set_yticklabels([f"{int(soc_values[i])}" for i in range(0, len(soc_values), 2)])
        axis.set_xlabel("Temperature (°C)")
        axis.minorticks_off()
        if position == 0:
            axis.set_ylabel("Pulse-block index (% of schedule)")
        panel_label(axis, letter)
    fig.tight_layout(w_pad=1.8, rect=(0, 0, 0.90, 1))
    bar = fig.colorbar(image, ax=axes_row.ravel().tolist(),
                       fraction=0.05, pad=0.03, aspect=18)
    bar.set_label("Median $R_{1\\,\\mathrm{s}}$ (m$\\Omega$)")
    bar.outline.set_linewidth(0.0)
    save_figure(fig, "figure_s2_pulse_block_maps.png")


def figure4_eis(eis_summary: pd.DataFrame, consistency: pd.DataFrame,
                ecm_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(COL_FULL, 2.95))
    families = [f for f in FAMILY_ORDER if f in set(eis_summary["cell_family"])]
    ordered = eis_summary.set_index("cell_family").reindex(families)
    x = np.arange(len(families))
    width = 0.36
    axes[0].bar(x - width / 2 - 0.01, ordered["median_series_resistance_mohm"],
                width=width, label="high-frequency $Z'$", color=ACCENTS[0])
    axes[0].bar(x + width / 2 + 0.01, ordered["median_low_freq_zre_mohm"],
                width=width, label="low-frequency $Z'$", color=ACCENTS[1])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(families)
    axes[0].set_ylabel("Resistance proxy (m$\Omega$)")
    axes[0].set_title("EIS proxies")
    axes[0].xaxis.set_tick_params(which="minor", bottom=False)
    style_axis(axes[0])
    axes[0].margins(y=0.30)
    axes[0].legend(loc="upper right", fontsize=8.5)
    panel_label(axes[0], "a")

    for _, row in consistency.iterrows():
        family = row["cell_family"]
        axes[1].scatter(row["eis_low_freq_zre_mohm"], row["pulse_1c_discharge_r_1s_mohm"],
                        s=90, zorder=3, color=COLORS.get(family, INK_MUTED),
                        marker=MARKERS.get(family, "o"),
                        edgecolor="white", linewidth=0.8)
        nudge = {"LFP": (-6, -14), "SIB": (8, 2)}.get(family, (8, 2))
        axes[1].annotate(family,
                         (row["eis_low_freq_zre_mohm"], row["pulse_1c_discharge_r_1s_mohm"]),
                         xytext=nudge, textcoords="offset points",
                         fontsize=9.5, color=INK)
    axes[1].set_xlabel("EIS low-frequency $Z'$ (m$\Omega$)")
    axes[1].set_ylabel("Pulse $R_{1\,\mathrm{s}}$ (m$\Omega$)")
    axes[1].set_title("EIS\u2013pulse consistency")
    axes[1].margins(0.18)
    style_axis(axes[1], grid="both")
    panel_label(axes[1], "b")

    ecm = ecm_summary.copy()
    # Every screened circuit is plotted, including the two-arc candidates: the
    # argument is that a second arc does not repair the residual, which is only
    # visible if all four appear.
    ecm_models = [
        ("randles_cpe", "Randles-CPE"),
        ("randles_cpe_warburg", "Randles-CPE-W"),
        ("two_arc_cpe", "two-arc CPE"),
        ("two_arc_cpe_warburg", "two-arc CPE-W"),
    ]
    ecm_models = [entry for entry in ecm_models if entry[0] in set(ecm["model"])]
    width = 0.8 / max(len(ecm_models), 1)
    for index, (model_name, label) in enumerate(ecm_models):
        view = ecm[ecm["model"] == model_name].set_index("cell_family").reindex(families)
        offset = (index - (len(ecm_models) - 1) / 2) * width
        axes[2].bar(x + offset, view["median_residual_nrmse"], width=width * 0.92,
                    label=label, color=ACCENTS[index % len(ACCENTS)])
    axes[2].axhline(0.05, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=4)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(families)
    axes[2].set_ylabel("Median scaled NRMSE")
    axes[2].set_title("Equivalent-circuit screen")
    axes[2].xaxis.set_tick_params(which="minor", bottom=False)
    style_axis(axes[2])
    axes[2].set_ylim(0, 0.142)
    axes[2].legend(ncol=1, fontsize=7.5, loc="upper left", labelspacing=0.25,
                   handlelength=1.3, handletextpad=0.45)
    panel_label(axes[2], "c")
    fig.tight_layout(w_pad=2.4)
    save_figure(fig, "figure5_eis_bottleneck_decomposition.png")


def figure5_ocv_entropy_thermal(ocv_sens: pd.DataFrame, ocv_slope: pd.DataFrame,
                                entropy_summary: pd.DataFrame,
                                thermal_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(COL_FULL, 5.2))
    flat = axes.ravel()
    view = ocv_sens[(ocv_sens["temperature_deg_c"] == 5)
                    & (ocv_sens["smoothing_window_segments"] == 1)]
    windows = ["full", "early_discharge", "central", "late_discharge"]
    families = FAMILY_ORDER
    x = np.arange(len(windows))
    width = 0.19
    for index, family in enumerate(families):
        values = (
            view[view["cell_family"] == family]
            .set_index("capacity_window").reindex(windows)["energy_retention_vs_25c"]
            .to_numpy(dtype=float)
        )
        flat[0].bar(x + (index - 1.5) * width, values, width=width * 0.92,
                    label=family, color=COLORS[family])
    flat[0].axhline(1.0, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=4)
    flat[0].set_xticks(x)
    flat[0].set_xticklabels(["full", "early", "central", "late"])
    flat[0].set_ylabel("5 °C energy retention")
    flat[0].set_title("OCV-window sensitivity")
    flat[0].set_ylim(0, 1.26)
    flat[0].xaxis.set_tick_params(which="minor", bottom=False)
    style_axis(flat[0])
    flat[0].legend(ncol=4, fontsize=8.5, loc="upper center")
    panel_label(flat[0], "a")

    panels = [
        (1, ocv_slope[(ocv_slope["direction"] == "discharge")
                      & (ocv_slope["temperature_deg_c"] == 5)],
         "ocv_slope_penalty_v", "Central OCV span",
         "Central OCV span (V)", "b"),
        (2, entropy_summary[entropy_summary["direction"] == "discharge"],
         "mean_abs_dudt_mv_per_k", "Entropy coefficient",
         "Mean $|\mathrm{d}U/\mathrm{d}T|$ (mV K$^{-1}$)", "c"),
        (3, thermal_summary[thermal_summary["direction"] == "discharge"],
         "mean_temperature_rise_max_k", "Thermal response",
         "Max temperature rise (K)", "d"),
    ]
    for slot, frame, column, title, ylabel, letter in panels:
        values = frame.set_index("cell_family").reindex(families)[column]
        bars = flat[slot].bar(families, values, width=0.62,
                              color=[COLORS[f] for f in families])
        for rect, value in zip(bars, values):
            flat[slot].annotate(f"{value:.2f}", (rect.get_x() + rect.get_width() / 2,
                                                 rect.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", fontsize=8.5, color=INK_MUTED)
        flat[slot].set_title(title)
        flat[slot].set_ylabel(ylabel)
        flat[slot].margins(y=0.16)
        flat[slot].xaxis.set_tick_params(which="minor", bottom=False)
        style_axis(flat[slot])
        panel_label(flat[slot], letter)
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    save_figure(fig, "figure6_ocv_entropy_thermal_coupling.png")


def _fig2_panel_delivered_energy(ax, summary: pd.DataFrame, families, *,
                                 label_dx: float = -7, xlim=(-2.0, 47.0)) -> None:
    """(a) delivered-energy retention against temperature, one line per family.

    The direct labels sit left of the 5 degC points, so the left limit has to
    leave them room; at the default they crowd the axis and the markers.
    """
    for family in families:
        view = summary[summary["cell_family"] == family].sort_values("temperature_deg_c")
        ax.plot(view["temperature_deg_c"],
                view["delivered_energy_wh_retention_vs_25c"],
                label=family, **family_kwargs(family))
    ax.axhline(1.0, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Delivered energy / 25 °C")
    ax.set_title("Delivered energy")
    ax.set_xticks([5, 15, 25, 35, 45])
    # Direct labels at the cold end, where the four series are furthest apart.
    for family in families:
        view = summary[summary["cell_family"] == family].sort_values("temperature_deg_c")
        y0 = float(view["delivered_energy_wh_retention_vs_25c"].iloc[0])
        label_line(ax, 5.0, y0, family, family=family, dx=label_dx, ha="right")
    ax.set_xlim(*xlim)
    style_axis(ax)


def _fig2_panel_shortfall(ax, shortfall: pd.DataFrame, families, *,
                          conservative_line: bool, ymargin: float = 0.46,
                          ylim: tuple[float, float] | None = None,
                          xlim: tuple[float, float] | None = None) -> None:
    """(b) the 5 degC shortfall split into the three measured accounting terms.

    ``conservative_line`` draws the dashed voltage-side reassignment bound. The
    ACS variant omits it: it is a sensitivity scenario, not a fourth measured
    component, and it competes with the three bars for the reader's attention.
    """
    cold = shortfall[shortfall["temperature_deg_c"] == 5.0].set_index("cell_family").reindex(families)
    # All three terms are measured (Eqs. 5 and 6). The rested-voltage shift is
    # the only one open to attribution, so it is hatched rather than drawn as a
    # separate assumed component stacked on top of the measurement.
    polarisation = cold["shortfall_polarisation_wh"].to_numpy(dtype=float)
    relaxed_shift = cold["shortfall_relaxed_shift_wh"].to_numpy(dtype=float)
    truncation = cold["shortfall_truncation_wh"].to_numpy(dtype=float)

    ax.bar(families, polarisation, width=0.62, color=TERMS["gap"],
           label=TERM_LABELS["gap"])
    ax.bar(families, relaxed_shift, bottom=polarisation, width=0.62,
           color=TERMS["shift"], hatch="///", edgecolor="white",
           linewidth=0.8, label=TERM_LABELS["shift"])
    ax.bar(families, truncation, bottom=polarisation + relaxed_shift,
           width=0.62, color=TERMS["window"], label=TERM_LABELS["window"])
    if conservative_line:
        # The attribution bound: everything below this line is polarisation-associated
        # if the whole rested-voltage shift is credited to polarisation.
        for index, family in enumerate(families):
            upper = polarisation[index] + relaxed_shift[index]
            ax.plot([index - 0.31, index + 0.31], [upper, upper],
                    color=INK_MUTED, linestyle=(0, (2, 1.6)), linewidth=1.0,
                    zorder=5)
    ax.set_ylabel("Shortfall vs 25 °C (Wh)")
    ax.set_title("Shortfall composition, 5 °C")
    ax.margins(y=ymargin)
    if ylim is not None:
        # Just clear of the tallest bar (1.17 Wh) and of the legend above the
        # three shorter ones; the default margin leaves a third of the panel empty.
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.xaxis.set_tick_params(which="minor", bottom=False)
    handles, labels_b = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels_b[::-1], fontsize=7.5,
              loc="upper right", labelspacing=0.25,
              handlelength=1.3, handletextpad=0.45)
    style_axis(ax)


def _fig2_panel_load_to_rest(ax, summary: pd.DataFrame, families) -> None:
    """(c, JES only) effective mean load-to-rest gap against temperature."""
    for family in families:
        view = summary[summary["cell_family"] == family].sort_values("temperature_deg_c")
        ax.plot(view["temperature_deg_c"], 1000.0 * view["mean_overpotential_v"],
                label=family, **family_kwargs(family))
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Effective mean load-to-rest gap (mV)")
    ax.set_title("Load-to-rest gap")
    ax.set_xticks([5, 15, 25, 35, 45])
    for family in families:
        view = summary[summary["cell_family"] == family].sort_values("temperature_deg_c")
        y0 = 1000.0 * float(view["mean_overpotential_v"].iloc[0])
        label_line(ax, 5.0, y0, family, family=family, dx=-7, ha="right")
    ax.set_xlim(-2.0, 47.0)
    style_axis(ax)


def _fig2_panel_rested_shift(ax, summary: pd.DataFrame, families, *,
                             inset_title: str,
                             inset_rect: list[float],
                             median_mv: float | None,
                             median_xy: tuple[float, float] = (0.985, 0.0),
                             median_align: tuple[str, str] = ("right", "bottom"),
                             sib_label_xy: tuple[float, float] = (0.60, -158.0),
                             ylim: tuple[float, float] | None = None,
                             inset_axis_titles: bool = False) -> None:
    """(d) matched-throughput rested-voltage shift, with the retention inset.

    ``median_mv`` draws the headline number inside the panel. It is passed in
    rather than recomputed so that the annotation cannot drift from the value
    quoted in the title and abstract; the caller takes it from
    table_common_window_ocv_comparison.csv.
    """
    seg = read_csv("results/tables/table_delivered_energy_segments.csv")
    seg = seg[seg["direction"] == "discharge"]
    for family in families:
        cold_run = seg[(seg.cell_family == family) & (seg.temperature_deg_c == 5.0)] \
            .sort_values("segment_index")
        warm_run = seg[(seg.cell_family == family) & (seg.temperature_deg_c == 25.0)] \
            .sort_values("segment_index")
        if cold_run.empty or warm_run.empty:
            continue
        qc = np.cumsum(cold_run.segment_capacity_ah.to_numpy())
        qw = np.cumsum(warm_run.segment_capacity_ah.to_numpy())
        window = min(qc[-1], qw[-1])
        # the final few per cent diverge as both curves approach cutoff;
        # the body of the window is what the quoted median describes
        grid = np.linspace(0.02 * window, 0.95 * window, 200)
        dv = 1000.0 * (np.interp(grid, qc, cold_run.ocv_voltage_v.to_numpy())
                       - np.interp(grid, qw, warm_run.ocv_voltage_v.to_numpy()))
        ax.plot(grid / window, dv, label=family,
                **family_kwargs(family, markevery=40))
    ax.axhline(0.0, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    label_line(ax, sib_label_xy[0], sib_label_xy[1], "SIB", family="SIB",
               dx=0, dy=11, ha="center", va="bottom")

    if median_mv is not None:
        # The number in the title, drawn where a reader who scans only the
        # figures will meet it.
        blended = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhline(median_mv, color=COLORS["SIB"], linestyle=(0, (1.4, 2.0)),
                   linewidth=1.1, alpha=0.8, zorder=3)
        # Left of the crossing point and just below the line: the SIB trace is
        # still above -100 mV there and the references hug zero, so the label
        # clears every curve and sits above the inset.
        ax.text(median_xy[0], median_mv + median_xy[1],
                f"median SIB = −{abs(median_mv):.0f} mV",
                transform=blended, fontsize=8.5, color=COLORS["SIB"],
                ha=median_align[0], va=median_align[1], zorder=6)

    # Inset: the two SIB retention curves that hide this panel's shift. They are
    # near-identical by construction -- that cancellation IS the result -- so
    # showing them beside the shift makes the concealment visible rather than
    # asserted. Without it the reader takes the central claim on trust.
    inset = ax.inset_axes(inset_rect)
    sib = summary[summary["cell_family"] == "SIB"].sort_values("temperature_deg_c")
    temp = sib["temperature_deg_c"].to_numpy()
    cap = sib["capacity_ah_retention_vs_25c"].to_numpy()
    eno = sib["ocv_energy_wh_retention_vs_25c"].to_numpy()
    inset.plot(temp, cap, color=COLORS["SIB"], linewidth=1.5, marker="o",
               markersize=3.2, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    inset.plot(temp, eno, color=COLORS["SIB"], linewidth=1.5, marker="s",
               markersize=3.2, linestyle=(0, (3.0, 1.6)), markerfacecolor="white",
               markeredgecolor=COLORS["SIB"], markeredgewidth=1.0, zorder=5)
    # the two 5 degC values, set in the inset's empty upper-left rather than beside
    # the points, where they collided with the axis labels
    inset.text(0.40, 0.15, "5 °C:  0.816 / 0.825", transform=inset.transAxes,
               fontsize=6.5, color=INK, ha="left", va="center")
    inset.set_xticks([5, 25, 45])
    inset.set_yticks([0.8, 1.0])
    inset.tick_params(labelsize=6.3, length=2.3, width=0.6, pad=1.5)
    if inset_axis_titles:
        # Without these the inset's two axes are unnamed, and its tick labels
        # read as if they belonged to the panel behind it.
        inset.set_xlabel("temperature (°C)", fontsize=6.3, labelpad=1.6)
        inset.set_ylabel("retention", fontsize=6.3, labelpad=1.6)
    inset.set_title(inset_title, fontsize=6.6,
                    color=INK_MUTED, pad=2.5, loc="left")
    for side in ("top", "right"):
        inset.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        inset.spines[side].set_linewidth(0.6)
    inset.patch.set_alpha(0.0)
    ax.set_title("Rested-voltage shift, 5 − 25 °C")
    ax.set_xlabel("fraction of the common charge window")
    ax.set_xlim(0, 0.97)
    if ylim is not None:
        # Only the SIB trace leaves the axis, over the last few per cent of the
        # window where both curves are converging on cutoff. Every reference
        # curve stays inside, and the caption states the clip and the extremum.
        ax.set_ylim(*ylim)
    ax.set_ylabel("$V_{\\rm rest}(5)-V_{\\rm rest}(25)$ (mV)")
    # No key here: colour, marker and line style are fixed for every family in
    # (a) and (c) and never change, so a third legend would only cost the space
    # the inset needs. The three references are identified by that grammar.
    style_axis(ax)


def figure2_energy_decomposition(summary: pd.DataFrame, shortfall: pd.DataFrame,
                                 common_window: pd.DataFrame) -> None:
    """Delivered energy, where the cold shortfall goes, and what the retentions hide.

    Merged in the sparring-review revision from the former Figures 2 and 3. The
    two panels dropped were the discharge-capacity and rested-voltage-energy
    retention curves: they are near-identical by construction -- their
    cancellation is the point -- and both are tabulated as adjacent rows of
    Table 4, immediately above the common-window shift that panel (d) draws.
    Plotting them cost two panels and showed one curve twice.

    This is the J. Energy Storage layout. The ACS Energy Letters Letter uses
    figure2_energy_decomposition_acs() below; do not merge the two.
    """
    families = FAMILY_ORDER
    fig, axes = plt.subplots(2, 2, figsize=(COL_FULL, 5.6))

    _fig2_panel_delivered_energy(axes[0, 0], summary, families)
    panel_label(axes[0, 0], "a")
    _fig2_panel_shortfall(axes[0, 1], shortfall, families, conservative_line=True)
    panel_label(axes[0, 1], "b")
    _fig2_panel_load_to_rest(axes[1, 0], summary, families)
    panel_label(axes[1, 0], "c")
    _fig2_panel_rested_shift(axes[1, 1], summary, families,
                             inset_title="SIB retentions coincide",
                             inset_rect=[0.10, 0.09, 0.38, 0.26],
                             median_mv=None)
    panel_label(axes[1, 1], "d")

    fig.tight_layout(w_pad=1.8, h_pad=2.4)
    save_figure(fig, "figure2_energy_decomposition.png")


def figure2_energy_decomposition_acs(summary: pd.DataFrame, shortfall: pd.DataFrame,
                                     common_window: pd.DataFrame) -> None:
    """ACS Energy Letters variant: three panels, the rested-voltage shift enlarged.

    Three differences from the JES layout, all from the round-2 review:

    1. the cross-family load-to-rest gap panel is dropped. Figure 1c already
       shows that term geometrically for the SIB, SI S3.1 gives the numbers, and
       a prominent four-family panel invites the chemistry-class reading the
       Letter spends a paragraph disclaiming;
    2. the matched-throughput shift becomes a full-width panel and carries the
       median -131 mV in the artwork. It is the number in the title;
    3. the dashed voltage-side reassignment is removed from the bars, so the
       main figure shows only measured accounting terms.

    The inset wording also changes: "coincide" is not true of 0.816 and 0.825,
    which is why the ACS title says "nearly equal".
    """
    families = FAMILY_ORDER
    # The shift panel carries the title result, so it gets the height: its
    # informative band is only about a quarter of the plotted data range.
    fig = plt.figure(figsize=(COL_FULL, 6.9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.55])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    # Panel letters sit inside their own panels: bigger, and they cost no
    # layout space, which is where most of this figure's white space went.
    inside = dict(dx=0.012, dy=0.975, fontsize=13.0, va="top")

    _fig2_panel_delivered_energy(ax_a, summary, families, label_dx=-9,
                                 xlim=(-7.0, 47.0))
    panel_label(ax_a, "a", **inside)
    _fig2_panel_shortfall(ax_b, shortfall, families, conservative_line=False,
                          ymargin=0.30, ylim=(0.0, 1.23), xlim=(-0.72, 3.5))
    panel_label(ax_b, "b", **inside)
    _fig2_panel_rested_shift(ax_c, summary, families,
                             inset_title="SIB retentions differ by 0.009",
                             inset_rect=[0.085, 0.11, 0.34, 0.185],
                             median_mv=-131.1, median_xy=(0.015, 6.0),
                             median_align=("left", "bottom"),
                             sib_label_xy=(0.29, -93.0),
                             ylim=(-245.0, 75.0), inset_axis_titles=True)
    panel_label(ax_c, "c", **inside)

    fig.tight_layout(w_pad=1.8, h_pad=2.4)
    save_figure(fig, "figure2_energy_decomposition_acs.png")


def write_figure_index() -> None:
    """Map every rendered figure to the document and slot that uses it.

    The main text carries three figures: the former Figures 2 and 3 were
    merged in the sparring-review revision, and the pulse-block maps then moved
    out of Figure 3 into the supplement as S2. The workflow, EIS and context
    figures are used by the Supplementary Information. The trace figure is built by
    scripts/build_decomposition_trace_figure.py, and the graphical abstract by
    scripts/build_graphical_abstract.py, so both are listed with their own
    source rather than this script's.
    """
    path = Path("manuscript/figures/final_figure_index.md")
    rows = [
        ("Manuscript, Figure 1", "figure_decomposition_trace.png",
         "scripts/build_decomposition_trace_figure.py"),
        ("Manuscript, Figure 2", "figure2_energy_decomposition.png",
         "scripts/build_manuscript_figures.py"),
        ("Manuscript, Figure 3", "figure4_raw_pulse_bottleneck.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S1", "figure_s1_arrhenius_activation.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S2", "figure_s2_pulse_block_maps.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S3", "figure_s2_kramers_kronig.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S4", "figure5_eis_bottleneck_decomposition.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S5", "figure6_ocv_entropy_thermal_coupling.png",
         "scripts/build_manuscript_figures.py"),
        ("SI, Figure S6", "figure1_data_architecture.png",
         "scripts/build_manuscript_figures.py"),
        ("Submitted separately", "graphical_abstract.png",
         "scripts/build_graphical_abstract.py"),
        # ACS Energy Letters uses variants of the first two: the stopping line is
        # labelled as inferred, and Figure 2 drops the cross-family load-to-rest
        # panel to enlarge the matched-throughput shift.
        ("ACS Letter, Figure 1", "figure_decomposition_trace_acs.png",
         "scripts/build_decomposition_trace_figure.py --acs"),
        ("ACS Letter, Figure 2", "figure2_energy_decomposition_acs.png",
         "scripts/build_manuscript_figures.py --acs-only"),
        # ACS Energy Letters asks for 3 x 2 in; the Elsevier pane is two and a
        # half times as wide and carries a different palette, so the TOC graphic
        # also branches. graphical_abstract_acs is the submitted composition;
        # build_graphical_abstract.py --acs also writes an _alt variant.
        ("ACS Letter, TOC graphic", "graphical_abstract_acs.png",
         "scripts/build_graphical_abstract.py --acs"),
    ]
    lines = ["# Final Figure Index", "", "| Slot | File | Source script |", "|---|---|---|"]
    lines += [f"| {slot} | `{name}` | `{script}` |" for slot, name, script in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acs-only", action="store_true",
        help="rebuild only the ACS Energy Letters figure variants and leave the "
             "artwork shared with the JES manuscript exactly as it is on disk")
    args = parser.parse_args()

    if args.acs_only:
        figure2_energy_decomposition_acs(
            read_csv("results/tables/table_delivered_energy_decomposition.csv"),
            read_csv("results/tables/table_energy_shortfall_split.csv"),
            read_csv("results/tables/table_common_window_ocv_comparison.csv"),
        )
        print("manuscript/figures/figure2_energy_decomposition_acs.png")
        print("run scripts/build_decomposition_trace_figure.py --acs for the other one")
        return 0

    figure1_workflow()
    figure3_pulse_maps(
        read_csv("results/tables/table_raw_pulse_temperature_summary.csv"),
        read_csv("results/tables/table_raw_pulse_soc_temperature_map.csv"),
        read_csv("results/tables/table_pulse_timepoint_sensitivity.csv"),
    )
    figure_s2_pulse_block_maps(
        read_csv("results/tables/table_raw_pulse_soc_temperature_map.csv"),
    )
    figure4_eis(
        read_csv("results/tables/table_eis_family_summary.csv"),
        read_csv("results/tables/table_eis_hppc_consistency.csv"),
        read_csv("results/tables/table_eis_ecm_fit_reliability_summary.csv"),
    )
    figure5_ocv_entropy_thermal(
        read_csv("results/tables/table_ocv_window_smoothing_sensitivity.csv"),
        read_csv("results/tables/table_ocv_slope_summary.csv"),
        read_csv("results/tables/table_entropy_summary.csv"),
        read_csv("results/tables/table_thermal_response_summary.csv"),
    )
    decomposition_inputs = (
        read_csv("results/tables/table_delivered_energy_decomposition.csv"),
        read_csv("results/tables/table_energy_shortfall_split.csv"),
        read_csv("results/tables/table_common_window_ocv_comparison.csv"),
    )
    figure2_energy_decomposition(*decomposition_inputs)
    figure2_energy_decomposition_acs(*decomposition_inputs)
    write_figure_index()
    for name in [
        "figure1_data_architecture.png",
        "figure2_energy_decomposition.png",
        "figure2_energy_decomposition_acs.png",
        "figure4_raw_pulse_bottleneck.png",
        "figure_s2_pulse_block_maps.png",
        "figure5_eis_bottleneck_decomposition.png",
        "figure6_ocv_entropy_thermal_coupling.png",
    ]:
        print(f"manuscript/figures/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
