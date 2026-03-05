"""
plotting.py
===========
Generates all analysis plots:
  - Individual timing fit plots with pull distributions
  - Cross-section plots (x and y axes through PMT)
  - 2D heatmaps of scan parameters
  - Monitor stability plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

from scan_geometry import ScanGeometry

logger = logging.getLogger(__name__)

# Colour palette
COLOURS = {
    "pmt_x": "red",
    "pmt_y": "blue",
    "sipm_x": "lightpink",
    "sipm_y": "deepskyblue",
    "norm_x": "darkred",
    "norm_y": "darkblue",
    "monitor": "darkorange",
}


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Timing fit plots (individual per scan point)
# =============================================================================

def plot_timing_fit(coord: Tuple[float, float],
                    data_np: np.ndarray,
                    fit_result: dict,
                    model_pdf_func=None,
                    channel: str = "PMT",
                    xr: List[float] = None,
                    nbins: int = 50,
                    output_dir: str = ".",
                    run_id: str = "",
                    fmt: str = "png",
                    dpi: int = 150):
    """
    Plot a timing distribution fit with pull distribution.

    Note: Since zfit models don't survive pickling, this function accepts
    raw data and fit parameters to reconstruct a simple overlay.
    For full model plots, call this during the fit loop.
    """
    if xr is None:
        xr = [300, 330]

    theta, phi = coord

    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [4, 1]},
                                    figsize=(10, 10))

    # Histogram
    counts, bin_edges = np.histogram(data_np, bins=nbins, range=xr)
    bin_centres = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    ax1.errorbar(bin_centres, counts, yerr=np.sqrt(np.maximum(counts, 1)),
                 fmt="ok", label="data")

    # Text box with fit parameters
    textstr = "\n".join([
        rf"$\mu = {fit_result.get('mu', 0):.2f}$",
        rf"$\lambda = {fit_result.get('lambd', 0):.2f}$",
        rf"$\sigma = {fit_result.get('sigma', 0):.2f}$",
        f"sig yield = {fit_result.get('sig_yield', 0):.0f}",
        f"bkg yield = {fit_result.get('bkg_yield', 0):.0f}",
        f"FWHM = {fit_result.get('tts_fwhm', 0):.2f}",
    ])
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    ax1.text(0.70, 0.95, textstr, transform=ax1.transAxes, fontsize=16,
             verticalalignment="top", bbox=props)

    ax1.set_ylabel("Events", fontsize=16)
    ax1.set_xlim(xr)
    ax1.set_title(f"{channel} ({theta}, {phi})", fontsize=16)

    # Pull (placeholder — zeros if no model curve available)
    ax2.axhline(0, ls="--", color="black")
    ax2.set_ylabel("Pull", fontsize=12)
    ax2.set_xlim(xr)
    ax2.set_ylim([-5, 5])
    ax2.set_yticks([-5, 0, 5])
    ax2.set_xlabel("Time (ns)", fontsize=16)

    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=14)

    fig.tight_layout()
    outpath = Path(output_dir) / "figures" / "timing_fits" / run_id
    _ensure_dir(outpath)
    fig.savefig(outpath / f"{channel}_theta{theta}_phi{phi}.{fmt}", dpi=dpi)
    plt.close(fig)


# =============================================================================
# Cross-section plots
# =============================================================================

def plot_cross_sections(summary_df: pd.DataFrame,
                        geometry: ScanGeometry,
                        output_dir: str = ".",
                        run_id: str = "",
                        fmt: str = "png",
                        dpi: int = 150):
    """
    Generate cross-section plots along x and y axes through the PMT centre.

    Plots: relative PMT yield, relative SiPM yield, corrected efficiency,
           relative transit time, TTS.
    """
    x_move, y_move, angle_axis = geometry.cross_section_indices()
    outpath = Path(output_dir) / "figures" / "cross_sections" / run_id
    _ensure_dir(outpath)

    # Helper to safely index
    def _get(col, indices):
        vals = summary_df[col].values
        return vals[indices]

    def _get_safe(col, indices):
        if col in summary_df.columns:
            return summary_df[col].values[indices]
        return np.full(len(indices), np.nan)

    # ---- Plot 1: Relative signal yields (PMT, SiPM, corrected) ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.errorbar(angle_axis, _get("rel_pmt_yield", x_move),
                yerr=_get("rel_pmt_yield_err", x_move),
                label="PMT: x-axis", color=COLOURS["pmt_x"],
                marker="o", markersize=8, ls="none")
    ax.errorbar(angle_axis, _get("rel_pmt_yield", y_move),
                yerr=_get("rel_pmt_yield_err", y_move),
                label="PMT: y-axis", color=COLOURS["pmt_y"],
                marker="o", markersize=8, ls="none")
    ax.errorbar(angle_axis, _get("rel_sipm_yield", x_move),
                yerr=_get("rel_sipm_yield_err", x_move),
                label="SiPM: x-axis", color=COLOURS["sipm_x"],
                marker="s", markersize=8, ls="none")
    ax.errorbar(angle_axis, _get("rel_sipm_yield", y_move),
                yerr=_get("rel_sipm_yield_err", y_move),
                label="SiPM: y-axis", color=COLOURS["sipm_y"],
                marker="s", markersize=8, ls="none")

    ax.axhline(1, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel("Zenith angle (deg)", fontsize=15)
    ax.set_ylabel("Relative signal yield", fontsize=15)
    ax.set_xticks(angle_axis)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath / f"rel_yields.{fmt}", dpi=dpi)
    plt.close(fig)

    # ---- Plot 2: Corrected relative detection efficiency ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.errorbar(angle_axis, _get("corrected_rel_efficiency", x_move),
                yerr=_get("corrected_rel_efficiency_err", x_move),
                label="Corrected ε: x-axis", color=COLOURS["norm_x"],
                marker="x", markersize=10, ls="none")
    ax.errorbar(angle_axis, _get("corrected_rel_efficiency", y_move),
                yerr=_get("corrected_rel_efficiency_err", y_move),
                label="Corrected ε: y-axis", color=COLOURS["norm_y"],
                marker="x", markersize=10, ls="none")

    ax.axhline(1, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel("Zenith angle (deg)", fontsize=15)
    ax.set_ylabel("Corrected relative detection efficiency", fontsize=15)
    ax.set_xticks(angle_axis)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath / f"corrected_efficiency.{fmt}", dpi=dpi)
    plt.close(fig)

    # ---- Plot 3: Transit time offset ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.errorbar(angle_axis, _get("rel_transit_time", x_move),
                yerr=_get("rel_transit_time_err", x_move),
                label="x-axis", color=COLOURS["norm_x"],
                marker="x", markersize=10, ls="none")
    ax.errorbar(angle_axis, _get("rel_transit_time", y_move),
                yerr=_get("rel_transit_time_err", y_move),
                label="y-axis", color=COLOURS["norm_y"],
                marker="x", markersize=10, ls="none")

    ax.axhline(0, ls="--", color="grey", alpha=0.5)
    ax.set_xlabel("Zenith angle (deg)", fontsize=15)
    ax.set_ylabel("Transit time offset from centre (ns)", fontsize=15)
    ax.set_xticks(angle_axis)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath / f"transit_time.{fmt}", dpi=dpi)
    plt.close(fig)

    # ---- Plot 4: TTS (FWHM) ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.scatter(angle_axis, _get("pmt_tts_fwhm", x_move),
               label="x-axis", color=COLOURS["norm_x"], marker="x", s=100)
    ax.scatter(angle_axis, _get("pmt_tts_fwhm", y_move),
               label="y-axis", color=COLOURS["norm_y"], marker="x", s=100)

    ax.set_xlabel("Zenith angle (deg)", fontsize=15)
    ax.set_ylabel("TTS FWHM (ns)", fontsize=15)
    ax.set_xticks(angle_axis)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath / f"tts_fwhm.{fmt}", dpi=dpi)
    plt.close(fig)

    # ---- Plot 5: Monitor stability (if available) ----
    if "rel_mon_yield" in summary_df.columns:
        fig, ax = plt.subplots(1, 1, figsize=(18, 7))

        labels = summary_df["coord"].apply(str).values
        x_vals = np.arange(len(labels))
        rel_mon = summary_df["rel_mon_yield"].values
        rel_mon_err = summary_df["rel_mon_yield_err"].values

        ax.errorbar(x_vals, rel_mon, yerr=rel_mon_err,
                     label="monitor−laser", marker="o", markersize=8,
                     color=COLOURS["monitor"], ls="none")
        ax.axhline(1, ls="--", color="grey", alpha=0.5)
        ax.set_ylabel("Relative monitor yield", fontsize=15)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.3)
        _style_axis(ax)
        fig.tight_layout()
        fig.savefig(outpath / f"monitor_stability.{fmt}", dpi=dpi)
        plt.close(fig)


# =============================================================================
# 2D Heatmaps
# =============================================================================

def plot_heatmaps(summary_df: pd.DataFrame,
                  geometry: ScanGeometry,
                  output_dir: str = ".",
                  run_id: str = "",
                  fmt: str = "png",
                  dpi: int = 150):
    """
    Generate 2D heatmaps of key parameters over the scan grid.
    """
    zenith_vals, azimuth_vals, grid_index = geometry.heatmap_grid()
    outpath = Path(output_dir) / "figures" / "heatmaps" / run_id
    _ensure_dir(outpath)

    quantities = {
        "corrected_rel_efficiency": ("Corrected Relative Detection Efficiency", "RdYlGn"),
        "rel_pmt_yield": ("Relative PMT Yield", "viridis"),
        "pmt_tts_fwhm": ("TTS FWHM (ns)", "plasma"),
        "rel_transit_time": ("Transit Time Offset (ns)", "coolwarm"),
    }

    if "rel_gain" in summary_df.columns:
        quantities["rel_gain"] = ("Relative Gain", "viridis")

    for col, (title, cmap) in quantities.items():
        if col not in summary_df.columns:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw={"projection": "polar"})

        # Convert to polar coordinates for the heatmap
        values = summary_df[col].values
        coords = summary_df["coord"].values

        # Plot as scatter on polar axes
        thetas_rad = []
        phis_rad = []
        vals = []

        for i, c in enumerate(coords):
            theta, phi = c
            thetas_rad.append(np.radians(theta))  # zenith as radial
            # Swap: zenith = radial distance, azimuth = angular position
            phis_rad.append(np.radians(phi))
            vals.append(values[i])

        # For the centre point (theta=0), it should appear at radius 0
        scatter = ax.scatter(phis_rad, thetas_rad, c=vals, cmap=cmap,
                              s=300, edgecolors="black", linewidth=0.5, zorder=5)

        ax.set_title(f"{title}\n{run_id}", fontsize=14, pad=20)
        ax.set_rticks(list(geometry.zeniths))
        ax.set_rlabel_position(45)
        ax.set_thetagrids(list(geometry.azimuths))

        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label(title, fontsize=12)

        fig.tight_layout()
        fig.savefig(outpath / f"heatmap_{col}.{fmt}", dpi=dpi)
        plt.close(fig)

    # --- Also produce a simple rectangular grid heatmap ----
    for col, (title, cmap) in quantities.items():
        if col not in summary_df.columns:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        values = summary_df[col].values
        coords_list = list(summary_df["coord"].values)

        # Build grid: rows = zenith, cols = azimuth
        grid = np.full((len(zenith_vals), len(azimuth_vals)), np.nan)
        for i, c in enumerate(coords_list):
            theta, phi = c
            if theta == 0:
                # Centre point: fill all azimuths
                for j in range(len(azimuth_vals)):
                    if np.isnan(grid[0, j]):
                        grid[0, j] = values[i]
            else:
                row = np.where(zenith_vals == theta)[0]
                col_idx = np.where(azimuth_vals == phi)[0]
                if len(row) > 0 and len(col_idx) > 0:
                    grid[row[0], col_idx[0]] = values[i]

        im = ax.imshow(grid, cmap=cmap, aspect="auto",
                        extent=[-0.5, len(azimuth_vals) - 0.5,
                                len(zenith_vals) - 0.5, -0.5])
        ax.set_xticks(range(len(azimuth_vals)))
        ax.set_xticklabels([f"{a}°" for a in azimuth_vals])
        ax.set_yticks(range(len(zenith_vals)))
        ax.set_yticklabels([f"{z}°" for z in zenith_vals])
        ax.set_xlabel("Azimuth", fontsize=14)
        ax.set_ylabel("Zenith", fontsize=14)
        ax.set_title(f"{title} — {run_id}", fontsize=14)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(title, fontsize=12)

        # Annotate cells
        for row in range(len(zenith_vals)):
            for col_j in range(len(azimuth_vals)):
                val = grid[row, col_j]
                if not np.isnan(val):
                    ax.text(col_j, row, f"{val:.3f}", ha="center", va="center",
                            fontsize=10, color="white" if val < np.nanmean(grid) else "black")

        fig.tight_layout()
        fig.savefig(outpath / f"grid_{col}.{fmt}", dpi=dpi)
        plt.close(fig)


# =============================================================================
# Summary parameter plots (like your existing 2x2 parameter grid)
# =============================================================================

def plot_parameter_summary(summary_df: pd.DataFrame,
                           output_dir: str = ".",
                           run_id: str = "",
                           fmt: str = "png",
                           dpi: int = 150):
    """Plot EMG fit parameters across all scan points."""
    outpath = Path(output_dir) / "figures" / "summary" / run_id
    _ensure_dir(outpath)

    coords = [str(c) for c in summary_df["coord"].values]
    x = np.arange(len(coords))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

    ax1.errorbar(x, summary_df["pmt_mu"], yerr=summary_df["pmt_mu_err"],
                  fmt="o", ls="none")
    ax1.set_ylabel("μ (ns)", fontsize=13)

    ax2.errorbar(x, summary_df["pmt_sigma"],
                  yerr=summary_df.get("pmt_sigma_err", np.nan), fmt="o", ls="none")
    ax2.set_ylabel("σ (ns)", fontsize=13)

    ax3.errorbar(x, summary_df["pmt_lambd"],
                  yerr=summary_df.get("pmt_lambd_err", np.nan), fmt="o", ls="none")
    ax3.set_ylabel("λ", fontsize=13)

    ax4.scatter(x, summary_df["pmt_tts_fwhm"])
    ax4.set_ylabel("TTS FWHM (ns)", fontsize=13)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xticks(x)
        ax.set_xticklabels(coords, rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"EMG Fit Parameters — {run_id}", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath / f"fit_parameters.{fmt}", dpi=dpi)
    plt.close(fig)


def _style_axis(ax):
    """Apply consistent font sizing."""
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
