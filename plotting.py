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
import zfit
import matplotlib as mpl

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
    "monitor": "blueviolet",
}


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# Timing fit plots (individual per scan point)
# =============================================================================

class FitPlotter:
    def __init__(self, coord, channel, pmt_serial, output_dir, run_id, nbins, fmt, dpi):
        self.coord = coord
        self.channel = channel
        self.pmt_serial = pmt_serial
        self.output_dir = output_dir
        self.run_id = run_id
        self.nbins = nbins
        self.fmt = fmt
        self.dpi = dpi

    def hist_data(self, data, nbins=50):

        lower, upper = data.data_range.limit1d
        data_np = zfit.run(data.value()[:, 0])

        counts, bin_edges = np.histogram(data_np, nbins, range=(lower, upper))
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

        return counts, bin_centers

    def plot_model(self, model, data, size, ax, scale=1, model_name=None, plot_data=True, comp=False, nbins=50):  # we will use scale later on

        lower, upper = data.data_range.limit1d

        x = np.linspace(lower, upper, num=1000)  # np.linspace also works
        y = model.pdf(x) * size / nbins * data.data_range.area()
        y *= scale
        if comp==True:
            ax.plot(x, y, '--', label=model_name)
        else:
            ax.plot(x, y, linewidth=3)

        if plot_data:

            counts, bin_centers = self.hist_data(data, nbins=nbins)

            ax.errorbar(bin_centers, counts, yerr = np.sqrt(counts), fmt='ok', label='data')

    def plot_comp_model(self, model, data, size, ax, model_names=None, nbins=50, extra_frac=1):
        # print(model.models)
        for i, (mod, frac) in enumerate(zip(model.pdfs, model.params.values())):
            # print(str(mod))
            model_name = model_names[i]
            if str(mod.name) == 'SumPDF_ext':
                self.plot_comp_model(mod, data, size, ax, model_names=model_name, extra_frac=frac, nbins=nbins)
                continue
            # print(str(frac))
            self.plot_model(mod, data, size, ax, scale=frac*extra_frac, model_name=model_name, plot_data=False, comp=True, nbins=nbins)

    def plot_fit_and_pull(self, model, comp_models, model_names,
                        data, data_np, inc_bkg, size, xr,
                        fit_params, fwhm_val):
        """
        Generate fit plot with:
        - Data histogram with Poisson errors
        - Total model curve (solid)
        - Individual component curves (dashed)
        - Pull distribution
        - Parameter text box

        Called from inside fit_timing() while the model is alive.
        """
        coord = self.coord
        theta, phi = coord
        nbins = self.nbins
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(10, 10))

        if inc_bkg:
            # self.plot_comp_model(sig_ext, data, size, ax1)
            self.plot_comp_model(model, data, size, ax1, model_names=model_names, nbins=nbins)
        else:
            self.plot_model(model, data, size, ax1, model_names=model_names, nbins=nbins)
            # self.plot_comp_model(model, data, size, ax1)
        
        self.plot_model(model, data, size, ax1, nbins=nbins)

        # Main axis
        ax1.set_ylabel(f'Events', loc='top', fontsize=12)
        ax1.set_xlim([xr[0], xr[1]])
        ax1.set_title(f"{self.channel} ({theta}, {phi})", fontsize=14)

        textstr = '\n'.join((
        r'$\mu=%.2f$' % (fit_params["mu"], ),
        r'$\lambda=%.2f$' % (fit_params["lambd"], ),
        r'$\sigma=%.2f$' % (fit_params["sigma"], ),
        r'sig yield$=%.0f$' % (fit_params["sig_yield"], ),
        r'bkg yield$=%.0f$' % (fit_params["bkg_yield"], ),
        # r'bkg left$=%.2f$' % (bkg_in_left_window, ),
        # r'bkg right$=%.2f$' % (bkg_in_right_window, ),
        'FHWM=%.2f' % (fwhm_val, ))) #FWHM

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

        # place a text box in upper left in axes coords
        ax1.text(0.7, 0.95, textstr, transform=ax1.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

        # Pull axis
        counts, bin_centers = self.hist_data(data)
        y_obs = counts
        y_exp = model.pdf(bin_centers) * size / nbins * data.data_range.area()

        resid = y_obs - y_exp
        pull = resid/np.sqrt(counts)

        ax2.plot(np.linspace(xr[0], xr[1], 500), np.zeros(500), '--', color='black')
        ax2.errorbar(bin_centers, pull, fmt='ok')
        ax2.set_ylabel(f'Pull', fontsize=12)
        ax2.set_xlim([xr[0], xr[1]])
        # if (max(pull) < 5) and  (min(pull) > -5):
        #     ax2.set_ylim([-5, 5])
        #     ax2.set_yticks([-5, 0, 5])
        ax2.set_ylim([-5, 5])
        ax2.set_yticks([-5, 0, 5])

        plt.xlabel('Time (ns)', fontsize=20)

        for ax in [ax1, ax2]:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + 
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)

        fig.tight_layout()
        outpath = Path(self.output_dir) / "figures" / "timing_fits" / f"{self.run_id}-{self.pmt_serial}" / self.channel
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"{self.channel}_theta{theta}_phi{phi}.{self.fmt}", dpi=self.dpi)
        plt.close(fig)


    def plot_fwhm(self, model, data, size, fwhm_val):
        """Generate standalone FWHM visualisation plot."""
        coord = self.coord
        theta, phi = coord
        nbins = self.nbins
        lower, upper = data.data_range.limit1d
        x = np.linspace(lower, upper, 1000)
        y = model.pdf(x).numpy() * size / nbins * data.data_range.area()

        colours = ['#173F5F', '#20639B', '#3CAEA3', '#F6D55C', '#ED553B']

        try:
            spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
            roots = spline.roots()
            if len(roots) < 2:
                return
            r1, r2 = roots[0], roots[-1]
        except Exception:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(x, y, color=colours[0])
        ax.axvspan(r1, r2, facecolor=colours[4], alpha=0.75,
                label=f"r1={r1:.2f}; r2={r2:.2f}\nFWHM={fwhm_val:.2f}")
        ax.legend(fontsize=14)
        ax.set_ylim(0, np.max(y) * 1.1)
        ax.set_title(f"{self.channel} ({theta}, {phi})", fontsize=14)
        ax.set_xlabel("Time (ns)", fontsize=14)
        fig.tight_layout()

        outpath = Path(self.output_dir) / "figures" / "FWHM" / f"{self.run_id}-{self.pmt_serial}-" / self.channel
        outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / f"{self.channel}_theta{theta}_phi{phi}_FWHM.{self.fmt}", dpi=self.dpi)
        plt.close(fig)


# =============================================================================
# Cross-section plots
# =============================================================================

def plot_cross_sections(summary_df: pd.DataFrame,
                        geometry: ScanGeometry,
                        pmt_serial: str = "",
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
    outpath = Path(output_dir) / "figures" / "cross_sections" / f"{run_id}-{pmt_serial}"
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

    ax.plot([], [], ' ', label=f'PMT: {pmt_serial}') # Invisible plot for PMT serial
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

    ax.plot([], [], ' ', label=f'PMT: {pmt_serial}') # Invisible plot for PMT serial
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

    # ---- Plot 3: pmt occupancy ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.plot([], [], ' ', label=f'PMT: {pmt_serial}') # Invisible plot for PMT serial
    ax.errorbar(angle_axis, _get("pmt_occupancy", x_move),
                yerr=_get("pmt_occupancy_err", x_move),
                label="rel. DE: x-axis", color=COLOURS["pmt_x"],
                marker="o", markersize=10, ls="none")
    ax.errorbar(angle_axis, _get("pmt_occupancy", y_move),
                yerr=_get("pmt_occupancy_err", y_move),
                label="rel. DE: y-axis", color=COLOURS["pmt_y"],
                marker="o", markersize=10, ls="none")

    ax.set_xlabel("Zenith angle (deg)", fontsize=15)
    ax.set_ylabel("Detection efficiency", fontsize=15)
    ax.set_xticks(angle_axis)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(outpath / f"pmt_occupancy.{fmt}", dpi=dpi)
    plt.close(fig)

    # ---- Plot 4: Transit time offset ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.plot([], [], ' ', label=f'PMT: {pmt_serial}') # Invisible plot for PMT serial
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

    # ---- Plot 5: TTS (FWHM) ----
    fig, ax = plt.subplots(1, 1, figsize=(18, 7))

    ax.plot([], [], ' ', label=f'PMT: {pmt_serial}') # Invisible plot for PMT serial
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

    # ---- Plot 6: Monitor stability (if available) ----
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
                  pmt_serial: str = "",
                  output_dir: str = ".",
                  run_id: str = "",
                  fmt: str = "png",
                  dpi: int = 150):
    """
    Generate 2D heatmaps of key parameters over the scan grid.
    """
    zenith_vals, azimuth_vals, grid_index = geometry.heatmap_grid()
    outpath = Path(output_dir) / "figures" / "heatmaps" / f"{run_id}-{pmt_serial}"
    _ensure_dir(outpath)

    quantities = {
        "corrected_rel_efficiency": ("Corrected Relative Detection Efficiency", "plasma"),
        "rel_pmt_yield": ("Relative PMT Yield", "plasma"),
        "pmt_tts_fwhm": ("TTS FWHM (ns)", "plasma"),
        "rel_transit_time": ("Transit Time Offset (ns)", "coolwarm"),
    }

    if "rel_gain" in summary_df.columns:
        quantities["rel_gain"] = ("Relative Gain", "viridis")

    for col, (title, colormap) in quantities.items():
        if col not in summary_df.columns:
            continue

        values = summary_df[col].values
        coords_list = list(summary_df["coord"].values)

        # Build grid: rows = azimuth, cols = zenith
        n_azimuths = len(azimuth_vals)
        n_zeniths = len(zenith_vals)
        z_grid = np.full((n_azimuths, n_zeniths), np.nan)

        for i, c in enumerate(coords_list):
            theta, phi = c
            zenith_idx = np.where(zenith_vals == theta)[0][0]
            azimuth_idx = np.where(azimuth_vals == phi)[0][0]
            z_grid[azimuth_idx, zenith_idx] = values[i]

        # Fill centre (zenith=0) for all azimuths with the same value
        center_value = z_grid[0, 0]
        z_grid[:, 0] = center_value

        # Create polar wedge plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax.set_theta_zero_location('S')
        ax.set_theta_direction(-1)

        cmap = mpl.colormaps[colormap]
        if col=="corrected_rel_efficiency" or col=="rel_pmt_yield":
            norm = mpl.colors.Normalize(vmin=0.2, vmax=1.5)
        else:
            norm = mpl.colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))

        bin_width = 5  # degrees half-width on each side

        for i_azimuth in range(n_azimuths):
            for i_zenith in range(n_zeniths):
                value = z_grid[i_azimuth, i_zenith]
                if np.isnan(value):
                    continue

                theta_center = np.deg2rad(azimuth_vals[i_azimuth])
                theta_width = np.deg2rad(360 / n_azimuths)

                zenith = zenith_vals[i_zenith]
                if i_zenith == 0:
                    r_inner = 0
                    r_outer = bin_width
                else:
                    r_inner = max(0, zenith - bin_width)
                    r_outer = zenith + bin_width

                r_height = r_outer - r_inner
                color = cmap(norm(value))

                ax.bar(theta_center, r_height, width=theta_width,
                       bottom=r_inner, color=color, edgecolor='grey', linewidth=0.5)

        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=colormap),
                     ax=ax, orientation='vertical', label=title, pad=0.1)

        ax.set_ylim(0, max(zenith_vals) + bin_width)
        ax.set_yticks(list(zenith_vals))
        ax.set_yticklabels([f'{int(z)}°' for z in zenith_vals])
        ax.set_xticks(np.deg2rad(list(azimuth_vals)))
        ax.set_xticklabels([f'{int(a)}°' for a in azimuth_vals])
        ax.set_title(f"{title}\n{run_id}", fontsize=14, pad=20)

        # fig.tight_layout()
        fig.savefig(outpath / f"heatmap_{col}.{fmt}", dpi=dpi, bbox_inches='tight')
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
                           pmt_serial: str="",
                           output_dir: str = ".",
                           run_id: str = "",
                           fmt: str = "png",
                           dpi: int = 150):
    """Plot EMG fit parameters across all scan points."""
    outpath = Path(output_dir) / "figures" / "summary" / f"{run_id}-{pmt_serial}"
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