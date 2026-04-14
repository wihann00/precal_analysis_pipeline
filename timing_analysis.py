"""
timing_analysis.py
==================
Fits timing offset distributions (PMT-laser, SiPM-laser, monitor-laser)
with an Exponentially Modified Gaussian + optional Chebyshev background.

Plots are generated inside the fit function while the zfit model is still
in scope, so the fitted curve and pull distribution are properly overlaid.
"""

import zfit
from zfit import z
import zfit.z.numpy as znp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging
import plotting

from zfit_models import ExponentiallyModifiedGaussian

logger = logging.getLogger(__name__)


@dataclass
class TimingFitResult:
    """Container for a single timing fit result."""
    coord: Tuple[float, float]
    channel: str

    mu: float
    mu_err: float
    sigma: float
    sigma_err: float
    lambd: float
    lambd_err: float

    transit_time: float
    transit_time_err: float
    tts_fwhm: float

    sig_yield: float
    sig_yield_err: float
    bkg_yield: float
    bkg_yield_err: float
    total_events: int

    bkg_left: float
    bkg_right: float

    converged: bool
    coeff: float

    def to_dict(self) -> dict:
        return {
            "coord": self.coord, "channel": self.channel,
            "mu": self.mu, "mu_err": self.mu_err,
            "sigma": self.sigma, "sigma_err": self.sigma_err,
            "lambd": self.lambd, "lambd_err": self.lambd_err,
            "transit_time": self.transit_time, "transit_time_err": self.transit_time_err,
            "tts_fwhm": self.tts_fwhm,
            "sig_yield": self.sig_yield, "sig_yield_err": self.sig_yield_err,
            "bkg_yield": self.bkg_yield, "bkg_yield_err": self.bkg_yield_err,
            "total_events": self.total_events,
            "bkg_left": self.bkg_left, "bkg_right": self.bkg_right,
            "converged": self.converged, "coeff": self.coeff,
        }


# =============================================================================
# Internal helpers
# =============================================================================

def _compute_fwhm(model, data, size, nbins=50):
    """Compute FWHM of fitted model via spline root-finding."""
    lower, upper = data.data_range.limit1d
    x = np.linspace(lower, upper, 1000)
    y = model.pdf(x).numpy() * size / nbins * data.data_range.area()

    try:
        spline = UnivariateSpline(x, y - np.max(y) / 2, s=0)
        roots = spline.roots()
        if len(roots) >= 2:
            return abs(roots[-1] - roots[0])
    except Exception as e:
        logger.warning(f"FWHM computation failed: {e}")

    half_max = np.max(y) / 2
    above = np.where(y >= half_max)[0]
    if len(above) > 1:
        return x[above[-1]] - x[above[0]]
    return np.nan


# =============================================================================
# Main fit function
# =============================================================================

def fit_timing(coord, data_np, channel="PMT", pmt_serial="", xr=None,
               include_background=True, nbins=50, sideband_multiplier=2,
               make_plot=False, output_dir=".", run_id="",
               fmt="png", dpi=150):
    """
    Fit EMG + optional Chebyshev background to a timing distribution.
    Generates fit plot while model is still in scope if make_plot=True.
    """
    if xr is None:
        xr = [300, 330]

    label = f"{channel}_{coord}"
    size = data_np.shape[0]
    if size < 10:
        logger.warning(f"Too few events ({size}) at {coord} for {channel}")
        return _empty_result(coord, channel, size)

    bkg_left, bkg_right = 0.0, 0.0

    # --- Build model ---------------------------------------------------------
    obs = zfit.Space("x", limits=(xr[0], xr[1]))
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    mu_naive = float(np.mean(data_np))
    sig_yield_naive = size * 0.8
    bkg_yield_naive = size - sig_yield_naive

    mu = zfit.Parameter(f"mu_{label}", mu_naive, mu_naive * 0.8, mu_naive * 1.2)
    lambd = zfit.Parameter(f"lambd_{label}", 0.1, 0.005, 5)
    sigma = zfit.Parameter(f"sigma_{label}", 5, 0.2, 50)

    emg = ExponentiallyModifiedGaussian(obs=obs, mu=mu, lambd=lambd, sigma=sigma)
    emg_yield = zfit.Parameter(f"emg_yield_{label}", sig_yield_naive,
                               sig_yield_naive * 0.01, sig_yield_naive * 1.5,
                               step_size=1)
    emg_ext = emg.create_extended(emg_yield)

    comp_models = [emg_ext]
    comp_names = ["EMG signal"]
    coeff_val = 0.0

    if include_background:
        coeffs = [zfit.Parameter(f"coeff_0_{label}", 0, -2, 1, floating=False)]
        chebyshev = zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)
        bkg_yield_param = zfit.Parameter(f"bkg_yield_{label}", bkg_yield_naive,
                                         bkg_yield_naive * 0.005,
                                         bkg_yield_naive * 3, step_size=1)
        bkg_ext = chebyshev.create_extended(bkg_yield_param)
        model = zfit.pdf.SumPDF([emg_ext, bkg_ext])
        comp_models.append(bkg_ext)
        comp_names.append("Chebyshev bkg")
    else:
        model = emg_ext
        bkg_yield_param = None

    # --- Fit -----------------------------------------------------------------
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)
    logger.info(f"\nParameter Results for {label}:\n{result.params}")

    converged = result.converged
    if not converged:
        logger.warning(f"Fit did not converge for {label}")

    try:
        param_hesse = result.hesse()
    except Exception as e:
        logger.warning(f"Hesse failed for {label}: {e}")
        param_hesse = {}

    # Log correlation matrix
    try:
        corr = result.correlation()
        param_names = [p.name for p in result.params]
        corr_df = pd.DataFrame(corr, index=param_names, columns=param_names)
        logger.info(f"\nCorrelation Matrix for {label}:\n{corr_df}")
    except Exception:
        pass

    # --- Extract results -----------------------------------------------------
    mu_val = float(zfit.run(mu.value()))
    sigma_val = float(zfit.run(sigma.value()))
    lambd_val = float(zfit.run(lambd.value()))

    mu_err = param_hesse.get(mu, {}).get("error", np.nan)
    sigma_err = param_hesse.get(sigma, {}).get("error", np.nan)
    lambd_err = param_hesse.get(lambd, {}).get("error", np.nan)

    if lambd_val != 0:
        transit_time = mu_val + 1.0 / lambd_val
        transit_time_err = np.sqrt(mu_err**2 + (lambd_err / lambd_val**2)**2)
    else:
        transit_time = mu_val
        transit_time_err = mu_err

    tts_fwhm = _compute_fwhm(model, data, size, nbins=nbins)

    sig_yield_val = float(zfit.run(emg_yield.value()))
    sig_yield_err = param_hesse.get(emg_yield, {}).get("error", np.nan)

    if include_background and bkg_yield_param is not None:
        bkg_yield_val = float(zfit.run(bkg_yield_param.value()))
        bkg_yield_err = param_hesse.get(bkg_yield_param, {}).get("error", np.nan)
        coeff_val = float(zfit.run(coeffs[0].value()))
    else:
        bkg_yield_val = 0.0
        bkg_yield_err = 0.0

    # --- Plots (while model is alive!) ---------------------------------------
    if make_plot:
#        fit_params = {
#            "mu": mu_val, "lambd": lambd_val, "sigma": sigma_val,
#            "sig_yield": sig_yield_val, "bkg_yield": bkg_yield_val,
#        }

        fitplotter = plotting.FitPlotter(coord, channel, pmt_serial, output_dir, run_id, nbins, fmt, dpi)

        fit_params = {
                r"\mu": f"{mu_val:.2f}", 
                r"\lambda": f"{lambd_val:.2f}", 
                r"\sigma": f"{sigma_val:.2f}",
                "sig yield": f"{sig_yield_val:.0f}", 
                "bkg yield": f"{bkg_yield_val:.0f}",
                "FWHM": f"{tts_fwhm:.2f}",
        }
        fitplotter.plot_fit_and_pull(model, data, size, include_background,
                                    comp_names, fit_params, xr,
                                    x_label="Time (ns)", fit_type="timing")

        fitplotter.plot_fwhm(model, data, size, tts_fwhm)

    return TimingFitResult(
        coord=coord, channel=channel,
        mu=mu_val, mu_err=mu_err,
        sigma=sigma_val, sigma_err=sigma_err,
        lambd=lambd_val, lambd_err=lambd_err,
        transit_time=transit_time, transit_time_err=transit_time_err,
        tts_fwhm=tts_fwhm,
        sig_yield=sig_yield_val, sig_yield_err=sig_yield_err,
        bkg_yield=bkg_yield_val, bkg_yield_err=bkg_yield_err,
        total_events=size,
        bkg_left=bkg_left, bkg_right=bkg_right,
        converged=converged, coeff=coeff_val,
    )


def _empty_result(coord, channel, size):
    return TimingFitResult(
        coord=coord, channel=channel,
        mu=np.nan, mu_err=np.nan,
        sigma=np.nan, sigma_err=np.nan,
        lambd=np.nan, lambd_err=np.nan,
        transit_time=np.nan, transit_time_err=np.nan,
        tts_fwhm=np.nan,
        sig_yield=np.nan, sig_yield_err=np.nan,
        bkg_yield=np.nan, bkg_yield_err=np.nan,
        total_events=size,
        bkg_left=0, bkg_right=0,
        converged=False, coeff=np.nan,
    )


# =============================================================================
# Data extraction
# =============================================================================

def extract_timing_data(df, delta_column, xr, require_pulse=None, charge_cut=None):
    """Extract timing offset data, handling scalar and LEDTimes (vector) columns."""
    mask = pd.Series(True, index=df.index)
    if require_pulse and require_pulse in df.columns:
        mask &= df[require_pulse] > 0

    if charge_cut is not None:
        col, threshold = charge_cut  # e.g. ("sipm_PulseCharge", 790)
        if col in df.columns:
            mask &= df[col] < threshold

    if delta_column not in df.columns:
        logger.warning(f"Column {delta_column} not found in DataFrame")
        return np.array([])

    filtered = df.loc[mask, delta_column]

    if delta_column.endswith("_LED"):
        arrays = filtered.dropna().values
        if len(arrays) == 0:
            return np.array([])
        all_vals = np.concatenate([np.atleast_1d(a) for a in arrays])
    else:
        all_vals = filtered.dropna().values.astype(float)

    return all_vals[(all_vals >= xr[0]) & (all_vals <= xr[1])]


# =============================================================================
# Batch runner
# =============================================================================

def run_timing_analysis(scan_data, coords, config):
    """
    Run timing fits for all scan points for PMT, SiPM, and optionally monitor.
    Plots are generated during the fit (not after) so the model curve is available.
    """
    timing_cfg = config["timing"]
    pmt_cfg = timing_cfg["pmt"]
    sipm_cfg = timing_cfg["sipm"]
    sideband_mult = timing_cfg.get("sideband_multiplier", 2)
    plot_cfg = config.get("plotting", {})
    save_plots = plot_cfg.get("save_fit_plots", False)
    output_dir = config.get("output_dir", ".")
    run_id = config.get("run_id", "")
    fmt = plot_cfg.get("figure_format", "png")
    dpi = plot_cfg.get("dpi", 150)
    pmt_serial = config.get("pmt_serial", "")

    pmt_results = []
    sipm_results = []
    mon_results = []

    for coord in coords:
        if coord not in scan_data:
            logger.warning(f"No data for {coord}, skipping")
            continue

        df = scan_data[coord]
        theta, phi = coord
        logger.info(f"Fitting ({theta}, {phi})...")

        # --- PMT timing fit ---
        pmt_data = extract_timing_data(
            df, "delta_PMT_laser_LED", pmt_cfg["fit_range"],
            require_pulse="PMT_PulseStart"
        )
        if len(pmt_data) > 0:
            pmt_result = fit_timing(
                coord, pmt_data, channel="PMT", pmt_serial=pmt_serial,
                xr=pmt_cfg["fit_range"],
                include_background=pmt_cfg["include_background"],
                nbins=pmt_cfg["nbins"],
                sideband_multiplier=sideband_mult,
                make_plot=save_plots,
                output_dir=output_dir, run_id=run_id, fmt=fmt, dpi=dpi,
            )
            pmt_results.append(pmt_result.to_dict())
        else:
            logger.warning(f"No PMT timing data for {coord}")

        # --- SiPM timing fit ---
        sipm_data = extract_timing_data(
            df, "delta_sipm_laser_LED", sipm_cfg["fit_range"],
            require_pulse="sipm_PulseStart",
            charge_cut=("sipm_PulseCharge", sipm_cfg["charge_cut"]),
        )
        if len(sipm_data) > 0:
            sipm_result = fit_timing(
                coord, sipm_data, channel="sipm", pmt_serial=pmt_serial,
                xr=sipm_cfg["fit_range"],
                include_background=sipm_cfg["include_background"],
                nbins=sipm_cfg["nbins"],
                sideband_multiplier=sideband_mult,
                make_plot=save_plots,
                output_dir=output_dir, run_id=run_id, fmt=fmt, dpi=dpi,
            )
            sipm_results.append(sipm_result.to_dict())
        else:
            logger.warning(f"No SiPM timing data for {coord}")

        # --- Monitor (if enabled) ---
        if config["monitor"]["enabled"]:
            mon_xr = config["monitor"]["fit_range"]
            mon_data = extract_timing_data(
                df, "delta_mon_laser", mon_xr,
                require_pulse="mon_PulseStart"
            )
            if len(mon_data) > 0:
                try:
                    mon_result = fit_timing(
                        coord, mon_data, channel="mon", pmt_serial=pmt_serial,
                        xr=mon_xr, include_background=True, nbins=50,
                        sideband_multiplier=sideband_mult,
                        make_plot=save_plots,
                        output_dir=output_dir, run_id=run_id, fmt=fmt, dpi=dpi,
                    )
                    mon_results.append(mon_result.to_dict())
                except Exception as e:
                    logger.warning(f"Monitor fit failed for {coord}: {e}")
                    mon_results.append(_empty_result(coord, "mon", len(mon_data)).to_dict())
            else:
                # NaN placeholder to keep aligned with PMT/SiPM
                mon_results.append(_empty_result(coord, "mon", 0).to_dict())

    pmt_df = pd.DataFrame(pmt_results) if pmt_results else pd.DataFrame()
    sipm_df = pd.DataFrame(sipm_results) if sipm_results else pd.DataFrame()
    mon_df = pd.DataFrame(mon_results) if mon_results else None

    return pmt_df, sipm_df, mon_df
