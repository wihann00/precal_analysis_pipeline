"""
timing_analysis.py
==================
Fits timing offset distributions (PMT−laser, SiPM−laser, monitor−laser)
with an Exponentially Modified Gaussian + optional Chebyshev background.

Extracts: timing offset (mu), TTS (FWHM), signal/background yields,
and all associated uncertainties.
"""

import zfit
from zfit import z
import zfit.z.numpy as znp
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging

from zfit_models import ExponentiallyModifiedGaussian

logger = logging.getLogger(__name__)


@dataclass
class TimingFitResult:
    """Container for a single timing fit result."""
    coord: Tuple[float, float]
    channel: str                # "PMT", "sipm", "mon"

    # Fit parameters
    mu: float
    mu_err: float
    sigma: float
    sigma_err: float
    lambd: float
    lambd_err: float

    # Derived quantities
    transit_time: float         # mu + 1/lambda (sample mean of EMG)
    transit_time_err: float
    tts_fwhm: float             # FWHM of the fitted model

    # Yields
    sig_yield: float
    sig_yield_err: float
    bkg_yield: float
    bkg_yield_err: float
    total_events: int

    # Background sideband counts
    bkg_left: float
    bkg_right: float

    # Fit quality
    converged: bool
    coeff: float                # Chebyshev coefficient (if background included)

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "coord": self.coord,
            "channel": self.channel,
            "mu": self.mu,
            "mu_err": self.mu_err,
            "sigma": self.sigma,
            "sigma_err": self.sigma_err,
            "lambd": self.lambd,
            "lambd_err": self.lambd_err,
            "transit_time": self.transit_time,
            "transit_time_err": self.transit_time_err,
            "tts_fwhm": self.tts_fwhm,
            "sig_yield": self.sig_yield,
            "sig_yield_err": self.sig_yield_err,
            "bkg_yield": self.bkg_yield,
            "bkg_yield_err": self.bkg_yield_err,
            "total_events": self.total_events,
            "bkg_left": self.bkg_left,
            "bkg_right": self.bkg_right,
            "converged": self.converged,
            "coeff": self.coeff,
        }


def _compute_fwhm(model, data, size: int, nbins: int = 50) -> float:
    """Compute FWHM of a fitted model using spline root-finding."""
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

    # Fallback: estimate from half-max points directly
    half_max = np.max(y) / 2
    above = np.where(y >= half_max)[0]
    if len(above) > 1:
        return x[above[-1]] - x[above[0]]
    return np.nan


def _count_sideband(data_np: np.ndarray, xr: List[float],
                    sideband_multiplier: float = 2) -> Tuple[float, float]:
    """Count events in left and right sidebands for background estimation."""
    width = xr[1] - xr[0]
    xr_l = (xr[0] - sideband_multiplier * width, xr[0] - width)
    xr_r = (xr[1] + width, xr[1] + sideband_multiplier * width)

    n_left = np.sum((data_np >= xr_l[0]) & (data_np < xr_l[1]))
    n_right = np.sum((data_np >= xr_r[0]) & (data_np < xr_r[1]))
    return float(n_left), float(n_right)


def fit_timing(coord: Tuple[float, float],
               data_np: np.ndarray,
               channel: str = "PMT",
               xr: List[float] = None,
               include_background: bool = True,
               nbins: int = 50,
               sideband_multiplier: float = 2) -> TimingFitResult:
    """
    Fit an EMG + optional Chebyshev background to a timing distribution.

    Parameters
    ----------
    coord : tuple
        (theta, phi) scan coordinate.
    data_np : ndarray
        Timing offset values (already filtered to the fit range).
    channel : str
        Channel identifier ("PMT", "sipm", "mon").
    xr : list
        [lower, upper] fit range in ns.
    include_background : bool
        Whether to include a Chebyshev background component.
    nbins : int
        Number of bins for plotting/FWHM calculation.
    sideband_multiplier : float
        Sideband width multiplier for background estimation.

    Returns
    -------
    TimingFitResult
    """
    if xr is None:
        xr = [300, 330]

    label = f"{channel}_{coord}"
    size = data_np.shape[0]
    if size < 10:
        logger.warning(f"Too few events ({size}) at {coord} for {channel}")
        return _empty_result(coord, channel, size)

    # Count sidebands (on the full uncut distribution is better,
    # but we work with what's passed in — caller should pass wider range if needed)
    bkg_left, bkg_right = 0.0, 0.0  # sidebands computed externally if needed

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

    coeff_val = 0.0

    if include_background:
        coeffs = [zfit.Parameter(f"coeff_0_{label}", 0, -2, 1, floating=False)]
        chebyshev = zfit.pdf.Chebyshev(obs=obs, coeffs=coeffs)
        bkg_yield_param = zfit.Parameter(f"bkg_yield_{label}", bkg_yield_naive,
                                         bkg_yield_naive * 0.01,
                                         bkg_yield_naive * 2.5, step_size=1)
        bkg_ext = chebyshev.create_extended(bkg_yield_param)
        model = zfit.pdf.SumPDF([emg_ext, bkg_ext])
    else:
        model = emg_ext
        bkg_yield_param = None

    # --- Fit -----------------------------------------------------------------
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    converged = result.converged
    if not converged:
        logger.warning(f"Fit did not converge for {label}")

    # Hesse errors
    try:
        param_hesse = result.hesse()
    except Exception as e:
        logger.warning(f"Hesse failed for {label}: {e}")
        param_hesse = {}

    # --- Extract results -----------------------------------------------------
    mu_val = float(zfit.run(mu.value()))
    sigma_val = float(zfit.run(sigma.value()))
    lambd_val = float(zfit.run(lambd.value()))

    mu_err = param_hesse.get(mu, {}).get("error", np.nan)
    sigma_err = param_hesse.get(sigma, {}).get("error", np.nan)
    lambd_err = param_hesse.get(lambd, {}).get("error", np.nan)

    # Transit time = mu + 1/lambda (mean of EMG distribution)
    if lambd_val != 0:
        transit_time = mu_val + 1.0 / lambd_val
        # Error propagation: delta(TT) = sqrt(delta_mu^2 + (delta_lambda / lambda^2)^2)
        transit_time_err = np.sqrt(mu_err**2 + (lambd_err / lambd_val**2)**2)
    else:
        transit_time = mu_val
        transit_time_err = mu_err

    # FWHM (TTS)
    tts_fwhm = _compute_fwhm(model, data, size, nbins=nbins)

    # Yields
    sig_yield_val = float(zfit.run(emg_yield.value()))
    sig_yield_err = param_hesse.get(emg_yield, {}).get("error", np.nan)

    if include_background and bkg_yield_param is not None:
        bkg_yield_val = float(zfit.run(bkg_yield_param.value()))
        bkg_yield_err = param_hesse.get(bkg_yield_param, {}).get("error", np.nan)
        coeff_val = float(zfit.run(coeffs[0].value()))
    else:
        bkg_yield_val = 0.0
        bkg_yield_err = 0.0

    return TimingFitResult(
        coord=coord,
        channel=channel,
        mu=mu_val,
        mu_err=mu_err,
        sigma=sigma_val,
        sigma_err=sigma_err,
        lambd=lambd_val,
        lambd_err=lambd_err,
        transit_time=transit_time,
        transit_time_err=transit_time_err,
        tts_fwhm=tts_fwhm,
        sig_yield=sig_yield_val,
        sig_yield_err=sig_yield_err,
        bkg_yield=bkg_yield_val,
        bkg_yield_err=bkg_yield_err,
        total_events=size,
        bkg_left=bkg_left,
        bkg_right=bkg_right,
        converged=converged,
        coeff=coeff_val,
    )


def _empty_result(coord, channel, size):
    """Return a NaN-filled result for failed fits."""
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


def extract_timing_data(df: pd.DataFrame, delta_column: str,
                        xr: List[float],
                        require_pulse: Optional[str] = None) -> np.ndarray:
    """
    Extract timing offset data from a DataFrame, handling both scalar
    and LEDTimes (vector) delta columns.

    Parameters
    ----------
    df : DataFrame
        Full event DataFrame for one scan point.
    delta_column : str
        Name of the delta column (e.g. "delta_PMT_laser_LED").
    xr : list
        [lower, upper] range to select events.
    require_pulse : str or None
        If set, require this column > 0 (e.g. "PMT_PulseStart").

    Returns
    -------
    ndarray
        Timing offset values within the specified range.
    """
    mask = pd.Series(True, index=df.index)
    if require_pulse and require_pulse in df.columns:
        mask &= df[require_pulse] > 0

    filtered = df.loc[mask, delta_column]

    # Handle LEDTimes columns (arrays per event) vs scalar columns
    if delta_column.endswith("_LED"):
        # Each entry is an array — concatenate all
        arrays = filtered.dropna().values
        if len(arrays) == 0:
            return np.array([])
        all_vals = np.concatenate([np.atleast_1d(a) for a in arrays])
    else:
        all_vals = filtered.dropna().values.astype(float)

    # Apply range cut
    in_range = all_vals[(all_vals >= xr[0]) & (all_vals <= xr[1])]
    return in_range


def run_timing_analysis(scan_data: Dict[Tuple, pd.DataFrame],
                        coords: List[Tuple[float, float]],
                        config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run timing fits for all scan points for PMT, SiPM, and optionally monitor.

    Parameters
    ----------
    scan_data : dict
        Mapping (theta, phi) → DataFrame.
    coords : list
        Ordered scan coordinates.
    config : dict
        Full configuration dictionary.

    Returns
    -------
    pmt_results_df : DataFrame
        Timing fit results for the PMT channel.
    sipm_results_df : DataFrame
        Timing fit results for the SiPM channel.
    mon_results_df : DataFrame or None
        Timing fit results for the monitor (if enabled).
    """
    timing_cfg = config["timing"]
    pmt_cfg = timing_cfg["pmt"]
    sipm_cfg = timing_cfg["sipm"]
    sideband_mult = timing_cfg.get("sideband_multiplier", 2)

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
                coord, pmt_data, channel="PMT",
                xr=pmt_cfg["fit_range"],
                include_background=pmt_cfg["include_background"],
                nbins=pmt_cfg["nbins"],
                sideband_multiplier=sideband_mult,
            )
            pmt_results.append(pmt_result.to_dict())
        else:
            logger.warning(f"No PMT timing data for {coord}")

        # --- SiPM timing fit ---
        sipm_data = extract_timing_data(
            df, "delta_sipm_laser_LED", sipm_cfg["fit_range"],
            require_pulse="sipm_PulseStart"
        )
        if len(sipm_data) > 0:
            sipm_result = fit_timing(
                coord, sipm_data, channel="sipm",
                xr=sipm_cfg["fit_range"],
                include_background=sipm_cfg["include_background"],
                nbins=sipm_cfg["nbins"],
                sideband_multiplier=sideband_mult,
            )
            sipm_results.append(sipm_result.to_dict())
        else:
            logger.warning(f"No SiPM timing data for {coord}")

        # --- Monitor (if enabled) ---
        if config["monitor"]["enabled"]:
            mon_xr = config["monitor"]["fit_range"]
            mon_data = extract_timing_data(
                df, "delta_mon_laser_LED", mon_xr,
                require_pulse="mon_PulseStart"
            )
            if len(mon_data) > 0:
                mon_result = fit_timing(
                    coord, mon_data, channel="mon",
                    xr=mon_xr,
                    include_background=True,
                    nbins=50,
                    sideband_multiplier=sideband_mult,
                )
                mon_results.append(mon_result.to_dict())

    pmt_df = pd.DataFrame(pmt_results) if pmt_results else pd.DataFrame()
    sipm_df = pd.DataFrame(sipm_results) if sipm_results else pd.DataFrame()
    mon_df = pd.DataFrame(mon_results) if mon_results else None

    return pmt_df, sipm_df, mon_df
