"""
relative_quantities.py
======================
Computes relative quantities normalised to the centre scan point:
  - Relative PMT signal yield
  - Relative SiPM signal yield
  - Corrected relative detection efficiency (PMT/SiPM double ratio)
  - Relative timing offset
  - Relative TTS
  - Relative gain (when charge analysis is available)

All with proper error propagation.
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _relative_to_centre(values: np.ndarray, errors: np.ndarray,
                         centre_val: float, centre_err: float):
    """
    Compute values relative to centre point: rel = val / centre_val.

    Error propagation for ratio a/b:
        δ(a/b) = (a/b) × sqrt((δa/a)² + (δb/b)²)
    """
    rel = values / centre_val
    rel_err = rel * np.sqrt(
        (errors / values)**2 + (centre_err / centre_val)**2
    )
    return rel, rel_err


def _double_ratio(pmt_yield: np.ndarray, pmt_err: np.ndarray,
                   sipm_yield: np.ndarray, sipm_err: np.ndarray,
                   pmt_centre: float, pmt_centre_err: float,
                   sipm_centre: float, sipm_centre_err: float):
    """
    Corrected relative detection efficiency via double ratio:
        ε_rel = (N_PMT / N_SiPM) / (N_PMT_centre / N_SiPM_centre)

    This is equivalent to: (N_PMT/N_PMT_centre) / (N_SiPM/N_SiPM_centre)
    i.e. the ratio of the two relative yields.
    """
    rel_pmt, rel_pmt_err = _relative_to_centre(pmt_yield, pmt_err,
                                                 pmt_centre, pmt_centre_err)
    rel_sipm, rel_sipm_err = _relative_to_centre(sipm_yield, sipm_err,
                                                   sipm_centre, sipm_centre_err)

    ratio = rel_pmt / rel_sipm
    ratio_err = ratio * np.sqrt(
        (rel_pmt_err / rel_pmt)**2 + (rel_sipm_err / rel_sipm)**2
    )

    return ratio, ratio_err, rel_pmt, rel_pmt_err, rel_sipm, rel_sipm_err


def compute_relative_quantities(pmt_df: pd.DataFrame,
                                 sipm_df: pd.DataFrame,
                                 mon_df: Optional[pd.DataFrame] = None,
                                 charge_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Compute all relative quantities normalised to the centre (first) scan point.

    Parameters
    ----------
    pmt_df : DataFrame
        PMT timing fit results (from timing_analysis).
    sipm_df : DataFrame
        SiPM timing fit results.
    mon_df : DataFrame or None
        Monitor timing fit results (for laser stability tracking).
    charge_df : DataFrame or None
        Charge fit results (for gain analysis).

    Returns
    -------
    DataFrame
        Combined results with all relative quantities.
    """
    # Use first point as centre reference
    pmt_centre = pmt_df.iloc[0]
    sipm_centre = sipm_df.iloc[0]

    # --- Relative signal yields & corrected efficiency -----------------------
    pmt_sig = pmt_df["sig_yield"].values
    pmt_sig_err = pmt_df["sig_yield_err"].values
    sipm_sig = sipm_df["sig_yield"].values
    sipm_sig_err = sipm_df["sig_yield_err"].values

    (corr_eff, corr_eff_err,
     rel_pmt, rel_pmt_err,
     rel_sipm, rel_sipm_err) = _double_ratio(
        pmt_sig, pmt_sig_err,
        sipm_sig, sipm_sig_err,
        pmt_centre["sig_yield"], pmt_centre["sig_yield_err"],
        sipm_centre["sig_yield"], sipm_centre["sig_yield_err"],
    )

    # --- Relative timing offset (transit time) -------------------------------
    tt = pmt_df["transit_time"].values
    tt_err = pmt_df["transit_time_err"].values
    # Express as difference from centre rather than ratio (timing is additive)
    rel_tt = tt - tt[0]
    rel_tt_err = np.sqrt(tt_err**2 + tt_err[0]**2)

    # --- Relative TTS --------------------------------------------------------
    tts = pmt_df["tts_fwhm"].values
    tts_centre = tts[0]
    rel_tts = tts / tts_centre
    # Simple error estimate (no error on FWHM from fit, so just relative)
    rel_tts_err = np.full_like(rel_tts, np.nan)

    # --- Build output DataFrame -----------------------------------------------
    out = pd.DataFrame({
        "coord": pmt_df["coord"].values,
        # Absolute quantities
        "pmt_sig_yield": pmt_sig,
        "pmt_sig_yield_err": pmt_sig_err,
        "sipm_sig_yield": sipm_sig,
        "sipm_sig_yield_err": sipm_sig_err,
        "pmt_transit_time": tt,
        "pmt_transit_time_err": tt_err,
        "pmt_tts_fwhm": tts,
        "pmt_mu": pmt_df["mu"].values,
        "pmt_mu_err": pmt_df["mu_err"].values,
        "pmt_sigma": pmt_df["sigma"].values,
        "pmt_lambd": pmt_df["lambd"].values,
        # Relative quantities
        "rel_pmt_yield": rel_pmt,
        "rel_pmt_yield_err": rel_pmt_err,
        "rel_sipm_yield": rel_sipm,
        "rel_sipm_yield_err": rel_sipm_err,
        "corrected_rel_efficiency": corr_eff,
        "corrected_rel_efficiency_err": corr_eff_err,
        "rel_transit_time": rel_tt,
        "rel_transit_time_err": rel_tt_err,
        "rel_tts": rel_tts,
        "rel_tts_err": rel_tts_err,
    })

    # --- Monitor stability (if available) ------------------------------------
    if mon_df is not None and not mon_df.empty:
        mon_sig = mon_df["sig_yield"].values
        mon_sig_err = mon_df["sig_yield_err"].values
        mon_centre = mon_sig[0]
        mon_centre_err = mon_sig_err[0]

        rel_mon, rel_mon_err = _relative_to_centre(
            mon_sig, mon_sig_err, mon_centre, mon_centre_err
        )
        out["mon_sig_yield"] = mon_sig
        out["mon_sig_yield_err"] = mon_sig_err
        out["rel_mon_yield"] = rel_mon
        out["rel_mon_yield_err"] = rel_mon_err

    # --- Gain (if charge analysis available) ---------------------------------
    if charge_df is not None and not charge_df.empty:
        gains = charge_df["gain"].values
        gain_centre = gains[0]
        rel_gain = gains / gain_centre
        out["gain"] = gains
        out["rel_gain"] = rel_gain
        out["mu_spe"] = charge_df["mu_spe"].values

    return out
