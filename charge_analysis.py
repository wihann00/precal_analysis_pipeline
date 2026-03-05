"""
charge_analysis.py
==================
Fits PMT charge distributions to extract gain (mean SPE charge / e).

Model: Pedestal + 1PE Gaussian + optional underamplified/backscatter
       + optional multi-PE peaks (2PE, 3PE).

This module is ready to use once pyrate supports simultaneous
PulseStart + PulseCharge extraction.
"""

import zfit
from zfit import z
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging

from zfit_models import BackscatterPDF

logger = logging.getLogger(__name__)

E_CHARGE_PC = 1.602e-7  # elementary charge in picocoulombs


@dataclass
class ChargeFitResult:
    """Container for a single charge fit result."""
    coord: Tuple[float, float]

    # SPE parameters
    mu_spe: float
    mu_spe_err: float
    sigma_spe: float
    gain: float                 # mu_spe / e

    # Yields
    pedestal_yield: float
    spe_yield: float
    spe_yield_err: float
    uamp_yield: float           # underamplified
    backscatter_yield: float
    twope_yield: float
    threepe_yield: float

    total_events: int
    converged: bool

    def to_dict(self) -> dict:
        return {
            "coord": self.coord,
            "mu_spe": self.mu_spe,
            "mu_spe_err": self.mu_spe_err,
            "sigma_spe": self.sigma_spe,
            "gain": self.gain,
            "pedestal_yield": self.pedestal_yield,
            "spe_yield": self.spe_yield,
            "spe_yield_err": self.spe_yield_err,
            "uamp_yield": self.uamp_yield,
            "backscatter_yield": self.backscatter_yield,
            "twope_yield": self.twope_yield,
            "threepe_yield": self.threepe_yield,
            "total_events": self.total_events,
            "converged": self.converged,
        }


def fit_charge(coord: Tuple[float, float],
               data_np: np.ndarray,
               xr: List[float] = None,
               npe: int = 2,
               include_pedestal: bool = True,
               include_underamplified: bool = False,
               include_backscatter: bool = False,
               nbins: int = 50) -> ChargeFitResult:
    """
    Fit a charge distribution with multi-component Gaussian model.

    Parameters
    ----------
    coord : tuple
        (theta, phi) scan coordinate.
    data_np : ndarray
        Charge values (already filtered by timing cut if applicable).
    xr : list
        [lower, upper] charge range in pC.
    npe : int
        Number of PE peaks to include (1, 2, or 3).
    include_pedestal : bool
        Include pedestal Gaussian.
    include_underamplified : bool
        Include underamplified Gaussian component.
    include_backscatter : bool
        Include backscatter flat distribution.
    nbins : int
        Number of bins for histogramming.

    Returns
    -------
    ChargeFitResult
    """
    if xr is None:
        xr = [1.5, 5.0]

    label = f"charge_{coord}"
    size = data_np.shape[0]

    if size < 10:
        logger.warning(f"Too few events ({size}) at {coord} for charge fit")
        return _empty_charge_result(coord, size)

    obs = zfit.Space("x", limits=(xr[0], xr[1]))
    data = zfit.Data.from_numpy(obs=obs, array=data_np)

    int_mean = float(np.mean(data_np))
    int_std = float(np.std(data_np))
    e_val = 1.602  # expected SPE mean in pC (approximate)

    model_names = []
    models_to_sum = []

    # --- Pedestal ---
    if include_pedestal:
        mu0 = zfit.Parameter(f"mu0_{label}", 0, -0.15, 0.15)
        sigma0 = zfit.Parameter(f"sigma0_{label}", 2, 0.01, 10)
        gauss0 = zfit.pdf.Gauss(obs=obs, mu=mu0, sigma=sigma0)
        gauss0_yield = zfit.Parameter(f"gauss0_yield_{label}", size * 0.5,
                                       0, size * 0.9, step_size=1)
        gauss0_ext = gauss0.create_extended(gauss0_yield)
        models_to_sum.append(gauss0_ext)
        model_names.append("Pedestal")
    else:
        gauss0_yield = None

    # --- 1PE Gaussian ---
    mu1 = zfit.Parameter(f"mu1_{label}", e_val, 0.8 * e_val, 1.2 * e_val)
    sigma1 = zfit.Parameter(f"sigma1_{label}", int_std,
                             0.0001 * int_std, 10 * int_std)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)
    gauss1_yield = zfit.Parameter(f"gauss1_yield_{label}", size * 0.3,
                                   0, size, step_size=1)
    gauss1_ext = gauss1.create_extended(gauss1_yield)
    models_to_sum.append(gauss1_ext)
    model_names.append("1PE")

    # --- Underamplified ---
    uamp_yield_param = None
    if include_underamplified:
        p_uamp = zfit.Parameter(f"p_uamp_{label}", 0.01, 0.001, 0.8)
        sf_uamp = zfit.Parameter(f"sf_uamp_{label}", 0.15, 0.1, 0.5)
        sf_sigma_uamp = zfit.Parameter(f"sf_sigma_uamp_{label}", 2, 0.3, 10)

        mu_uamp = zfit.ComposedParameter(
            f"mu_uamp_{label}",
            lambda params: params["sf"] * params["mu1"],
            params={"sf": sf_uamp, "mu1": mu1}
        )
        sigma_uamp = zfit.ComposedParameter(
            f"sigma_uamp_{label}",
            lambda params: params["sigma1"] * (params["mu_uamp"] / params["mu1"]) * params["sf_sigma"],
            params={"sigma1": sigma1, "mu_uamp": mu_uamp, "mu1": mu1, "sf_sigma": sf_sigma_uamp}
        )

        gauss_uamp = zfit.pdf.Gauss(obs=obs, mu=mu_uamp, sigma=sigma_uamp)
        uamp_yield_param = zfit.ComposedParameter(
            f"uamp_yield_{label}",
            lambda params: params["p_uamp"] * params["gauss1_yield"],
            params={"p_uamp": p_uamp, "gauss1_yield": gauss1_yield}
        )
        gauss_uamp_ext = gauss_uamp.create_extended(uamp_yield_param)
        models_to_sum.append(gauss_uamp_ext)
        model_names.append("Underamplified")

    # --- Backscatter ---
    backscatter_yield_param = None
    if include_backscatter and include_pedestal:
        backscatter_pdf = BackscatterPDF(obs=obs, mu0=mu0, sigma0=sigma0,
                                          mu1=mu1, sigma1=sigma1,
                                          name=f"backscatter_{label}")
        backscatter_yield_param = zfit.Parameter(f"bs_yield_{label}", size * 0.05,
                                                   0, size * 0.2, step_size=1)
        backscatter_ext = backscatter_pdf.create_extended(backscatter_yield_param)
        models_to_sum.append(backscatter_ext)
        model_names.append("Backscatter")

    # --- Multi-PE peaks ---
    gauss2_yield = None
    gauss3_yield = None

    if npe > 1:
        mu2 = zfit.ComposedParameter(f"mu2_{label}", lambda mu1: 2 * mu1, params=[mu1])
        sigma2 = zfit.ComposedParameter(f"sigma2_{label}",
                                          lambda s: np.sqrt(2) * s, params=[sigma1])
        gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
        gauss2_yield = zfit.Parameter(f"gauss2_yield_{label}", size * 0.1,
                                       0, size * 0.4, step_size=1)
        gauss2_ext = gauss2.create_extended(gauss2_yield)
        models_to_sum.append(gauss2_ext)
        model_names.append("2PE")

    if npe > 2:
        mu3 = zfit.ComposedParameter(f"mu3_{label}", lambda mu1: 3 * mu1, params=[mu1])
        sigma3 = zfit.ComposedParameter(f"sigma3_{label}",
                                          lambda s: np.sqrt(3) * s, params=[sigma1])
        gauss3 = zfit.pdf.Gauss(obs=obs, mu=mu3, sigma=sigma3)
        gauss3_yield = zfit.Parameter(f"gauss3_yield_{label}", size * 0.1,
                                       0, size * 0.4, step_size=1)
        gauss3_ext = gauss3.create_extended(gauss3_yield)
        models_to_sum.append(gauss3_ext)
        model_names.append("3PE")

    # --- Fit -----------------------------------------------------------------
    model = zfit.pdf.SumPDF(models_to_sum)
    nll = zfit.loss.ExtendedUnbinnedNLL(model, data)
    minimizer = zfit.minimize.Minuit()
    result = minimizer.minimize(nll)

    converged = result.converged
    if not converged:
        logger.warning(f"Charge fit did not converge for {label}")

    try:
        param_hesse = result.hesse()
    except Exception:
        param_hesse = {}

    # --- Extract results -----------------------------------------------------
    mu1_val = float(zfit.run(mu1.value()))
    sigma1_val = float(zfit.run(sigma1.value()))
    mu1_err = param_hesse.get(mu1, {}).get("error", np.nan)

    gain = mu1_val / E_CHARGE_PC

    spe_yield_val = float(zfit.run(gauss1_yield.value()))
    spe_yield_err = param_hesse.get(gauss1_yield, {}).get("error", np.nan)

    ped_yield = float(zfit.run(gauss0_yield.value())) if gauss0_yield else 0.0
    uamp_yield = float(zfit.run(uamp_yield_param.value())) if uamp_yield_param else 0.0
    bs_yield = float(zfit.run(backscatter_yield_param.value())) if backscatter_yield_param else 0.0
    twope = float(zfit.run(gauss2_yield.value())) if gauss2_yield else 0.0
    threepe = float(zfit.run(gauss3_yield.value())) if gauss3_yield else 0.0

    return ChargeFitResult(
        coord=coord,
        mu_spe=mu1_val,
        mu_spe_err=mu1_err,
        sigma_spe=sigma1_val,
        gain=gain,
        pedestal_yield=ped_yield,
        spe_yield=spe_yield_val,
        spe_yield_err=spe_yield_err,
        uamp_yield=uamp_yield,
        backscatter_yield=bs_yield,
        twope_yield=twope,
        threepe_yield=threepe,
        total_events=size,
        converged=converged,
    )


def _empty_charge_result(coord, size):
    return ChargeFitResult(
        coord=coord,
        mu_spe=np.nan, mu_spe_err=np.nan, sigma_spe=np.nan,
        gain=np.nan,
        pedestal_yield=np.nan, spe_yield=np.nan, spe_yield_err=np.nan,
        uamp_yield=np.nan, backscatter_yield=np.nan,
        twope_yield=np.nan, threepe_yield=np.nan,
        total_events=size, converged=False,
    )


def run_charge_analysis(scan_data: Dict[Tuple, pd.DataFrame],
                        coords: List[Tuple[float, float]],
                        config: dict) -> pd.DataFrame:
    """
    Run charge fits for all scan points.

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
    DataFrame
        Charge fit results for all points.
    """
    charge_cfg = config["charge"]
    results = []

    for coord in coords:
        if coord not in scan_data:
            continue

        df = scan_data[coord]
        theta, phi = coord

        # Apply timing cut to select good events, then take charge
        timing_cut = charge_cfg["timing_cut"]
        xr = charge_cfg["fit_range"]

        query_str = (f"{xr[0]} < PMT_PulseCharge < {xr[1]} & "
                     f"{timing_cut[0]} < delta_PMT_laser < {timing_cut[1]}")
        try:
            df_cut = df.query(query_str)
        except Exception:
            # If delta column doesn't exist, just use charge range
            query_str = f"{xr[0]} < PMT_PulseCharge < {xr[1]}"
            df_cut = df.query(query_str)

        data_np = df_cut["PMT_PulseCharge"].values.astype(float)

        logger.info(f"Charge fit ({theta}, {phi}): {len(data_np)} events")

        res = fit_charge(
            coord, data_np,
            xr=xr,
            npe=charge_cfg["npe"],
            include_pedestal=charge_cfg["include_pedestal"],
            include_underamplified=charge_cfg["include_underamplified"],
            include_backscatter=charge_cfg.get("include_backscatter", False),
            nbins=charge_cfg["nbins"],
        )
        results.append(res.to_dict())

    return pd.DataFrame(results) if results else pd.DataFrame()
