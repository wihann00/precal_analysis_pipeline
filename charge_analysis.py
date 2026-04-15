"""
charge_analysis.py
==================
Fits PMT charge distributions to extract gain (mean SPE charge / e).

Model: Pedestal + 1PE (mixture of fully-amplified + optional underamplified
       + optional backscatter) + optional multi-PE peaks (2PE, 3PE).

Plots are generated inside fit_charge() while the zfit model is still
in scope, matching the pattern used in timing_analysis.py.
"""

import zfit
from zfit import z
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import logging
import plotting

from zfit_models import BackscatterPDF
from zfit_models import compute_chi2_ndf

logger = logging.getLogger(__name__)

E_CHARGE_PC = 1.602e-7  # elementary charge in picocoulombs


@dataclass
class ChargeFitResult:
    """Container for a single charge fit result."""
    coord: Tuple[float, float]
    mu_spe: float
    mu_spe_err: float
    sigma_spe: float
    gain: float
    pedestal_yield: float
    spe_yield: float          # total 1PE yield (includes uamp + backscatter)
    spe_yield_err: float
    p_uamp: float             # branching fraction (0 if not included)
    p_backscatter: float      # branching fraction (0 if not included)
    twope_yield: float
    threepe_yield: float
    total_events: int
    converged: bool
    chi2_ndf: float

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
            "p_uamp": self.p_uamp,
            "p_backscatter": self.p_backscatter,
            "twope_yield": self.twope_yield,
            "threepe_yield": self.threepe_yield,
            "total_events": self.total_events,
            "converged": self.converged,
            "chi2_ndf": self.chi2_ndf,
        }


def fit_charge(coord: Tuple[float, float],
               data_np: np.ndarray,
               xr: List[float] = None,
               npe: int = 2,
               include_pedestal: bool = True,
               include_underamplified: bool = False,
               include_backscatter: bool = False,
               include_log: bool = False,
               nbins: int = 50,
               make_plot: bool = False,
               output_dir: str = ".",
               run_id: str = "",
               pmt_serial: str = "",
               fmt: str = "png",
               dpi: int = 150) -> ChargeFitResult:
    """
    Fit a charge distribution with multi-component Gaussian model.
    Generates fit plot while model is still in scope if make_plot=True.

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
        Include underamplified Gaussian component in 1PE mixture.
    include_backscatter : bool
        Include backscatter flat distribution in 1PE mixture.
    nbins : int
        Number of bins for histogramming.
    make_plot : bool
        Whether to generate fit plots.
    output_dir, run_id, pmt_serial, fmt, dpi
        Plotting parameters (passed to FitPlotter).

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
        mu0 = zfit.Parameter(f"mu0_{label}", 0.03, -0.15, 0.15)
        sigma0 = zfit.Parameter(f"sigma0_{label}", 0.14, 0.01, 10)
        gauss0 = zfit.pdf.Gauss(obs=obs, mu=mu0, sigma=sigma0)
        gauss0_yield = zfit.Parameter(f"gauss0_yield_{label}", size * 0.5,
                                       0, size * 0.9, step_size=1)
        gauss0_ext = gauss0.create_extended(gauss0_yield)
        models_to_sum.append(gauss0_ext)
        model_names.append("Pedestal")
    else:
        gauss0_yield = None

    # --- 1PE model (mixture of full + optional uamp + optional backscatter) ---
    mu1 = zfit.Parameter(f"mu1_{label}", e_val, 0.8 * e_val, 1.5 * e_val)
    sigma1 = zfit.Parameter(f"sigma1_{label}", 0.47,
                             0.0001 * int_std, 10 * int_std)
    gauss1 = zfit.pdf.Gauss(obs=obs, mu=mu1, sigma=sigma1)

    # Sub-components and branching fractions for the 1PE mixture
    spe_sub_pdfs = []
    spe_fracs = []

    if include_underamplified:
        p_uamp = zfit.Parameter(f"p_uamp_{label}", 0.36, 0.001, 0.4)
        sf_uamp = zfit.Parameter(f"sf_uamp_{label}", 0.12, 0.1, 0.5)
        sf_sigma_uamp = zfit.Parameter(f"sf_sigma_uamp_{label}", 7.1, 0.3, 15)

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
        spe_sub_pdfs.append(gauss_uamp)
        spe_fracs.append(p_uamp)

    if include_backscatter and include_pedestal:
        p_backscatter = zfit.Parameter(f"p_backscatter_{label}", 0.014, 0.001, 0.8)
        backscatter_pdf = BackscatterPDF(obs=obs, mu0=mu0, sigma0=sigma0,
                                          mu1=mu1, sigma1=sigma1,
                                          name=f"backscatter_{label}")
        spe_sub_pdfs.append(backscatter_pdf)
        spe_fracs.append(p_backscatter)

    # Build the 1PE model: mixture if sub-components exist, plain Gaussian otherwise
    if spe_sub_pdfs:
        model_1pe = zfit.pdf.SumPDF([*spe_sub_pdfs, gauss1], fracs=spe_fracs)
        model_names.append([name for name in
                            (["Under Amplified"] if include_underamplified else []) +
                            (["Backscatter"] if include_backscatter else []) +
                            ["1PE"]])
    else:
        model_1pe = gauss1
        model_names.append("1PE")

    model_1pe_yield = zfit.Parameter(f"model_1pe_yield_{label}", size * 0.3,
                                      0, size * 0.8, step_size=1)
    model_1pe_ext = model_1pe.create_extended(model_1pe_yield)
    models_to_sum.append(model_1pe_ext)

    # --- Multi-PE peaks ---
    gauss2_yield = None
    gauss3_yield = None

    if npe > 1:
        mu2 = zfit.ComposedParameter(f"mu2_{label}", lambda mu1: 2 * mu1, params=[mu1])
        sigma2 = zfit.ComposedParameter(f"sigma2_{label}",
                                          lambda s: np.sqrt(2) * s, params=[sigma1])
        gauss2 = zfit.pdf.Gauss(obs=obs, mu=mu2, sigma=sigma2)
        gauss2_yield = zfit.Parameter(f"gauss2_yield_{label}", size * 0.01,
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

    try:
        result = minimizer.minimize(nll)
    except Exception as e:
        logger.warning(f"Charge fit failed for {label}: {e}")
        return _empty_charge_result(coord, size)

    converged = result.converged
    if not converged:
        logger.warning(f"Charge fit did not converge for {label}")

    # Log fitted parameter values (check if any hit their limits)
    logger.info(f"\nParameter Results for {label}:\n{result.params}")
    # logger.info(f"\nFit result for {label}:")
    # for p in result.params:
    #     val = float(zfit.run(p.value()))
    #     lower = p.lower if p.has_limits else None
    #     upper = p.upper if p.has_limits else None
    #     at_limit = ""
    #     if lower is not None and abs(val - lower) < 1e-6:
    #         at_limit = " *** AT LOWER LIMIT ***"
    #     if upper is not None and abs(val - upper) < 1e-6:
    #         at_limit = " *** AT UPPER LIMIT ***"
    #     logger.info(f"  {p.name}: {val:.6f}  (limits: [{lower}, {upper}]){at_limit}")

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
    mu1_val = float(zfit.run(mu1.value()))
    sigma1_val = float(zfit.run(sigma1.value()))
    mu1_err = param_hesse.get(mu1, {}).get("error", np.nan)
    gain = mu1_val / E_CHARGE_PC

    spe_yield_val = float(zfit.run(model_1pe_yield.value()))
    spe_yield_err = param_hesse.get(model_1pe_yield, {}).get("error", np.nan)

    ped_yield = float(zfit.run(gauss0_yield.value())) if gauss0_yield else 0.0
    p_uamp_val = float(zfit.run(p_uamp.value())) if include_underamplified else 0.0
    p_bs_val = float(zfit.run(p_backscatter.value())) if (include_backscatter and include_pedestal) else 0.0
    twope = float(zfit.run(gauss2_yield.value())) if gauss2_yield else 0.0
    threepe = float(zfit.run(gauss3_yield.value())) if gauss3_yield else 0.0

    # --- Plots (while model is alive!) ---------------------------------------
    if make_plot:
        fitplotter = plotting.FitPlotter(coord, "charge", pmt_serial,
                                          output_dir, run_id, nbins, fmt, dpi)

        fit_params = {
            r"$\mu_{SPE}$": f"{mu1_val:.2f}",
            r"$\sigma_{SPE}$": f"{sigma1_val:.2f}",
            "Gain": f"{gain:.2E}",
            "SPE yield": f"{spe_yield_val:.0f}",
            "Ped yield": f"{ped_yield:.0f}",
        }

        fitplotter.plot_fit_and_pull(model, data, size, True,
                                      model_names, fit_params, xr,
                                      x_label="Charge (pC)",
                                      fit_type="charge",
                                      inc_ped=include_pedestal,
                                      inc_log=include_log)

    # Count free parameters
    n_free_params = len([p for p in result.params if p.floating])
    chi2, ndf = compute_chi2_ndf(model, data, size, nbins, xr, n_free_params)
    chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
    logger.info(f"  {label}: chi2/ndf = {chi2:.1f}/{ndf} = {chi2_ndf:.2f}")

    return ChargeFitResult(
        coord=coord,
        mu_spe=mu1_val,
        mu_spe_err=mu1_err,
        sigma_spe=sigma1_val,
        gain=gain,
        pedestal_yield=ped_yield,
        spe_yield=spe_yield_val,
        spe_yield_err=spe_yield_err,
        p_uamp=p_uamp_val,
        p_backscatter=p_bs_val,
        twope_yield=twope,
        threepe_yield=threepe,
        total_events=size,
        converged=converged,
        chi2_ndf=chi2_ndf
    )


def _empty_charge_result(coord, size):
    return ChargeFitResult(
        coord=coord,
        mu_spe=np.nan, mu_spe_err=np.nan, sigma_spe=np.nan,
        gain=np.nan,
        pedestal_yield=np.nan, spe_yield=np.nan, spe_yield_err=np.nan,
        p_uamp=np.nan, p_backscatter=np.nan,
        twope_yield=np.nan, threepe_yield=np.nan,
        total_events=size, converged=False,
    )


def run_charge_analysis(scan_data: Dict[Tuple, pd.DataFrame],
                        coords: List[Tuple[float, float]],
                        config: dict) -> pd.DataFrame:
    """
    Run charge fits for all scan points.
    """
    charge_cfg = config["charge"]
    plot_cfg = config.get("plotting", {})
    save_plots = plot_cfg.get("save_fit_plots", False)
    output_dir = config.get("output_dir", ".")
    run_id = config.get("run_id", "")
    pmt_serial = config.get("pmt_serial", "")
    fmt = plot_cfg.get("figure_format", "png")
    dpi = plot_cfg.get("dpi", 150)

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

        try:
            res = fit_charge(
                coord, data_np,
                xr=xr,
                npe=charge_cfg["npe"],
                include_pedestal=charge_cfg["include_pedestal"],
                include_underamplified=charge_cfg["include_underamplified"],
                include_backscatter=charge_cfg.get("include_backscatter", False),
                include_log=charge_cfg.get("include_log", False),
                nbins=charge_cfg["nbins"],
                make_plot=save_plots,
                output_dir=output_dir,
                run_id=run_id,
                pmt_serial=pmt_serial,
                fmt=fmt,
                dpi=dpi,
            )
            results.append(res.to_dict())
        except Exception as e:
            logger.warning(f"Charge fit failed for {coord}: {e}")
            results.append(_empty_charge_result(coord, len(data_np)).to_dict())

    # Log chi2/ndf ranking
    if results:
        results_df = pd.DataFrame(results)
        if "chi2_ndf" in results_df.columns:
            ranked = results_df[["coord", "chi2_ndf"]].dropna()
            ranked["dist_from_1"] = abs(ranked["chi2_ndf"] - 1)
            ranked = ranked.sort_values("dist_from_1")
            logger.info("\n" + "=" * 50)
            logger.info(" CHARGE FIT QUALITY RANKING (chi2/ndf)")
            logger.info("=" * 50)
            for _, row in ranked.iterrows():
                quality = "GOOD" if row["dist_from_1"] < 0.5 else "OK" if row["dist_from_1"] < 1.0 else "POOR"
                logger.info(f"  {str(row['coord']):>15s}:  {row['chi2_ndf']:.2f}  [{quality}]")
            logger.info("=" * 50)

    return pd.DataFrame(results) if results else pd.DataFrame()