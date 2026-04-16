#!/usr/bin/env python3
"""
run_analysis.py
===============
Main entry point for the PMT pre-calibration analysis pipeline.

Usage:
    python run_analysis.py --config config.yaml

This script:
  1. Reads the YAML configuration
  2. Generates scan coordinates
  3. Loads ROOT files for all scan points
  4. Runs timing fits (EMG) for PMT, SiPM, and optionally monitor
  5. Optionally runs charge distribution fits for gain
  6. Computes all relative quantities with error propagation
  7. Generates fit plots, cross-section plots, and 2D heatmaps
  8. Saves a summary DataFrame
"""

import argparse
import yaml
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scan_geometry import ScanGeometry
from data_loader import load_all_scan_points
from timing_analysis import run_timing_analysis
from charge_analysis import run_charge_analysis
from relative_quantities import compute_relative_quantities
from plotting import (
    plot_cross_sections,
    plot_heatmaps,
    plot_parameter_summary,
)


def setup_logging(output_dir: str, run_id: str):
    """Configure logging to both console and file."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"{run_id}_analysis.log"),
        ],
    )


def main(config_path: str):
    # ---- Load configuration --------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_id = config["run_id"]
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    pmt_channel = config["pmt_channel"]
    sample_to_ns = config["sample_to_ns"]
    plot_cfg = config["plotting"]
    pmt_serial = config["pmt_serial"]

    setup_logging(output_dir, run_id)
    logger = logging.getLogger("run_analysis")
    logger.info(f"Starting analysis for run {run_id}")
    logger.info(f"PMT channel: CH{pmt_channel}")

    t_start = time.time()

    # ---- Generate scan geometry ----------------------------------------------
    geometry = ScanGeometry(
        zeniths=config["scan"]["zeniths"],
        azimuths=config["scan"]["azimuths"],
    )
    logger.info(f"Scan geometry: {geometry.n_points} unique points")
    logger.info(f"Coordinates: {geometry.coords}")

    # ---- Load ROOT files -----------------------------------------------------
    logger.info("Loading ROOT files...")
    scan_data = load_all_scan_points(
        data_dir=data_dir,
        run_id=run_id,
        coords=geometry.coords,
        pmt_channel=pmt_channel,
        sample_to_ns=sample_to_ns,
    )
    logger.info(f"Loaded {len(scan_data)}/{geometry.n_points} scan points")

    if len(scan_data) == 0:
        logger.error("No data loaded. Check data_dir and run_id in config.")
        sys.exit(1)

    # Use only coords that were successfully loaded
    loaded_coords = [c for c in geometry.coords if c in scan_data]

    # ---- Timing analysis -----------------------------------------------------
    # Fit plots are generated INSIDE run_timing_analysis (while zfit model
    # is in scope) when save_fit_plots is True — this ensures the model curve
    # and pull distribution are properly overlaid on the data.
    if config["timing"]["enabled"]:
        logger.info("Running timing analysis...")
        pmt_df, sipm_df, mon_df = run_timing_analysis(scan_data, loaded_coords, config)

        logger.info(f"PMT fits: {len(pmt_df)} points")
        logger.info(f"SiPM fits: {len(sipm_df)} points")
        if mon_df is not None:
            logger.info(f"Monitor fits: {len(mon_df)} points")

    # ---- Charge analysis (if enabled) ----------------------------------------
    charge_df = None
    if config["charge"]["enabled"]:
        logger.info("Running charge analysis...")
        charge_df = run_charge_analysis(scan_data, loaded_coords, config)
        logger.info(f"Charge fits: {len(charge_df)} points")

    # ---- Compute relative quantities -----------------------------------------
    logger.info("Computing relative quantities...")
    summary_df = compute_relative_quantities(
        scan_data=scan_data,
        pmt_df=pmt_df,
        sipm_df=sipm_df,
        mon_df=mon_df,
        charge_df=charge_df,
    )

    # ---- Generate cross-section and heatmap plots ----------------------------
    eff_range = plot_cfg.get("eff_range")
    if eff_range is None:
        eff_range = [0.2, 1.5] # default efficiency range if not given in config file

    if plot_cfg["save_cross_sections"]:
        logger.info("Generating cross-section plots...")
        plot_cross_sections(summary_df, geometry, pmt_serial,
                            output_dir=output_dir, run_id=run_id, eff_range=eff_range,
                            fmt=plot_cfg["figure_format"], dpi=plot_cfg["dpi"])

    if plot_cfg["save_heatmap"]:
        logger.info("Generating heatmaps...")
        plot_heatmaps(summary_df, geometry, pmt_serial,
                       output_dir=output_dir, run_id=run_id, eff_range=eff_range,
                       fmt=plot_cfg["figure_format"], dpi=plot_cfg["dpi"])

    # Parameter summary plot
    plot_parameter_summary(summary_df, pmt_serial=pmt_serial, output_dir=output_dir, run_id=run_id,
                           fmt=plot_cfg["figure_format"], dpi=plot_cfg["dpi"])

    # ---- Save results --------------------------------------------------------
    results_dir = Path(output_dir) / "data" / "results" / f"{run_id}-{pmt_serial}"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_pickle(results_dir / "summary.pkl")
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    pmt_df.to_pickle(results_dir / "pmt_timing_fits.pkl")
    sipm_df.to_pickle(results_dir / "sipm_timing_fits.pkl")
    if mon_df is not None:
        mon_df.to_pickle(results_dir / "monitor_fits.pkl")
    if charge_df is not None and not charge_df.empty:
        charge_df.to_pickle(results_dir / "charge_fits.pkl")

    # Save config for reproducibility
    with open(results_dir / "config_used.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    elapsed = time.time() - t_start
    logger.info(f"Analysis complete in {elapsed:.1f}s")
    logger.info(f"Results saved to {results_dir}")
    logger.info(f"Figures saved to {Path(output_dir) / 'figures'}")

    # Print quick summary
    print("\n" + "=" * 60)
    print(f" ANALYSIS SUMMARY — {run_id}")
    print("=" * 60)
    print(f" Scan points analysed:  {len(summary_df)}")
    print(f" PMT channel:           CH{pmt_channel}")
    if "corrected_rel_efficiency" in summary_df.columns:
        eff = summary_df["corrected_rel_efficiency"].values
        print(f" Corrected ε range:     [{np.nanmin(eff):.3f}, {np.nanmax(eff):.3f}]")
    if "pmt_tts_fwhm" in summary_df.columns:
        tts = summary_df["pmt_tts_fwhm"].values
        print(f" TTS FWHM range:        [{np.nanmin(tts):.2f}, {np.nanmax(tts):.2f}] ns")
    if "rel_transit_time" in summary_df.columns:
        tt = summary_df["rel_transit_time"].values
        print(f" Transit time spread:   {np.nanmax(tt) - np.nanmin(tt):.2f} ns")
    if charge_df is not None and "gain" in charge_df.columns:
        gains = charge_df["gain"].values
        print(f" Gain range:            [{np.nanmin(gains):.2E}, {np.nanmax(gains):.2E}]")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PMT Pre-Calibration Analysis Pipeline"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    main(args.config)
