"""
data_loader.py
==============
Handles ROOT file discovery and extraction of waveform parameters into
pandas DataFrames, computing all relevant timing deltas.
"""

import numpy as np
import pandas as pd
import uproot
from glob import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

SAMPLE_TO_NS = 2  # default; overridden by config


def find_root_file(data_dir: str, run_id: str, theta: float, phi: float) -> str:
    """Locate the ROOT file for a given scan coordinate."""
    pattern = f"{data_dir}/*theta{int(theta)}_phi{int(phi)}*.root"
    matches = glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No ROOT file found for pattern: {pattern}")
    if len(matches) > 1:
        logger.warning(f"Multiple ROOT files match {pattern}, using first: {matches[0]}")
    return matches[0]


def extract_dataframe(rootfile: str, pmt_channel: int = 2,
                      sample_to_ns: float = 2) -> pd.DataFrame:
    """
    Extract waveform parameters from a ROOT file and compute timing deltas.

    Parameters
    ----------
    rootfile : str
        Path to the ROOT file.
    pmt_channel : int
        PMT channel number (2 for PMT1, 3 for PMT2).
    sample_to_ns : float
        Conversion factor from digitizer samples to nanoseconds.

    Returns
    -------
    pd.DataFrame
        DataFrame with all pulse parameters and computed deltas.
    """
    f = uproot.open(rootfile)

    ch_laser = 0
    ch_sipm = 1
    ch_pmt = pmt_channel
    ch_monitor = 4

    trees = {}
    for ch in [ch_laser, ch_sipm, ch_pmt, ch_monitor]:
        key = f"Tree_CH{ch}"
        if key in f:
            trees[ch] = f[key]
        else:
            logger.warning(f"Tree {key} not found in {rootfile}")
            trees[ch] = None

    dfs = []

    # --- Scalar branches: PulseStart, CFDPulseStart, PulseCharge, PeakHeight ---
    channel_names = {
        ch_laser: "laser",
        ch_sipm: "sipm",
        ch_pmt: "PMT",
        ch_monitor: "mon",
    }

    scalar_branches = ["PulseStart", "CFDPulseStart", "PulseCharge", "PeakHeight", "PeakLocation"]
    timing_branches = ["PulseStart", "CFDPulseStart"]  # these get sample_to_ns conversion

    for ch, name in channel_names.items():
        tree = trees[ch]
        if tree is None:
            continue
        available = tree.keys()
        for branch in scalar_branches:
            if branch not in available:
                continue
            col = tree[branch].arrays(library="pd").astype("float32")
            col_name = f"{name}_{branch}"
            col = col.rename(columns={branch: col_name})
            if branch in timing_branches:
                col[col_name] *= sample_to_ns
            dfs.append(col)

    # --- Vector branches: LEDTimes ---
    for ch, name in channel_names.items():
        tree = trees[ch]
        if tree is None:
            continue
        if "LEDTimes" not in tree.keys():
            continue
        led = tree["LEDTimes"].arrays(library="pd") * sample_to_ns
        led = led.rename(columns={"LEDTimes": f"{name}_LEDTimes"})
        dfs.append(led)

    df = pd.concat(dfs, axis=1)

    # --- Compute timing deltas ------------------------------------------------
    # delta{A}{B} = channel_A - channel_B  (using PulseStart)
    delta_pairs = {
        "delta_PMT_laser": ("PMT_PulseStart", "laser_PulseStart"),
        "delta_PMT_sipm": ("PMT_PulseStart", "sipm_PulseStart"),
        "delta_sipm_laser": ("sipm_PulseStart", "laser_PulseStart"),
        "delta_mon_laser": ("mon_PulseStart", "laser_PulseStart"),
    }
    for delta_name, (col_a, col_b) in delta_pairs.items():
        if col_a in df.columns and col_b in df.columns:
            df[delta_name] = df[col_a] - df[col_b]

    # CFD versions
    cfd_delta_pairs = {
        "delta_PMT_laser_cfd": ("PMT_CFDPulseStart", "laser_CFDPulseStart"),
        "delta_PMT_sipm_cfd": ("PMT_CFDPulseStart", "sipm_CFDPulseStart"),
        "delta_sipm_laser_cfd": ("sipm_CFDPulseStart", "laser_CFDPulseStart"),
        "delta_mon_laser_cfd": ("mon_CFDPulseStart", "laser_CFDPulseStart"),
    }
    for delta_name, (col_a, col_b) in cfd_delta_pairs.items():
        if col_a in df.columns and col_b in df.columns:
            df[delta_name] = df[col_a] - df[col_b]

    # LEDTimes deltas (vector - scalar → array per event)
    if "PMT_LEDTimes" in df.columns and "laser_PulseStart" in df.columns:
        df["delta_PMT_laser_LED"] = df["PMT_LEDTimes"].map(np.array) - df["laser_PulseStart"]
    if "sipm_LEDTimes" in df.columns and "laser_PulseStart" in df.columns:
        df["delta_sipm_laser_LED"] = df["sipm_LEDTimes"].map(np.array) - df["laser_PulseStart"]
    if "mon_LEDTimes" in df.columns and "laser_PulseStart" in df.columns:
        df["delta_mon_laser_LED"] = df["mon_LEDTimes"].map(np.array) - df["laser_PulseStart"]

    return df


def load_all_scan_points(data_dir: str, run_id: str,
                         coords: List[Tuple[float, float]],
                         pmt_channel: int = 2,
                         sample_to_ns: float = 2) -> Dict[Tuple[float, float], pd.DataFrame]:
    """
    Load ROOT files for all scan coordinates.

    Returns
    -------
    dict
        Mapping (theta, phi) → DataFrame.
    """
    data = {}
    for coord in coords:
        theta, phi = coord
        try:
            rootfile = find_root_file(data_dir, run_id, theta, phi)
            df = extract_dataframe(rootfile, pmt_channel=pmt_channel,
                                   sample_to_ns=sample_to_ns)
            data[coord] = df
            logger.info(f"Loaded ({theta}, {phi}): {len(df)} events")
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Failed to load ({theta}, {phi}): {e}")
            continue
    return data
