# PMT Pre-Calibration Analysis Pipeline

Analysis pipeline for characterising the photocathode uniformity of 50 cm Hyper-Kamiokande photomultiplier tubes (PMTs). Scans the PMT surface in spherical coordinates (zenith/azimuth) and extracts position-dependent timing, detection efficiency, and gain parameters.

Developed for the Hyper-K PMT pre-calibration programme at the University of Melbourne.


## Quick start

```bash
# Set up environment (Python 3.10)
python3.10 -m venv precal_env
source precal_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Edit config.yaml for your run, then:
python run_analysis.py --config config.yaml
```


## What it does

The pipeline takes ROOT files produced by [pyrate](https://github.com/your-pyrate-repo) (one per scan point) and in a single execution:

1. Loads waveform parameters (PulseStart, CFDPulseStart, PulseCharge, PeakHeight, LEDTimes) from all channels
2. Computes timing deltas between PMT, SiPM, monitor, and laser trigger
3. Fits timing offset distributions with an Exponentially Modified Gaussian (EMG) + optional Chebyshev background using [zfit](https://github.com/zfit/zfit) and iminuit
4. Extracts transit time (μ + 1/λ), transit time spread (FWHM), and signal/background yields per scan point
5. Computes relative quantities normalised to the centre point, including the corrected relative detection efficiency via the PMT/SiPM double ratio
6. Optionally fits charge distributions for SPE gain extraction
7. Generates all plots: individual fit overlays with pulls, cross-section profiles, polar and rectangular heatmaps, parameter summaries, and monitor stability


## Configuration

All run settings are controlled through a single YAML file. Key sections:

```yaml
run_id: "260204_225940_1"           # identifies this analysis run
pmt_serial: "EL1635-B"             # PMT serial number (appears on plots)
data_dir: "/path/to/rootfiles"     # directory containing ROOT files
output_dir: "/path/to/output"      # where results and figures are saved

pmt_channel: 2                     # 2 for PMT1, 3 for PMT2
sample_to_ns: 2                    # digitizer sample-to-ns conversion

scan:
  zeniths: [0, 10, 20, 30, 40, 50]
  azimuths: [0, 90, 180, 270]

timing:
  pmt:
    fit_range: [300, 320]          # ns — adjust to your PMT timing peak
    include_background: true
    nbins: 50
  sipm:
    fit_range: [95, 105]
    include_background: true
    nbins: 50
    charge_cut: 790                # SiPM PulseCharge upper cut

monitor:
  enabled: true
  fit_range: [214, 224]

charge:
  enabled: false                   # enable when charge data is available
```

See `config.yaml` for the full template with all options documented.


## Modules

| File | Description |
|---|---|
| `run_analysis.py` | Main entry point. Reads config, orchestrates the full analysis, saves results. |
| `config.yaml` | YAML configuration template with all tuneable parameters. |
| `scan_geometry.py` | Generates the serpentine scan coordinate pattern and provides indexing for cross-section/heatmap plots. |
| `data_loader.py` | Discovers ROOT files by coordinate, extracts branches via uproot, computes all timing deltas. |
| `timing_analysis.py` | EMG + Chebyshev fitting of timing distributions. Extracts μ, σ, λ, transit time, TTS, yields with Hesse errors. Calls `FitPlotter` while the model is in scope. |
| `charge_analysis.py` | SPE charge distribution fitting (pedestal + 1PE + optional underamplified/backscatter + multi-PE). Extracts gain. Ready for when pyrate supports simultaneous PulseStart + PulseCharge. |
| `relative_quantities.py` | Normalises all quantities to the centre scan point. Computes the corrected relative detection efficiency via the double ratio (PMT_rel / SiPM_rel). Full error propagation. |
| `plotting.py` | `FitPlotter` class for per-point fit plots with model overlay and pulls. Standalone functions for cross-section profiles, polar/rectangular heatmaps, parameter summaries, and monitor stability. |
| `zfit_models.py` | Custom zfit PDFs: `ExponentiallyModifiedGaussian`, `BackscatterPDF`, `JohnsonSU`. |
| `requirements.txt` | Pinned dependencies with known-compatible versions for Python 3.10. |


## Output structure

```
<output_dir>/
├── figures/
│   ├── timing_fits/<run_id>/       # per-point fit + pull plots (PMT, SiPM, monitor)
│   ├── FWHM/<run_id>/             # per-point FWHM visualisations
│   ├── cross_sections/<run_id>/   # yield, efficiency, transit time, TTS profiles
│   ├── heatmaps/<run_id>/         # polar wedge + rectangular grid heatmaps
│   └── summary/<run_id>/          # 2×2 parameter grid (μ, σ, λ, TTS)
├── data/results/<run_id>/
│   ├── summary.pkl                # full results DataFrame
│   ├── summary.csv
│   ├── pmt_timing_fits.pkl
│   ├── sipm_timing_fits.pkl
│   ├── monitor_fits.pkl
│   └── config_used.yaml           # config snapshot for reproducibility
└── logs/
    └── <run_id>_analysis.log
```


## Experimental setup

The pipeline assumes the following hardware configuration:

- **CH0**: Laser trigger / signal generator
- **CH1**: Hamamatsu MPPC (SiPM) — normalization reference via decoupling fibre
- **CH2/CH3**: 50 cm Hyper-K PMT under test (select via `pmt_channel`)
- **CH4**: Monitor PMT — laser stability tracking

Light is delivered via optical fibre with ND filters and beam splitters. The SiPM monitors the delivered light intensity at each scan position, enabling the double-ratio correction that cancels laser fluctuations from the relative detection efficiency measurement.


## Key physics notes

- **Corrected relative detection efficiency**: `ε_rel = (N_PMT/N_SiPM) / (N_PMT_centre/N_SiPM_centre)` — multiplicative normalisation, not additive.
- **Transit time**: Defined as μ + 1/λ from the EMG fit (the distribution mean, not the mode).
- **TTS**: Full width at half maximum of the fitted timing model, computed via spline root-finding.
- **Occupancy**: Optimal SPE calibration at ~0.08–0.09 (low enough to suppress multi-PE pile-up).
- **Coordinate convention**: 270° azimuth corresponds to the +x direction.


## Dependencies

Tested with Python 3.10.4. Key version constraints:

- `tensorflow==2.16.2` with `protobuf>=4.25,<5.0` (avoids the GetPrototype crash)
- `tensorflow-probability==0.24.0` (matched to TF 2.16)
- `tf-keras>=2.16,<2.17` (TFP requires Keras 2, not Keras 3)
- `zfit>=0.24,<0.29`
- `numpy>=1.23,<2.0`

See `requirements.txt` for the full list.


## Authors

Wi Han Ng — University of Melbourne
