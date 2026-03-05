"""
scan_geometry.py
================
Handles the scan coordinate generation (zenith/azimuth serpentine pattern)
and provides indexing utilities for cross-section and heatmap plots.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ScanGeometry:
    """
    Generates and manages the scan coordinate system for PMT photocathode mapping.

    The scan follows a serpentine pattern across azimuth groups, with the
    (0, 0) centre point scanned only once (first encounter).

    Parameters
    ----------
    zeniths : list of int/float
        Zenith angles in degrees (e.g. [0, 10, 20, 30, 40, 50]).
    azimuths : list of int/float
        Azimuth angles in degrees (e.g. [0, 90, 180, 270]).
    """

    zeniths: List[float]
    azimuths: List[float]

    # Populated by __post_init__
    coords: List[Tuple[float, float]] = field(init=False, repr=False)
    full_coords: List[Tuple[float, float]] = field(init=False, repr=False)
    coord_labels: List[str] = field(init=False, repr=False)
    n_points: int = field(init=False)

    def __post_init__(self):
        zeniths_rev = self.zeniths[::-1]
        zenith_dict = {0: list(self.zeniths), 1: list(zeniths_rev)}

        self.coords = []        # unique scan points (no duplicate centre)
        self.full_coords = []   # all points including skipped-centre markers
        zeroth_scan = False

        for i, phi in enumerate(self.azimuths):
            for theta in zenith_dict[i % 2]:
                if theta == 0:
                    coord = (0, 0)
                    if not zeroth_scan:
                        zeroth_scan = True
                    else:
                        self.full_coords.append(coord)
                        continue
                else:
                    coord = (theta, phi)

                self.full_coords.append(coord)
                self.coords.append(coord)

        self.coord_labels = [str(c) for c in self.coords]
        self.n_points = len(self.coords)

    # ----- Cross-section index helpers ----------------------------------------

    def _azimuth_group_indices(self, azimuth_group: int,
                               exclude_centre: bool = False) -> np.ndarray:
        """
        Return indices into self.coords for a given azimuth group.

        Groups: 0 → first azimuth (includes centre at index 0),
                1, 2, 3 → subsequent azimuths (no centre).
        """
        n_zen = len(self.zeniths)
        if azimuth_group == 0:
            indices = np.arange(n_zen)
            if exclude_centre:
                indices = indices[1:]
        else:
            start = n_zen + (azimuth_group - 1) * (n_zen - 1)
            indices = np.arange(start, start + (n_zen - 1))
        return indices

    def cross_section_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (x_move, y_move, angle_axis) for cross-section plots.

        Convention (matching your existing code):
          x-axis: azimuth 90° (group 1) → centre → azimuth 270° (group 3)
          y-axis: azimuth 0° (group 0, excl. centre) → centre → azimuth 180° (group 2)

        Returns
        -------
        x_move : ndarray of int
            Indices into self.coords for the x cross-section.
        y_move : ndarray of int
            Indices into self.coords for the y cross-section.
        angle_axis : ndarray
            Signed zenith angles (negative → positive) for the cross-section x-axis labels.
        """
        n_zen = len(self.zeniths)
        zeniths_rev_neg = [-z for z in self.zeniths[::-1][:-1]]  # e.g. [-50, -40, -30, -20, -10]
        angle_axis = np.array(zeniths_rev_neg + list(self.zeniths))

        # x cross-section: 90° reversed → centre → 270°
        x_move = np.concatenate((
            self._azimuth_group_indices(1),      # 90° (50→10 in serpentine)
            np.array([0]),                        # centre
            np.flip(self._azimuth_group_indices(3))  # 270° reversed
        )).astype(int)

        # y cross-section: 0° (excl centre, reversed) → centre → 180°
        y_move = np.concatenate((
            np.flip(self._azimuth_group_indices(0, exclude_centre=True)),
            np.array([0]),
            self._azimuth_group_indices(2)
        )).astype(int)

        return x_move, y_move, angle_axis

    def heatmap_grid(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Build a 2D grid mapping for zenith × azimuth heatmap visualisation.

        Returns
        -------
        zenith_vals : ndarray
            Unique zenith values (rows).
        azimuth_vals : ndarray
            Unique azimuth values (columns).
        grid_index : dict
            Mapping (theta, phi) → index into self.coords.
        """
        grid_index = {}
        for idx, (theta, phi) in enumerate(self.coords):
            grid_index[(theta, phi)] = idx

        return np.array(self.zeniths), np.array(self.azimuths), grid_index
