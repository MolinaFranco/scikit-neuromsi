#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Independent GSM data loader for Echeveste2020.

This module provides access only to the Gaussian Scale Mixture (GSM)
components used in Echeveste et al. (2020), decoupled from SSN data.
"""

from pathlib import Path

import numpy as np


class GSMDataLoader:
    """Loader for GSM-related data."""

    def __init__(self):
        """Initialize GSM data loader."""
        self.data_path = Path(__file__).parent / "gsm"

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"GSM data directory not found at {self.data_path}"
            )

    # GSM Data Loading Methods
    def load_gabor_filters(self):
        """
        Load pre-trained Gabor filters.

        Returns
        -------
        A : np.ndarray, shape (256, 50)
            Pre-trained Gabor filter matrix from natural image statistics
        """
        return np.loadtxt(self.data_path / "A")

    def load_prior_covariance(self):
        """
        Load prior covariance matrix.

        Returns
        -------
        C : np.ndarray, shape (50, 50)
            Prior covariance matrix learned from natural image statistics
        """
        return np.loadtxt(self.data_path / "C")

    def load_contrast_levels(self):
        """Load available contrast levels.

        Returns
        -------
        contrasts : np.ndarray, shape (10,)
            Array of 10 contrast levels used in the original experiments
        """
        return np.loadtxt(self.data_path / "z_array")

    # Stimuli
    def load_stimulus(self, contrast_idx):
        """
        Load stimulus x and transformed h for a given contrast index.

        Parameters
        ----------
        contrast_idx : int
            Index of contrast level (0, 1, 2, ...)

        Returns
        -------
        x : np.ndarray
            Stimulus image patch (256,)
        h : np.ndarray
            GSM transformed input (50,)
        """
        x = np.loadtxt(self.data_path / f"x_{contrast_idx}")
        h = np.loadtxt(self.data_path / f"h_{contrast_idx}")
        return x, h

    def load_all_stimuli(self, indices=(0, 1, 2)):
        """
        Load all reference stimuli for specified contrast indices.

        Returns
        -------
        dict
            {contrast_idx: (x, h)}
        """
        return {i: self.load_stimulus(i) for i in indices}

    # Validation
    def validate_data_integrity(self):
        """Validate that GSM data files exist and shapes are correct."""
        try:
            A = self.load_gabor_filters()
            assert A.shape == (256, 50), f"A shape {A.shape} != (256, 50)"

            C = self.load_prior_covariance()
            assert C.shape == (50, 50), f"C shape {C.shape} != (50, 50)"
            assert np.allclose(C, C.T, rtol=1e-10), "C must be symmetric"

            z_array = self.load_contrast_levels()
            assert z_array.ndim == 1, "z_array must be 1D"

            for i in [0, 1, 2]:
                x, h = self.load_stimulus(i)
                assert x.shape == (256,), f"x_{i} shape {x.shape} != (256,)"
                assert h.shape == (50,), f"h_{i} shape {h.shape} != (50,)"

            return True
        except Exception as e:
            print(f"GSM data validation failed: {e}")
            return False


# Convenience function
def load_gsm_data():
    """Return GSM data loader instance."""
    return GSMDataLoader()
