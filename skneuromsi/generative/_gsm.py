#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""Gaussian Scale Mixture (GSM) model from Echeveste et al. (2020).

This module implements the Gaussian Scale Mixture generative model used in
Echeveste et al. (2020) "Cortical-like dynamics in recurrent circuits optimized
for sampling-based probabilistic inference".

The GSM generates natural image patches using oriented Gabor filters and
contrast scaling, providing realistic visual stimuli for neural network models.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

# Default GSM parameters from Echeveste et al. (2020) Supplementary Material
DEFAULT_PATCH_SIZE = 16  # 16x16 pixel patches
DEFAULT_N_ORIENTATIONS = 50  # Number of oriented filters
DEFAULT_SPATIAL_FREQ = 0.2  # Spatial frequency of Gabor filters
DEFAULT_BANDWIDTH = 1.0  # Bandwidth of Gabor filters
DEFAULT_H_SCALE = 1.0 / 15.0  # Scaling factor for SSN input h
DEFAULT_GAMMA = 0.0  # Contrast-proportional contribution (disabled)


# =============================================================================
# GABOR FILTER UTILITIES
# =============================================================================


def generate_gabor_filters(
    patch_size,
    n_orientations,
    spatial_freq=DEFAULT_SPATIAL_FREQ,
    bandwidth=DEFAULT_BANDWIDTH,
):
    """Generate a bank of oriented Gabor filters.

    Parameters
    ----------
    patch_size : int
        Size of the image patches (patch_size x patch_size).
    n_orientations : int
        Number of oriented filters to generate.
    spatial_freq : float, optional
        Spatial frequency of the Gabor filters. Default: 0.2.
    bandwidth : float, optional
        Bandwidth of the Gabor filters. Default: 1.0.

    Returns
    -------
    numpy.ndarray
        Array of shape (patch_size**2, n_orientations) containing the
        vectorized Gabor filters.
    """
    # Create coordinate grids
    x = np.linspace(-patch_size / 2, patch_size / 2, patch_size)
    y = np.linspace(-patch_size / 2, patch_size / 2, patch_size)
    X, Y = np.meshgrid(x, y)

    # Generate orientations uniformly distributed from 0 to π
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    filters = np.zeros((patch_size * patch_size, n_orientations))

    for i, theta in enumerate(orientations):
        # Rotate coordinates
        x_rot = X * np.cos(theta) + Y * np.sin(theta)
        y_rot = -X * np.sin(theta) + Y * np.cos(theta)

        # Gabor filter parameters
        sigma = bandwidth / (2 * np.pi * spatial_freq)

        # Generate Gabor filter
        gaussian = np.exp(-(x_rot**2 + y_rot**2 / (2 * sigma**2)))
        cosine = np.cos(2 * np.pi * spatial_freq * x_rot)

        gabor = gaussian * cosine

        # Normalize to unit norm
        gabor = gabor.flatten()
        gabor = gabor / np.linalg.norm(gabor)

        filters[:, i] = gabor

    return filters


def generate_covariance_matrix(n_orientations, correlation_strength=0.5):
    """Generate prior covariance matrix for GSM.

    Parameters
    ----------
    n_orientations : int
        Number of orientations (size of covariance matrix).
    correlation_strength : float, optional
        Strength of correlations between orientations. Default: 0.5.

    Returns
    -------
    numpy.ndarray
        Covariance matrix of shape (n_orientations, n_orientations).
    """
    # Create distance matrix based on orientation differences
    orientations = np.linspace(0, np.pi, n_orientations, endpoint=False)

    C = np.zeros((n_orientations, n_orientations))

    for i in range(n_orientations):
        for j in range(n_orientations):
            # Circular distance between orientations
            angle_diff = np.abs(orientations[i] - orientations[j])
            angle_diff = min(
                angle_diff, np.pi - angle_diff
            )  # Wrap to [0, π/2]

            # Exponential decay correlation
            C[i, j] = np.exp(-angle_diff / correlation_strength)

    return C


# =============================================================================
# GSM CLASS
# =============================================================================


class GSM:
    """Gaussian Scale Mixture (GSM) generative model.

    This class implements a flexible GSM for generating natural image patches
    with oriented texture content using Gabor filter banks and prior
    covariances.

    The GSM generates oriented texture patches using:
    1. A bank of oriented Gabor filters (matrix A)
    2. A prior covariance matrix capturing orientation correlations (matrix C)
    3. Contrast scaling to simulate different stimulus intensities

    Parameters
    ----------
    patch_size : int, optional
        Size of image patches (patch_size x patch_size). Default: 16.
    n_orientations : int, optional
        Number of oriented filters. Default: 50.
    h_scale : float, optional
        Scaling factor for SSN input generation. Default: 1/15.
    gamma : float, optional
        Contrast-proportional contribution. Default: 0.0.
    spatial_freq : float, optional
        Spatial frequency of Gabor filters. Default: 0.2.
    bandwidth : float, optional
        Bandwidth of Gabor filters. Default: 1.0.
    correlation_strength : float, optional
        Strength of orientation correlations. Default: 0.5.
    noise_variance : float, optional
        Variance of observation noise. Default: 0.01.
    random_seed : int, optional
        Random seed for reproducibility. Default: None.
    """

    def __init__(
        self,
        *,
        patch_size=DEFAULT_PATCH_SIZE,
        n_orientations=DEFAULT_N_ORIENTATIONS,
        h_scale=DEFAULT_H_SCALE,
        gamma=DEFAULT_GAMMA,
        spatial_freq=DEFAULT_SPATIAL_FREQ,
        bandwidth=DEFAULT_BANDWIDTH,
        alpha_h=1.96,
        beta_h=0.10,
        gamma_h=2.03,
        correlation_strength=0.5,
        noise_variance=0.01,
        random_seed=None,
    ):
        """Initialize the GSM model."""
        # Model parameters
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.h_scale = h_scale
        self.gamma = gamma
        self.spatial_freq = spatial_freq
        self.bandwidth = bandwidth
        self.correlation_strength = correlation_strength
        self.noise_variance = noise_variance

        # Nonlinearity parameters for SSN input h
        self.alpha_h = alpha_h  # Input scaling
        self.beta_h = beta_h  # Input baseline
        self.gamma_h = gamma_h  # Input power

        # Initialize random state
        if random_seed is not None:
            np.random.seed(random_seed)
        self._random_state = np.random.RandomState(random_seed)

        # Generate Gabor filter bank
        self.A = generate_gabor_filters(
            patch_size, n_orientations, spatial_freq, bandwidth
        )

        # Generate prior covariance matrix
        self.C = generate_covariance_matrix(
            n_orientations, correlation_strength
        )
        self.C_inv = np.linalg.inv(self.C)

        # Precompute W_ff AQUI, en el constructor.
        A_concat = np.concatenate([self.A, self.A], axis=1)
        self.W_ff = A_concat.T / 15.0

        # Precompute useful matrices
        self.ATA = self.A.T @ self.A

        # Store dimensions
        self.patch_dim = patch_size * patch_size
        self.orientation_dim = n_orientations

    def set_random(self, random_seed):
        """Set random seed for reproducible stimulus generation."""
        if random_seed is not None:
            np.random.seed(random_seed)
        self._random_state = np.random.RandomState(random_seed)

    def generate_stimulus_patch(self, contrast):
        """Generate a single stimulus patch with given contrast.

        Parameters
        ----------
        contrast : float
            Contrast level for the generated patch.

        Returns
        -------
        dict
            Dictionary containing:
            - 'x': Generated image patch (flattened)
            - 'y': Latent orientation representation
            - 'z': True contrast value
        """
        # Sample from prior distribution over orientations eq.2
        y = self._random_state.multivariate_normal(
            np.zeros(self.n_orientations), self.C
        )

        # Generate image patch: x = z * A @ y + noise eq.1
        clean_patch = contrast * (self.A @ y)
        noise = self._random_state.normal(
            0, np.sqrt(self.noise_variance), self.patch_dim
        )
        x = clean_patch + noise

        return {"x": x, "y": y, "z": contrast}

    def generate_h_input_efficient(self, x):
        """
        Efficient version assuming W_ff is precomputed.

        based on Echeveste et al. (2020) eq. 14.
        """
        # Linear filtering
        filter_response = self.W_ff @ x

        # Nonlinearity with optimized parameters
        h = self.alpha_h * np.power(
            self.beta_h + filter_response, self.gamma_h
        )

        return h

    def generate_h_input(self, x):
        """Generate SSN input h from image patch.

        Following Echeveste et al. (2020) equation:
        h = h_scale * (A.T @ x)

        Parameters
        ----------
        x : numpy.ndarray
            Image patch (flattened).

        Returns
        -------
        numpy.ndarray
            Input vector h for SSN.
        """
        # Main GSM transformation: h = h_scale * (A.T @ x)
        filter_response = self.A.T @ x
        h = self.h_scale * filter_response
        return h

    def generate_stimuli(self, contrasts, n_samples):
        """Generate stimulus patches and SSN inputs.

        Parameters
        ----------
        contrasts : array_like
            Array of contrast levels to generate.
        n_samples : int
            Number of samples to generate per contrast level.

        Returns
        -------
        dict
            Dictionary containing generated stimuli, h_inputs, and
            true_contrasts.
        """
        contrasts = np.asarray(contrasts)
        n_contrasts = len(contrasts)

        # Storage arrays
        all_stimuli = np.zeros((n_contrasts, n_samples, self.patch_dim))
        all_h_inputs = np.zeros((n_contrasts, n_samples, self.n_orientations))
        all_true_contrasts = np.zeros((n_contrasts, n_samples))

        # Generate stimuli for each contrast level
        for i, contrast in enumerate(contrasts):
            for j in range(n_samples):
                # Generate stimulus patch
                stimulus_data = self.generate_stimulus_patch(contrast)

                # Store patch
                all_stimuli[i, j] = stimulus_data["x"]
                all_true_contrasts[i, j] = stimulus_data["z"]

                # Generate SSN input h
                h_input = self.generate_h_input(stimulus_data["x"])
                all_h_inputs[i, j] = h_input

        return {
            "stimuli": all_stimuli,
            "h_inputs": all_h_inputs,
            "true_contrasts": all_true_contrasts,
        }

    # Methods for compatibility with Echeveste SSN
    def get_h_for_contrast_level(self, contrast_level, n_samples=1):
        """Get SSN input h for specific contrast level (compatibility).

        Parameters
        ----------
        contrast_level : float
            Contrast level (0-4 in Echeveste's original implementation).
        n_samples : int, optional
            Number of samples to generate. Default: 1.

        Returns
        -------
        numpy.ndarray
            SSN input vectors h of shape (n_samples, n_orientations).
        """
        # Map contrast level to actual contrast value
        # Echeveste uses levels 0-4, we map to reasonable contrast range
        contrast_mapping = {
            0: 0.1,  # Low contrast
            1: 0.3,  # Medium-low contrast
            2: 0.5,  # Medium contrast
            3: 0.7,  # Medium-high contrast
            4: 1.0,  # High contrast
        }

        contrast = contrast_mapping.get(contrast_level, contrast_level)

        # Generate samples
        h_samples = []
        for _ in range(n_samples):
            stimulus_data = self.generate_stimulus_patch(contrast)
            h_input = self.generate_h_input(stimulus_data["x"])
            h_samples.append(h_input)

        return np.array(h_samples)

    def get_gabor_filters(self):
        """Get the Gabor filter bank (A matrix)."""
        return self.A.copy()

    def get_prior_covariance(self):
        """Get the prior covariance matrix (C matrix)."""
        return self.C.copy()


# CONVENIENCE FUNCTIONS =======================================================


def create_echeveste_gsm(random_seed=None):
    """Create GSM with Echeveste et al. (2020) default parameters.

    Parameters
    ----------
    random_seed : int, optional
        Random seed for reproducibility. Default: None.

    Returns
    -------
    GSM
        Configured GSM instance with Echeveste parameters.
    """
    return GSM(
        patch_size=16,
        n_orientations=50,
        h_scale=1.0 / 15.0,
        gamma=0.0,
        spatial_freq=0.2,
        bandwidth=1.0,
        correlation_strength=0.5,
        noise_variance=0.01,
        random_seed=random_seed,
    )
