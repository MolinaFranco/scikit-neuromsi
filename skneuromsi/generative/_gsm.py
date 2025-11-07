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

from ..data import GSMDataLoader


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
        beta_h=1.9,  # Baseline from Echeveste et al. (2020) Table S1
        gamma_h=2.03,
        correlation_strength=0.5,
        noise_variance=0.01,
        random_seed=None,
        use_pretrained=True,
    ):
        """Initialize the GSM model.

        Parameters
        ----------
        use_pretrained : bool, optional
            If True, load pre-trained Gabor filters and covariance matrix
            from GSMDataLoader. If False, generate new ones.
            Default: True.
        """
        # Model parameters
        self.patch_size = patch_size
        self.n_orientations = n_orientations
        self.h_scale = h_scale
        self.gamma = gamma
        self.spatial_freq = spatial_freq
        self.bandwidth = bandwidth
        self.correlation_strength = correlation_strength
        self.noise_variance = noise_variance
        self.use_pretrained = use_pretrained

        # Nonlinearity parameters for SSN input h
        # Ref: Echeveste et al. 2020, ssn_inference_optimizer/objective.ml
        # Formula: h = α_h · (β_h + filter_response)^γ_h
        self.alpha_h = alpha_h  # α_h: Input scaling (multiplicador)
        self.beta_h = beta_h    # β_h: Input baseline (offset)
        self.gamma_h = gamma_h  # γ_h: Input power (exponente)

        # Initialize random state
        if random_seed is not None:
            np.random.seed(random_seed)
        self._random_state = np.random.RandomState(random_seed)

        # Load or generate Gabor filters and covariance matrix
        if use_pretrained:
            self._load_pretrained_data()
        else:
            self._generate_new_data()
            # Para _generate_new_data, necesitamos calcular patch_dim
            self.patch_dim = self.patch_size * self.patch_size
            self.orientation_dim = self.n_orientations

        # Precompute W_ff según especificación del paper
        # Ref: Echeveste et al. (2020) Supplementary Table S1, Input section:
        # "Feed-forward weights: [A A]^T / 15.0"
        # Para nuestro caso (solo población E): W_ff = A^T / 15.0
        # Dimensiones:
        # - A: (patch_dim, n_orientations) = (256, 50)
        # - W_ff = A.T / 15: (n_orientations, patch_dim) = (50, 256)
        # IMPORTANTE: El factor 1/15 es parte de W_ff, no se aplica después
        self.W_ff = self.A.T / 15.0

        # Verificar dimensiones
        expected_wff_shape = (self.n_orientations, self.patch_dim)
        assert self.W_ff.shape == expected_wff_shape, (
            f"W_ff shape mismatch: expected {expected_wff_shape}, "
            f"got {self.W_ff.shape}"
        )

        # Precompute useful matrices
        self.ATA = self.A.T @ self.A

    def _load_pretrained_data(self):
        """Load pre-trained Gabor filters and covariance matrix."""
        try:
            loader = GSMDataLoader()
            self.A = loader.load_gabor_filters()
            self.C = loader.load_prior_covariance()
            self.C_inv = np.linalg.inv(self.C)

            # Update dimensions
            self.patch_dim, self.orientation_dim = self.A.shape
            self.patch_size = int(np.sqrt(self.patch_dim))
            self.n_orientations = self.orientation_dim

            print("Loaded pre-trained GSM data from GSMDataLoader")

        except Exception as e:
            print(f"Warning: Could not load pre-trained GSM data: {e}")
            print("Falling back to generating new data")
            self._generate_new_data()

    def _generate_new_data(self):
        """Generate new Gabor filters and covariance matrix."""
        # Generate Gabor filter bank
        self.A = generate_gabor_filters(
            self.patch_size,
            self.n_orientations,
            self.spatial_freq,
            self.bandwidth,
        )

        # Generate prior covariance matrix
        self.C = generate_covariance_matrix(
            self.n_orientations, self.correlation_strength
        )
        self.C_inv = np.linalg.inv(self.C)

        print("Generated new GSM data")

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
        """Generate SSN input h with trained nonlinearity.

        Implements the full nonlinear transformation trained in the model:
        h = α_h · max(W_ff @ x + β_h, 0)^γ_h

        Where W_ff = A.T / 15.0 according to Echeveste et al. (2020)
        Supplementary Table S1.

        Parameters:
        - W_ff: Feed-forward weights = A.T / 15.0 (incluye h_scale)
        - α_h (alpha_h): input scaling (multiplicador) - controls magnitude
        - β_h (beta_h): input baseline (offset) - baseline before ReLU
        - γ_h (gamma_h): input power (exponente) - controls nonlinearity
        - max(·, 0): ReLU threshold - ensures non-negative values

        IMPORTANT: The threshold ReLU is applied AFTER adding the
        baseline β_h, as per the original implementation.

        Reference: Echeveste et al. (2020) Supplementary Table S1
                   transformed_moments_nonlinearity.py lines 27-28
                   ssn_inference_optimizer/objective.ml lines 410-413

        Parameters
        ----------
        x : numpy.ndarray
            Image patch (flattened).

        Returns
        -------
        numpy.ndarray
            Input vector h for SSN.
        """
        # Linear filtering: W_ff @ x donde W_ff = A.T / 15.0
        # Ref: Echeveste et al. (2020) Supplementary Table S1:
        # "Feed-forward weights: [A A]^T / 15.0"
        # El factor 1/15 ya está incluido en W_ff
        filter_response = self.W_ff @ x

        # Nonlinearidad con threshold ReLU:
        # h = α_h · max(filter_response + β_h, 0)^γ_h
        # Ref: transformed_moments_nonlinearity.py línea 27-28:
        # nl_fun(u, nl_scale, nl_baseline, nl_power) =
        #     nl_scale * (np.maximum(u + nl_baseline, 0) ** nl_power)
        argument = filter_response + self.beta_h

        # Aplicar threshold ReLU: max(argument, 0)
        argument_rectified = np.maximum(argument, 0.0)

        # Aplicar potencia y escala
        h = self.alpha_h * np.power(argument_rectified, self.gamma_h)

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

                # Generate SSN input h con transformación no lineal entrenada
                h_input = self.generate_h_input_efficient(stimulus_data["x"])
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
            h_input = self.generate_h_input_efficient(stimulus_data["x"])
            h_samples.append(h_input)

        return np.array(h_samples)

    def get_gabor_filters(self):
        """Get the Gabor filter bank (A matrix)."""
        return self.A.copy()

    def get_prior_covariance(self):
        """Get the prior covariance matrix (C matrix)."""
        return self.C.copy()

    def compute_posterior_moments(self, x, z_map=None):
        """
        Compute posterior moments (mean and covariance) for GSM inference.

        Implementa el cálculo del posterior P(y|x,z) del modelo GSM
        siguiendo Echeveste et al. (2020).

        Based on ssn_inference_numerical_experiments/GSM/GSM.py
        lines 101-112 (get_post_moments functions).

        Mathematical foundation:
        - Main paper Eq. 1-2: GSM generative model x = z*A*y + noise
        - Posterior: P(y|x,z) = N(μ_post, Σ_post)
        - μ_post = (z/σ_x²) * Σ_post * A^T * x
        - Σ_post = [C^(-1) + (z²/σ_x²) * A^T*A]^(-1)

        Parameters
        ----------
        x : np.ndarray
            Observed image patch (flattened), shape (patch_dim,)
        z_map : float, optional
            MAP estimate of contrast. If None, uses fixed value 0.5

        Returns
        -------
        mu_post : np.ndarray
            Posterior mean of orientation representation, shape (n_orient,)
        Sigma_post : np.ndarray
            Posterior covariance matrix, shape (n_orient, n_orient)

        Notes
        -----
        Esta función calcula los momentos del posterior condicional
        en el contraste z (asumiendo z conocido o en su valor MAP).
        Para inference completa, se debería integrar sobre P(z|x).
        """
        # Contraste MAP (por defecto o proporcionado)
        if z_map is None:
            # Heurística simple: usar contraste medio
            z_map = 0.5

        # Varianza del ruido de observación
        s_x_2 = self.noise_variance

        # Calcular Σ_post = [C^(-1) + (z²/σ_x²) * A^T*A]^(-1)
        # (GSM.py line 110-112)
        M = self.C_inv + (z_map**2 / s_x_2) * self.ATA
        Sigma_post = np.linalg.inv(M)

        # Calcular μ_post = (z/σ_x²) * Σ_post * A^T * x
        # (GSM.py line 106-107)
        mu_post = (z_map / s_x_2) * Sigma_post @ (self.A.T @ x)

        return mu_post, Sigma_post

    def compute_posterior_for_ssn_training(
        self, contrast, n_samples=100, z_map=None
    ):
        """
        Compute target posterior statistics for SSN training.

        Genera múltiples muestras de estímulos con un contraste dado
        y calcula los momentos promedio del posterior, que servirán
        como targets para el entrenamiento del SSN.

        Parameters
        ----------
        contrast : float
            Contrast level for stimulus generation
        n_samples : int, optional
            Number of samples to average over (default: 100)
        z_map : float, optional
            MAP contrast estimate (if None, uses contrast value)

        Returns
        -------
        dict
            Dictionary containing:
            - 'target_mu': Average posterior mean, shape (n_orientations,)
            - 'target_sigma': Average posterior covariance,
              shape (n_orientations, n_orientations)
            - 'h_inputs': SSN input vectors h, shape (n_samples, n_orient)
            - 'stimuli': Generated image patches, shape (n_samples, patch_dim)

        Notes
        -----
        Esta función genera los targets necesarios para entrenar el SSN:
        1. Genera n_samples estímulos con el contraste especificado
        2. Calcula el posterior GSM para cada uno
        3. Promedia los momentos para obtener estadísticas target
        4. Retorna también los inputs h correspondientes
        """
        if z_map is None:
            z_map = contrast

        # Almacenamiento
        all_mu = []
        all_Sigma = []
        all_h = []
        all_x = []

        # Generar muestras
        for _ in range(n_samples):
            # Generar estímulo
            stim_data = self.generate_stimulus_patch(contrast)
            x = stim_data['x']
            all_x.append(x)

            # Calcular posterior GSM
            mu_post, Sigma_post = self.compute_posterior_moments(x, z_map)
            all_mu.append(mu_post)
            all_Sigma.append(Sigma_post)

            # Generar input h para SSN con transformación no lineal entrenada
            h = self.generate_h_input_efficient(x)
            all_h.append(h)

        # Promediar momentos
        target_mu = np.mean(all_mu, axis=0)
        target_Sigma = np.mean(all_Sigma, axis=0)

        return {
            'target_mu': target_mu,
            'target_sigma': target_Sigma,
            'h_inputs': np.array(all_h),
            'stimuli': np.array(all_x),
        }

    def compute_input_baseline_lb(
            self, stimuli=None, n_samples=100, contrasts=None):
        """Compute lower bound for input baseline parameter.

        Calculates the minimum filter response across stimuli to determine
        a safe lower bound for the input baseline parameter β_h.
        This ensures that β_h + filter_response > 0 for all data.

        Following Echeveste et al. (2020) implementation:
        input_baseline_lb = 0.1 - min(filter_response)

        Parameters
        ----------
        stimuli : numpy.ndarray, optional
            Pre-generated stimuli of shape (n_samples, patch_dim).
            If None, generates new stimuli.
        n_samples : int, optional
            Number of samples to generate if stimuli is None. Default: 100.
        contrasts : array_like, optional
            Contrast levels to sample if generating new stimuli.
            Default: [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]

        Returns
        -------
        float
            Lower bound for input baseline parameter

        References
        ----------
        .. [1] ssn_inference_optimizer/train.ml lines 47-49

        Notes
        -----
        Este valor se usa durante el entrenamiento para parametrizar:
        β_h = input_baseline_lb + x²
        donde x es el parámetro optimizado.

        La fórmula completa es: h = α_h · (β_h + filter)^γ_h
        donde β_h debe ser suficientemente grande para evitar argumentos
        negativos en la potencia.
        """
        if stimuli is None:
            # Generar estímulos con diferentes contrastes
            if contrasts is None:
                contrasts = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]

            stimuli = []
            samples_per_contrast = n_samples // len(contrasts)

            for contrast in contrasts:
                for _ in range(samples_per_contrast):
                    stim_data = self.generate_stimulus_patch(contrast)
                    stimuli.append(stim_data['x'])

            stimuli = np.array(stimuli)

        # Calcular respuestas de filtros para todos los estímulos
        # usando W_ff que ya incluye el factor de escala 1/15
        # Ref: Echeveste et al. (2020) Supplementary Table S1:
        # "Feed-forward weights: [A A]^T / 15.0"
        filter_responses = []
        for x in stimuli:
            # W_ff @ x donde W_ff = A.T / 15.0
            filter_response = self.W_ff @ x
            filter_responses.append(filter_response)

        filter_responses = np.array(filter_responses)

        # Encontrar el mínimo global de las respuestas
        min_filter = np.min(filter_responses)

        # Calcular lower bound: garantiza que β_h + filter_response >= 0.1
        # Con W_ff = A.T / 15.0, este valor será ~0.1 (moderado)
        input_baseline_lb = 0.1 - min_filter

        return input_baseline_lb


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
