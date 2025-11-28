#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data loader for Echeveste2020 trained parameters and GSM components.

This module provides access to pre-trained model parameters extracted from
the original Echeveste et al. (2020) implementation, for use in production
models without requiring external dependencies.
"""

from pathlib import Path

import numpy as np


class EchevesteDataLoader:
    """
    Loader for Echeveste2020 pre-trained model data.

    Provides access to:
    - Pre-trained Gabor filters (A matrix)
    - Prior covariance matrix (C matrix)
    - Optimized SSN connectivity parameters
    - Noise covariance matrix

    This data represents the trained/optimized parameters from the original
    paper, ready for use in production models.
    """

    def __init__(self):
        """Initialize the data loader."""
        self.data_path = Path(__file__).parent / "echeveste2020"

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Echeveste2020 data directory not found at {self.data_path}"
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

    # SSN Data Loading Methods
    def load_ssn_connectivity_parameters(self):
        """
        Load optimized SSN connectivity parameters.

        Returns
        -------
        params : dict
            Dictionary containing the 8 optimized connectivity parameters:
            - a_EE, a_EI, a_IE, a_II: Connection amplitudes
            - d_EE, d_EI, d_IE, d_II: Connection widths
        """
        param_mapping = {
            "a_EE": "w_ee_height_learn",
            "a_EI": "w_ei_height_learn",
            "a_IE": "w_ie_height_learn",
            "a_II": "w_ii_height_learn",
            "d_EE": "w_ee_width_learn",
            "d_EI": "w_ei_width_learn",
            "d_IE": "w_ie_width_learn",
            "d_II": "w_ii_width_learn",
        }

        params = {}
        for param_key, file_name in param_mapping.items():
            params[param_key] = float(np.loadtxt(self.data_path / file_name))

        return params

    def load_input_transformation_parameters(self):
        """
        Load optimized input transformation parameters.

        These parameters control the nonlinear transformation of the
        feed-forward input h:
            h_transformed = α_h * (h_raw + β_h)^γ_h

        Returns
        -------
        params : dict
            Dictionary containing the 3 optimized input transformation
            parameters:
            - alpha_h (α_h): Input scaling = 1.96 (optimized)
            - beta_h (β_h): Input baseline = 0.10 (optimized)
            - gamma_h (γ_h): Input power = 2.03 (optimized)

        References
        ----------
        .. [1] Echeveste et al. (2020), Supplementary Table S1
        .. [2] ssn_inference_optimizer/train.ml lines 134-136
        .. [3] ssn_inference_optimizer/objective.ml lines 410-413

        Notes
        -----
        These parameters are optimized jointly with the connectivity
        matrix and noise covariance in the original Echeveste code.
        The transformation is applied to the raw filter responses
        before passing them to the SSN.
        """
        return {
            "alpha_h": float(np.loadtxt(self.data_path / "input_scaling")),
            "beta_h": float(np.loadtxt(self.data_path / "input_baseline")),
            "gamma_h": float(np.loadtxt(self.data_path / "input_nl_pow")),
        }

    def load_noise_variance_parameters(self):
        """
        Load optimized noise variance parameters.

        These parameters define the spatial structure of noise covariance:
        - var_e: Variance for excitatory neurons
        - var_i: Variance for inhibitory neurons
        - var_width: Width of spatial correlation (Gaussian)
        - rho: Cross-correlation between E and I populations

        Returns
        -------
        params : dict
            Dictionary containing:
            - var_e: float, variance for E neurons
            - var_i: float, variance for I neurons
            - var_width: float, spatial correlation width
            - rho: float, E-I cross-correlation

        References
        ----------
        .. [1] ssn_inference_optimizer/objective.ml lines 290-305
               Shows how sigma_eta is constructed from these parameters

        Notes
        -----
        These 4 parameters are among the 15 trained parameters in the
        original Echeveste code. They define the structure of the noise
        covariance matrix sigma_eta according to:

        sigma_eta[i,j] = sqrt(var_i * var_j) * rho^{cross} *
                         exp((cos(θ_i - θ_j) - 1) / var_width²) + ε*δ_ij

        where:
        - var_i, var_j are var_e or var_i depending on neuron type
        - rho^{cross} = 1 if i,j same type, rho if different types
        - ε = 0.01 added to diagonal for numerical stability
        """
        return {
            "var_e": float(np.loadtxt(self.data_path / "var_e_learn")),
            "var_i": float(np.loadtxt(self.data_path / "var_i_learn")),
            "var_width": float(np.loadtxt(self.data_path / "var_width_learn")),
            "rho": float(np.loadtxt(self.data_path / "rho_learn")),
        }

    def build_noise_covariance_from_parameters(
        self, var_e, var_i, var_width, rho, n_orientations=50, epsilon=0.01
    ):
        """
        Construct noise covariance matrix from parameters.

        Implements the construction formula from objective.ml:290-305.

        Parameters
        ----------
        var_e : float
            Variance for excitatory neurons
        var_i : float
            Variance for inhibitory neurons
        var_width : float
            Width parameter for spatial Gaussian correlation
        rho : float
            Cross-correlation coefficient between E and I
        n_orientations : int, optional
            Number of orientation channels (default: 50)
        epsilon : float, optional
            Regularization added to diagonal (default: 0.01)

        Returns
        -------
        sigma_eta : np.ndarray, shape (2*n_orientations, 2*n_orientations)
            Constructed noise covariance matrix

        References
        ----------
        .. [1] ssn_inference_optimizer/objective.ml lines 290-305
        """
        m = n_orientations
        n = 2 * m  # Total neurons (E + I)

        sigma_eta = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # Determinar varianza según tipo de neurona
                xi = var_e if i < m else var_i
                xj = var_e if j < m else var_i

                # Producto geométrico
                x = np.sqrt(xi * xj)

                # Aplicar rho para conexiones cruzadas E-I
                if not ((i < m and j < m) or (i >= m and j >= m)):
                    x = rho * x

                # Calcular distancia angular
                i_mod = i % m
                j_mod = j % m
                theta_i = 2.0 * np.pi * i_mod / m
                theta_j = 2.0 * np.pi * j_mod / m
                cos_diff_minus_one = np.cos(theta_i - theta_j) - 1.0

                # Aplicar exponencial gaussiana
                sigma_eta[i, j] = x * np.exp(
                    cos_diff_minus_one / (var_width**2)
                )

        # Agregar epsilon a la diagonal
        for i in range(n):
            sigma_eta[i, i] += epsilon

        return sigma_eta

    def load_noise_covariance(self, fallback_to_construction=True):
        """
        Load optimized noise covariance matrix.

        Attempts to load pre-computed sigma_eta_learn. If not found and
        fallback_to_construction=True, constructs it from variance
        parameters.

        Parameters
        ----------
        fallback_to_construction : bool, optional
            If True and sigma_eta_learn doesn't exist, construct from
            var_e, var_i, var_width, rho (default: True)

        Returns
        -------
        Sigma_eta : np.ndarray, shape (100, 100)
            Optimized noise covariance matrix for SSN inference

        Raises
        ------
        FileNotFoundError
            If sigma_eta_learn doesn't exist and fallback_to_construction
            is False, or if variance parameters are missing
        """
        sigma_eta_path = self.data_path / "sigma_eta_learn"

        if sigma_eta_path.exists():
            return np.loadtxt(sigma_eta_path)

        if not fallback_to_construction:
            raise FileNotFoundError(
                f"sigma_eta_learn not found at {sigma_eta_path} and "
                f"fallback_to_construction=False"
            )

        # Intentar construir desde parámetros
        try:
            params = self.load_noise_variance_parameters()
            sigma_eta = self.build_noise_covariance_from_parameters(
                var_e=params["var_e"],
                var_i=params["var_i"],
                var_width=params["var_width"],
                rho=params["rho"],
            )
            return sigma_eta
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"sigma_eta_learn not found at {sigma_eta_path} and "
                f"variance parameters (var_e_learn, var_i_learn, "
                f"var_width_learn, rho_learn) are also missing. "
                f"Cannot construct noise covariance matrix."
            ) from e

    def load_connectivity_matrix(self):
        """
        Load pre-computed connectivity matrix.

        This method loads the exact w_learn matrix from the original
        Echeveste code. Construction from parameters should be done
        using the Echeveste2020.build_connectivity_matrix() method,
        not in the data loader.

        Returns
        -------
        W : np.ndarray, shape (100, 100)
            Connectivity matrix

        Raises
        ------
        FileNotFoundError
            If w_learn file doesn't exist

        Notes
        -----
        The data loader is responsible only for loading pre-computed
        data, not for constructing matrices. If you need to build W
        from parameters, use:
            ssn = Echeveste2020(...)
            W = ssn.build_connectivity_matrix(connectivity_params)
        """
        w_learn_path = self.data_path / "w_learn"

        if not w_learn_path.exists():
            raise FileNotFoundError(
                f"w_learn not found at {w_learn_path}. "
                f"To construct connectivity matrix from parameters, use "
                f"Echeveste2020.build_connectivity_matrix() instead."
            )

        return np.loadtxt(w_learn_path)

    def load_exact_connectivity_matrix(self):
        """
        Load the exact connectivity matrix from the original Echeveste code.

        This is the pre-computed w_learn matrix that avoids numerical
        differences that can cause instability in the SSN dynamics.

        Returns
        -------
        W_exact : np.ndarray, shape (100, 100)
            Exact connectivity matrix from original Echeveste2020 code

        Notes
        -----
        This method is deprecated in favor of load_connectivity_matrix()
        which has fallback support.
        """
        return np.loadtxt(self.data_path / "w_learn")

    # Convenience Methods
    def load_all_gsm_data(self):
        """
        Load all GSM-related data.

        Returns
        -------
        dict
            Dictionary with keys: 'gabor_filters', 'prior_covariance'
        """
        return {
            "gabor_filters": self.load_gabor_filters(),
            "prior_covariance": self.load_prior_covariance(),
        }

    def load_all_ssn_data(self):
        """
        Load all SSN-related data.

        Returns
        -------
        dict
            Dictionary with keys: 'connectivity_params', 'noise_covariance',
            'input_transformation_params'
        """
        return {
            "connectivity_params": self.load_ssn_connectivity_parameters(),
            "noise_covariance": self.load_noise_covariance(),
            "input_transformation_params": (
                self.load_input_transformation_parameters()
            ),
        }

    def load_all_data(self):
        """
        Load all available pre-trained data.

        Returns
        -------
        dict
            Complete dataset with all GSM and SSN parameters
        """
        return {**self.load_all_gsm_data(), **self.load_all_ssn_data()}

    # Validation Methods
    def validate_data_integrity(self):
        """
        Validate that all data files exist and have expected properties.

        Returns
        -------
        bool
            True if all data validates successfully
        """
        try:
            # Test GSM data
            A = self.load_gabor_filters()
            if A.shape != (256, 50):
                return False

            C = self.load_prior_covariance()
            if C.shape != (50, 50):
                return False
            if not np.allclose(C, C.T, rtol=1e-10):  # Should be symmetric
                return False

            # Test SSN data
            params = self.load_ssn_connectivity_parameters()
            expected_params = [
                "a_EE",
                "a_EI",
                "a_IE",
                "a_II",
                "d_EE",
                "d_EI",
                "d_IE",
                "d_II",
            ]
            for param in expected_params:
                if param not in params:
                    return False
                if not isinstance(params[param], (int, float)):
                    return False

            Sigma_eta = self.load_noise_covariance()
            if Sigma_eta.shape != (100, 100):
                return False

            # Test input transformation parameters
            input_params = self.load_input_transformation_parameters()
            expected_input_params = ["alpha_h", "beta_h", "gamma_h"]
            for param in expected_input_params:
                if param not in input_params:
                    return False
                if not isinstance(input_params[param], (int, float)):
                    return False

            return True

        except Exception:
            return False

    def get_data_info(self):
        """
        Get information about the loaded data.

        Returns
        -------
        dict
            Information about data shapes and parameters
        """
        try:
            A = self.load_gabor_filters()
            C = self.load_prior_covariance()
            params = self.load_ssn_connectivity_parameters()
            Sigma_eta = self.load_noise_covariance()
            input_params = self.load_input_transformation_parameters()

            total_params = len(params) + len(input_params)

            return {
                "gabor_filters_shape": A.shape,
                "prior_covariance_shape": C.shape,
                "connectivity_parameters": list(params.keys()),
                "input_transformation_parameters": list(input_params.keys()),
                "noise_covariance_shape": Sigma_eta.shape,
                "data_source": "Echeveste et al. (2020), Nature Neuroscience",
                "total_parameters": total_params,
            }
        except Exception as e:
            return {"error": str(e)}


# Convenience function for easy access
def load_echeveste_data():
    """
    Convenience function to create and return a data loader instance.

    Returns
    -------
    EchevesteDataLoader
        Initialized data loader
    """
    return EchevesteDataLoader()
