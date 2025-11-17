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

    def load_noise_covariance(self):
        """
        Load optimized noise covariance matrix.

        Returns
        -------
        Sigma_eta : np.ndarray, shape (100, 100)
            Optimized noise covariance matrix for SSN inference
        """
        return np.loadtxt(self.data_path / "sigma_eta_learn")

    def load_exact_connectivity_matrix(self):
        """
        Load the exact connectivity matrix from the original Echeveste code.

        This is the pre-computed w_learn matrix that avoids numerical
        differences that can cause instability in the SSN dynamics.

        Returns
        -------
        W_exact : np.ndarray, shape (100, 100)
            Exact connectivity matrix from original Echeveste2020 code
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
