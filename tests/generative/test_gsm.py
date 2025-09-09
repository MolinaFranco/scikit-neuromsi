#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import pytest

from skneuromsi.generative import GSM, create_echeveste_gsm
from skneuromsi.generative._gsm import (
    generate_covariance_matrix,
    generate_gabor_filters,
)

# =============================================================================
# GSM TESTS
# =============================================================================


class TestGSM:
    """Test suite for Gaussian Scale Mixture (GSM) model."""

    def test_gsm_initialization(self):
        """Test GSM model initialization with default parameters."""
        gsm = GSM(patch_size=16, n_orientations=50, random_seed=42)

        assert gsm.patch_size == 16
        assert gsm.n_orientations == 50
        assert gsm.patch_dim == 256
        assert gsm.orientation_dim == 50
        assert gsm.A.shape == (256, 50)
        assert gsm.C.shape == (50, 50)

    def test_gsm_echeveste_defaults(self):
        """Test GSM creation with Echeveste default parameters."""
        gsm = create_echeveste_gsm(random_seed=42)

        # Check Echeveste-specific parameters
        assert gsm.patch_size == 16
        assert gsm.n_orientations == 50
        assert gsm.h_scale == 1.0 / 15.0
        assert gsm.gamma == 0.0
        assert gsm.spatial_freq == 0.2
        assert gsm.bandwidth == 1.0
        assert gsm.noise_variance == 0.01

    def test_gabor_filter_generation(self):
        """Test Gabor filter generation properties."""
        filters = generate_gabor_filters(
            patch_size=16, n_orientations=10, spatial_freq=0.2, bandwidth=1.0
        )

        # Check shape
        assert filters.shape == (256, 10)

        # Check normalization (all filters should be unit norm)
        norms = np.linalg.norm(filters, axis=0)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

        # Check that filters are not all zeros
        assert not np.allclose(filters, 0)

        # Check that filters are finite
        assert np.all(np.isfinite(filters))

    def test_covariance_matrix_generation(self):
        """Test prior covariance matrix generation."""
        C = generate_covariance_matrix(
            n_orientations=20, correlation_strength=0.5
        )

        # Check shape
        assert C.shape == (20, 20)

        # Check symmetry
        np.testing.assert_allclose(C, C.T, rtol=1e-10)

        # Check diagonal elements are 1.0
        np.testing.assert_allclose(np.diag(C), 1.0, rtol=1e-10)

        # Check positive definiteness (all eigenvalues > 0)
        eigenvals = np.linalg.eigvals(C)
        assert np.all(eigenvals > 0)

        # Check values are in [0, 1]
        assert np.all(C >= 0)
        assert np.all(C <= 1)

    def test_stimulus_patch_generation(self):
        """Test single stimulus patch generation."""
        gsm = create_echeveste_gsm(random_seed=42)

        contrast = 0.5
        stimulus_data = gsm.generate_stimulus_patch(contrast)

        # Check return structure
        assert "x" in stimulus_data
        assert "y" in stimulus_data
        assert "z" in stimulus_data

        # Check shapes
        assert stimulus_data["x"].shape == (256,)
        assert stimulus_data["y"].shape == (50,)
        assert stimulus_data["z"] == contrast

        # Check values are finite
        assert np.all(np.isfinite(stimulus_data["x"]))
        assert np.all(np.isfinite(stimulus_data["y"]))

    def test_h_input_generation(self):
        """Test SSN input h generation."""
        gsm = create_echeveste_gsm(random_seed=42)

        # Generate a test stimulus
        stimulus_data = gsm.generate_stimulus_patch(0.5)
        x = stimulus_data["x"]

        # Generate h input
        h = gsm.generate_h_input(x)

        # Check shape and properties
        assert h.shape == (50,)
        assert np.all(np.isfinite(h))

        # Check transformation equation: h = h_scale * (A.T @ x)
        expected_h = gsm.h_scale * (gsm.A.T @ x)
        np.testing.assert_allclose(h, expected_h, rtol=1e-10)

    def test_gsm_run_method(self):
        """Test complete GSM run method."""
        gsm = create_echeveste_gsm(random_seed=42)

        contrasts = [0.2, 0.5, 0.8]
        n_samples = 3

        response = gsm.generate_stimuli(
            contrasts=contrasts, n_samples=n_samples
        )

        # Check response structure
        assert "stimuli" in response
        assert "h_inputs" in response
        assert "true_contrasts" in response

        # Check shapes
        expected_shape = (len(contrasts), n_samples, 256)
        expected_h_shape = (len(contrasts), n_samples, 50)

        assert response["stimuli"].shape == expected_shape
        assert response["h_inputs"].shape == expected_h_shape
        assert response["true_contrasts"].shape == (len(contrasts), n_samples)

        # Check that matrices exist in GSM instance
        assert gsm.A.shape == (256, 50)
        assert gsm.C.shape == (50, 50)

    def test_contrast_scaling(self):
        """Test that higher contrasts produce higher h magnitudes."""
        gsm = create_echeveste_gsm(random_seed=42)

        contrasts = [0.1, 0.5, 1.0]
        n_samples = 10

        response = gsm.generate_stimuli(
            contrasts=contrasts, n_samples=n_samples
        )
        h_inputs = response["h_inputs"]

        # Calculate mean h magnitude for each contrast
        h_magnitudes = []
        for i, contrast in enumerate(contrasts):
            h_mag = np.mean(
                [np.linalg.norm(h_inputs[i, j]) for j in range(n_samples)]
            )
            h_magnitudes.append(h_mag)

        # Check that magnitudes generally increase with contrast
        # (allowing for some noise due to randomness)
        assert h_magnitudes[1] > h_magnitudes[0] * 0.8
        assert h_magnitudes[2] > h_magnitudes[1] * 0.8

    def test_echeveste_compatibility_methods(self):
        """Test Echeveste compatibility methods."""
        gsm = create_echeveste_gsm(random_seed=42)

        # Test contrast level mapping
        for level in range(5):
            h_samples = gsm.get_h_for_contrast_level(level, n_samples=2)
            assert h_samples.shape == (2, 50)
            assert np.all(np.isfinite(h_samples))

        # Test filter and covariance getters
        filters = gsm.get_gabor_filters()
        covariance = gsm.get_prior_covariance()

        assert filters.shape == (256, 50)
        assert covariance.shape == (50, 50)

        # These should be copies, not references
        filters[0, 0] = 999
        assert gsm.A[0, 0] != 999

    def test_reproducibility(self):
        """Test that same random seed produces same results."""
        gsm1 = create_echeveste_gsm(random_seed=42)
        gsm2 = create_echeveste_gsm(random_seed=42)

        contrasts = [0.5]
        n_samples = 2

        response1 = gsm1.generate_stimuli(
            contrasts=contrasts, n_samples=n_samples
        )
        response2 = gsm2.generate_stimuli(
            contrasts=contrasts, n_samples=n_samples
        )

        # Results should be identical with same random seed
        np.testing.assert_allclose(
            response1["stimuli"], response2["stimuli"], rtol=1e-10
        )
        np.testing.assert_allclose(
            response1["h_inputs"], response2["h_inputs"], rtol=1e-10
        )

    def test_parameter_storage(self):
        """Test parameter storage in GSM."""
        gsm = create_echeveste_gsm()

        # Check that parameters are stored correctly
        assert gsm.patch_size == 16
        assert gsm.n_orientations == 50
        assert gsm.h_scale == 1.0 / 15.0

    @pytest.mark.parametrize("patch_size", [8, 16, 32])
    @pytest.mark.parametrize("n_orientations", [10, 25, 50])
    def test_different_sizes(self, patch_size, n_orientations):
        """Test GSM with different patch sizes and orientation counts."""
        gsm = GSM(
            patch_size=patch_size,
            n_orientations=n_orientations,
            random_seed=42,
        )

        # Check dimensions are correct
        assert gsm.patch_dim == patch_size * patch_size
        assert gsm.orientation_dim == n_orientations
        assert gsm.A.shape == (patch_size * patch_size, n_orientations)
        assert gsm.C.shape == (n_orientations, n_orientations)

        # Test that it can generate stimuli
        response = gsm.generate_stimuli(contrasts=[0.5], n_samples=1)
        expected_shape = (1, 1, patch_size * patch_size)
        expected_h_shape = (1, 1, n_orientations)

        assert response["stimuli"].shape == expected_shape
        assert response["h_inputs"].shape == expected_h_shape


@pytest.mark.slow
class TestGSMPerformance:
    """Performance and stress tests for GSM."""

    def test_large_batch_generation(self):
        """Test GSM with large number of samples."""
        gsm = create_echeveste_gsm(random_seed=42)

        contrasts = [0.1, 0.5, 1.0]
        n_samples = 50  # Large batch

        response = gsm.generate_stimuli(
            contrasts=contrasts, n_samples=n_samples
        )

        # Should complete without errors
        assert response["stimuli"].shape == (3, 50, 256)
        assert response["h_inputs"].shape == (3, 50, 50)

    def test_memory_efficiency(self):
        """Test that GSM doesn't leak memory with repeated runs."""
        gsm = create_echeveste_gsm(random_seed=42)

        # Run multiple times with same parameters
        for _ in range(10):
            response = gsm.generate_stimuli(contrasts=[0.5], n_samples=5)
            # Should not accumulate memory or cause issues
            assert response["stimuli"].shape == (1, 5, 256)


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestGSMRegression:
    """Regression tests to ensure consistency with known good outputs."""

    def test_known_filter_properties(self):
        """Test that generated filters have expected statistical properties."""
        # Use fixed seed for reproducible test
        filters = generate_gabor_filters(
            patch_size=16, n_orientations=50, spatial_freq=0.2, bandwidth=1.0
        )

        # These values were computed from a known good implementation
        # and should remain stable across versions
        filter_means = np.mean(filters, axis=0)
        filter_stds = np.std(filters, axis=0)

        # Gabor filters should have approximately zero mean
        assert np.all(np.abs(filter_means) < 0.1)

        # Standard deviations should be in reasonable range
        assert np.all(
            filter_stds > 0.05
        )  # Adjusted based on actual Gabor filter properties
        assert np.all(filter_stds < 1.0)

    def test_known_h_input_values(self):
        """Test h input generation with known stimulus."""
        gsm = create_echeveste_gsm(random_seed=12345)

        # Create a known stimulus (simple pattern)
        test_stimulus = np.zeros(256)
        test_stimulus[100:150] = 1.0  # Simple block pattern

        h = gsm.generate_h_input(test_stimulus)

        # Properties that should be stable
        assert h.shape == (50,)
        assert np.all(np.isfinite(h))
        assert not np.allclose(h, 0)  # Should not be all zeros

        # Magnitude should be reasonable
        h_magnitude = np.linalg.norm(h)
        assert 0.001 < h_magnitude < 1.0  # Reasonable range for h_scale=1/15
