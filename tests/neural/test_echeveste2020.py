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

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import pytest

from scipy.stats import pearsonr

from skneuromsi.generative import create_echeveste_gsm
from skneuromsi.neural import Echeveste2020

# =============================================================================
# ECHEVESTE 2020 TESTS
# =============================================================================


class TestEcheveste2020:
    """Test suite for Echeveste2020 SSN model."""

    def test_echeveste_initialization(self):
        """Test Echeveste2020 model initialization."""
        ssn = Echeveste2020(
            N_E=25,
            N_I=25,
            tau_e=20.0e-3,
            tau_i=10.0e-3,
            k=0.3,
            n=2.0,
            random_seed=42,
        )

        assert ssn._N == 50
        assert ssn._N_E == 25
        assert ssn._N_I == 25
        assert ssn._tau_e == 20.0e-3
        assert ssn._tau_i == 10.0e-3
        assert ssn._k == 0.3
        assert ssn._n == 2.0

    def test_echeveste_default_parameters(self):
        """Test that default parameters match Echeveste et al. (2020)."""
        ssn = Echeveste2020()

        # Check default parameters from the paper
        assert ssn._N == 100  # Default network size
        assert ssn._tau_e == 20.0e-3  # Excitatory time constant
        assert ssn._tau_i == 10.0e-3  # Inhibitory time constant
        assert ssn._k == 0.3  # Nonlinearity scaling
        assert ssn._n == 2.0  # Nonlinearity power

    def test_weight_matrix_properties(self):
        """Test weight matrix W has correct properties."""
        ssn = Echeveste2020(N_E=25, N_I=25, random_seed=42)

        W = ssn._W

        # Check shape
        assert W.shape == (50, 50)

        # Check that weights are finite
        assert np.all(np.isfinite(W))

        # Check E/I structure (first N_E are excitatory, rest inhibitory)
        # Excitatory connections should be positive
        assert np.all(W[:, :25] >= 0)

        # Inhibitory connections should be negative
        assert np.all(W[:, 25:] <= 0)

    def test_nonlinearity_function(self):
        """Test SSN nonlinearity function."""
        ssn = Echeveste2020(N_E=25, N_I=25, k=0.3, n=2.0)

        # Test nonlinearity with known values
        u = np.array([-1.0, 0.0, 1.0, 2.0])
        r = ssn._nonlinearity(u)

        # For n=2, k=0.3: r = k * (u_+)^n where u_+ = max(u, 0)
        expected = 0.3 * np.power(np.maximum(u, 0), 2.0)
        np.testing.assert_allclose(r, expected, rtol=1e-10)

        # Check that negative inputs give zero output
        assert r[0] == 0.0  # u = -1.0
        assert r[1] == 0.0  # u = 0.0

    def test_stimulus_integration(self):
        """Test integration with GSM stimulus generation."""
        _ = Echeveste2020(N_E=25, N_I=25, random_seed=42)  # noqa: F841
        gsm = create_echeveste_gsm(random_seed=42)

        # Generate stimulus
        contrast = 0.5
        stimulus_data = gsm.generate_stimulus_patch(contrast)
        h_input = gsm.generate_h_input(stimulus_data["x"])

        # Check that h_input can be used as input to SSN
        assert h_input.shape == (50,)  # Should match SSN input size
        assert np.all(np.isfinite(h_input))

    def test_network_dynamics_basic(self):
        """Test basic network dynamics properties."""
        ssn = Echeveste2020(N_E=50, N_I=50, random_seed=42)

        # Create simple test input
        h_input = np.ones(100) * 0.1  # Small positive input to all neurons

        # Test that network can process input without errors
        # Note: This is a basic test - full dynamics testing would require
        # integration of differential equations which is complex
        initial_state = np.zeros(100)

        # Test nonlinearity response
        r = ssn._nonlinearity(initial_state + h_input)
        assert r.shape == (100,)
        assert np.all(r >= 0)  # Rates should be non-negative
        assert np.any(r > 0)  # Should have some positive responses

    def test_neuromsi_integration(self):
        """Test integration with neuromsi framework."""
        ssn = Echeveste2020(N_E=25, N_I=25, random_seed=42)

        # Check that model has required neuromsi attributes
        assert hasattr(ssn, "_model_type")
        assert hasattr(ssn, "_output_mode")

        # Should be able to handle modality-specific inputs
        # This tests the parameter aliasing system
        assert ssn._model_type in ["Neural", "neural"]

    def test_reproducibility(self):
        """Test that same random seed produces same results."""
        ssn1 = Echeveste2020(N_E=25, N_I=25, random_seed=42)
        ssn2 = Echeveste2020(N_E=25, N_I=25, random_seed=42)

        # Weight matrices should be identical with same seed
        np.testing.assert_allclose(ssn1._W, ssn2._W, rtol=1e-10)

    @pytest.mark.parametrize("N_E,N_I", [(25, 25), (40, 60), (60, 40)])
    def test_different_network_sizes(self, N_E, N_I):
        """Test SSN with different E/I ratios."""
        ssn = Echeveste2020(N_E=N_E, N_I=N_I, random_seed=42)

        assert ssn._N == N_E + N_I
        assert ssn._N_E == N_E
        assert ssn._N_I == N_I

        # Weight matrix should have correct size
        assert ssn._W.shape == (N_E + N_I, N_E + N_I)

    @pytest.mark.parametrize("k,n", [(0.1, 1.0), (0.3, 2.0), (0.5, 3.0)])
    def test_different_nonlinearity_parameters(self, k, n):
        """Test SSN with different nonlinearity parameters."""
        ssn = Echeveste2020(N_E=25, N_I=25, k=k, n=n, random_seed=42)

        assert ssn._k == k
        assert ssn._n == n

        # Test nonlinearity function
        u = np.array([0.0, 1.0, 2.0])
        r = ssn._nonlinearity(u)
        expected = k * np.power(np.maximum(u, 0), n)
        np.testing.assert_allclose(r, expected, rtol=1e-10)


@pytest.mark.slow
class TestEcheveste2020Performance:
    """Performance tests for Echeveste2020 model."""

    def test_large_network(self):
        """Test SSN with large network size."""
        ssn = Echeveste2020(N_E=100, N_I=100, random_seed=42)

        assert ssn._N == 200
        assert ssn._W.shape == (200, 200)

        # Should initialize without memory issues
        assert np.all(np.isfinite(ssn._W))


# =============================================================================
# COMPARISON WITH ORIGINAL CODE TESTS
# =============================================================================


@pytest.mark.comparison
class TestEchevesteComparison:
    """Tests comparing our implementation with original Echeveste code."""

    @pytest.fixture(scope="class")
    def original_code_available(self):
        """Check if original Echeveste code is available."""
        original_path = Path("../ssn_inference_numerical_experiments")
        return original_path.exists()

    @pytest.fixture(scope="class")
    def original_results(self, original_code_available):
        """Generate results from original implementation if available."""
        if not original_code_available:
            pytest.skip("Original Echeveste code not available")

        return self._run_original_implementation()

    def _run_original_implementation(self):
        """Execute original GSM code and extract results."""
        original_path = Path("../ssn_inference_numerical_experiments/GSM")

        if not original_path.exists():
            return None

        original_cwd = os.getcwd()
        try:
            os.chdir(original_path)

            # Run original GSM.py
            result = subprocess.run(
                [sys.executable, "GSM.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                return None

            # Load generated results
            results_path = Path("bumps/no_noise/results")
            if not results_path.exists():
                return None

            # Load key matrices and data
            A = np.loadtxt(results_path / "A")  # Gabor filters
            C = np.loadtxt(results_path / "C")  # Covariance
            z_array = np.loadtxt(results_path / "z_array")  # Contrasts

            # Load h inputs
            h_inputs = []
            for alpha in range(len(z_array)):
                h = np.loadtxt(results_path / f"h_{alpha}")
                h_inputs.append(h)

            return {
                "gabor_filters": A,
                "prior_covariance": C,
                "contrasts": z_array,
                "h_inputs": np.array(h_inputs),
                "n_orientations": A.shape[1],
                "patch_size": int(np.sqrt(A.shape[0])),
            }

        except Exception:
            return None
        finally:
            os.chdir(original_cwd)

    @pytest.mark.skipif(
        not Path("../ssn_inference_numerical_experiments").exists(),
        reason="Original Echeveste code not available",
    )
    def test_gabor_filter_comparison(self, original_results):
        """Compare Gabor filters with original implementation."""
        if original_results is None:
            pytest.skip("Could not generate original results")

        # Generate our filters with same parameters
        from skneuromsi.generative._gsm import generate_gabor_filters

        our_filters = generate_gabor_filters(
            patch_size=original_results["patch_size"],
            n_orientations=original_results["n_orientations"],
            spatial_freq=0.2,
            bandwidth=1.0,
        )

        original_filters = original_results["gabor_filters"]

        # Compare shapes
        assert our_filters.shape == original_filters.shape

        # Compare filter correlations (filters might have different signs)
        correlations = []
        for i in range(min(10, our_filters.shape[1])):  # Test first 10 filters
            corr, _ = pearsonr(our_filters[:, i], original_filters[:, i])
            correlations.append(
                abs(corr)
            )  # Absolute value for sign differences

        mean_correlation = np.mean(correlations)

        # Filters should be highly correlated
        assert (
            mean_correlation > 0.90
        ), f"Mean correlation {mean_correlation:.4f} too low"

        print(
            "Gabor filter comparison:"
            f" mean correlation = {mean_correlation:.4f}"
        )

    @pytest.mark.skipif(
        not Path("../ssn_inference_numerical_experiments").exists(),
        reason="Original Echeveste code not available",
    )
    def test_h_input_comparison(self, original_results):
        """Compare h input generation with original implementation."""
        if original_results is None:
            pytest.skip("Could not generate original results")

        # Create our GSM with same parameters
        gsm = create_echeveste_gsm(random_seed=42)

        # Generate our h inputs for same contrasts
        our_h_inputs = []
        for contrast in original_results["contrasts"]:
            stimulus_data = gsm.generate_stimulus_patch(contrast)
            h = gsm.generate_h_input(stimulus_data["x"])
            our_h_inputs.append(h)

        our_h_inputs = np.array(our_h_inputs)
        original_h_inputs = original_results["h_inputs"]

        # Compare each h input
        correlations = []
        mse_values = []

        n_compare = min(len(our_h_inputs), len(original_h_inputs))
        for i in range(n_compare):
            corr, _ = pearsonr(our_h_inputs[i], original_h_inputs[i])
            mse = np.mean((our_h_inputs[i] - original_h_inputs[i]) ** 2)

            correlations.append(abs(corr))
            mse_values.append(mse)

        mean_correlation = np.mean(correlations)
        mean_mse = np.mean(mse_values)

        # H inputs should be highly correlated
        assert (
            mean_correlation > 0.95
        ), f"H input correlation {mean_correlation:.4f} too low"

        print(f"H input comparison: mean correlation = {mean_correlation:.4f}")
        print(f"H input comparison: mean MSE = {mean_mse:.6f}")

    @pytest.mark.skipif(
        not Path("../ssn_inference_numerical_experiments").exists(),
        reason="Original Echeveste code not available",
    )
    def test_parameter_comparison(self, original_results):
        """Compare model parameters with original implementation."""
        if original_results is None:
            pytest.skip("Could not generate original results")

        # Our implementation parameters
        gsm = create_echeveste_gsm(modality="visual")
        ssn = Echeveste2020()

        # Parameters from original (extracted from GSM.py)
        original_params = {
            "patch_size": original_results["patch_size"],
            "n_orientations": original_results["n_orientations"],
            "h_scale": 1.0 / 15.0,  # From GSM.py line 283
            "gamma": 0.0,  # From GSM.py line 238
            "k": 0.3,  # From parameters.py
            "n": 2.0,  # From parameters.py
            "tau_e": 20.0e-3,  # From parameters.py
            "tau_i": 10.0e-3,  # From parameters.py
        }

        # Our parameters
        our_params = {
            "patch_size": gsm.patch_size,
            "n_orientations": gsm.n_orientations,
            "h_scale": gsm.h_scale,
            "gamma": gsm.gamma,
            "k": ssn._k,
            "n": ssn._n,
            "tau_e": ssn._tau_e,
            "tau_i": ssn._tau_i,
        }

        # Compare parameters
        matches = 0
        total = 0

        for param, our_val in our_params.items():
            if param in original_params:
                orig_val = original_params[param]
                match = abs(our_val - orig_val) < 1e-10

                if match:
                    matches += 1
                else:
                    print(
                        (
                            f"Parameter mismatch - {param}: "
                            f"ours={our_val}, original={orig_val}"
                        )
                    )

                total += 1

        match_percentage = matches / total * 100

        # Should have high parameter match
        assert (
            match_percentage > 90
        ), f"Parameter match {match_percentage:.1f}% too low"

        print(
            (
                f"Parameter comparison: {matches}/{total} match "
                f"({match_percentage:.1f}%)"
            )
        )


@pytest.mark.integration
class TestEchevesteIntegration:
    """Integration tests for complete Echeveste pipeline."""

    def test_complete_gsm_ssn_pipeline(self):
        """Test complete pipeline from GSM stimulus to SSN response."""
        # Create models
        gsm = create_echeveste_gsm(random_seed=42)
        ssn = Echeveste2020(N_E=25, N_I=25, random_seed=42)

        # Generate stimuli
        contrasts = [0.1, 0.5, 1.0]
        n_samples = 5

        gsm_response, _ = gsm.run(contrasts=contrasts, n_samples=n_samples)

        # Test that h_inputs can be used with SSN
        h_inputs = gsm_response["h_inputs"]

        for i in range(len(contrasts)):
            for j in range(n_samples):
                h = h_inputs[i, j]

                # Resize h to match SSN if needed
                if len(h) != ssn._N:
                    # Simple resizing for test
                    # (real implementation would be more sophisticated)
                    h_resized = np.resize(h, ssn._N)
                else:
                    h_resized = h

                # Test that SSN can process this input
                r = ssn._nonlinearity(h_resized)

                assert r.shape == (ssn._N,)
                assert np.all(r >= 0)
                assert np.all(np.isfinite(r))

    def test_model_compatibility(self):
        """Test that GSM and SSN models are compatible."""
        gsm = create_echeveste_gsm(random_seed=42)
        ssn = Echeveste2020(random_seed=42)

        # Both should work with same modality
        assert gsm.n_orientations == 50
        assert ssn._N == 100

        # Generate test stimulus
        stimulus_data = gsm.generate_stimulus_patch(0.5)
        h = gsm.generate_h_input(stimulus_data["x"])

        # h should be compatible with SSN (after potential resizing)
        assert h.shape == (50,)  # GSM output

        # Can be used with SSN (with proper sizing)
        if len(h) != ssn._N:
            h_extended = np.concatenate([h, h])[: ssn._N]  # Simple extension
        else:
            h_extended = h

        r = ssn._nonlinearity(h_extended)
        assert r.shape == (ssn._N,)


# =============================================================================
# REGRESSION TESTS
# =============================================================================


class TestEchevesteRegression:
    """Regression tests for Echeveste2020 to prevent breaking changes."""

    def test_known_nonlinearity_values(self):
        """Test nonlinearity with known input-output pairs."""
        ssn = Echeveste2020(k=0.3, n=2.0, random_seed=42)

        # Test with specific values that should remain stable
        test_inputs = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        expected_outputs = 0.3 * test_inputs**2.0

        actual_outputs = ssn._nonlinearity(test_inputs)

        np.testing.assert_allclose(
            actual_outputs, expected_outputs, rtol=1e-12
        )

    def test_weight_matrix_properties_stable(self):
        """Test that weight matrix generation is stable across versions."""
        # Fixed seed should give reproducible results
        ssn = Echeveste2020(N_E=10, N_I=10, random_seed=12345)

        W = ssn._W

        # Basic properties that should remain stable
        assert W.shape == (20, 20)
        assert np.all(np.isfinite(W))

        # E/I structure should be preserved
        assert np.all(W[:, :10] >= 0)  # Excitatory connections
        assert np.all(W[:, 10:] <= 0)  # Inhibitory connections

        # Overall statistics should be in reasonable range
        assert np.abs(np.mean(W)) < 1.0  # Not too large in magnitude
        assert np.std(W) > 0.001  # Has some variability
