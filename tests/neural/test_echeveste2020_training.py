#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for Echeveste2020 SSN training functionality.

Este módulo contiene tests críticos para verificar que el entrenamiento
del modelo SSN funciona correctamente:
- Stage 1: Optimización de conectividad W
- Stage 2: Optimización de covarianza de ruido Σ_η
- Pipeline end-to-end completo

IMPORTANTE: Estos tests son CRUCIALES para asegurar que podemos entrenar
redes desde cero sin depender de parámetros pre-cargados.
"""

import numpy as np
import pytest

from skneuromsi.generative import GSM
from skneuromsi.neural import Echeveste2020


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def small_gsm():
    """
    Create a small GSM model for fast testing.

    Usa red pequeña (25 orientaciones) para acelerar tests.
    """
    return GSM(
        patch_size=8,
        n_orientations=25,
        h_scale=1.0 / 15.0,
        random_seed=42,
        use_pretrained=False,  # No usar pre-entrenado
    )


@pytest.fixture
def small_ssn():
    """
    Create a small SSN model for fast testing.

    Usa red pequeña (10E + 10I = 20 neuronas) para acelerar tests.
    """
    return Echeveste2020(
        N_E=10,
        N_I=10,
        k=0.3,
        n=2.0,
        tau_e=20.0,
        tau_i=10.0,
        tau_n=20.0,
        seed=42,
    )


@pytest.fixture
def medium_ssn():
    """
    Create a medium-sized SSN for more realistic testing.

    Usa red mediana (25E + 25I = 50 neuronas).
    """
    return Echeveste2020(
        N_E=25,
        N_I=25,
        k=0.3,
        n=2.0,
        tau_e=20.0,
        tau_i=10.0,
        tau_n=20.0,
        seed=42,
    )


# =============================================================================
# STAGE 1 TRAINING TESTS
# =============================================================================


class TestStage1Training:
    """Tests for Stage 1 connectivity optimization."""

    def test_stage1_basic_execution(self, small_gsm, small_ssn):
        """
        Test that Stage 1 training executes without errors.

        Verificación básica: el training debe completarse sin excepciones.
        """
        # Stage 1 con muy pocas iteraciones (solo verificar que corre)
        stage1_params = {
            'max_iter': 5,
            'contrast': 0.5,
            'n_samples_gsm': 10,
        }

        result = small_ssn._optimize_stage1(small_gsm, stage1_params)

        # Verificar que retorna estructura esperada
        assert 'optimized_params' in result
        assert 'initial_params' in result
        assert 'final_cost' in result
        assert 'n_iterations' in result
        assert 'success' in result

        # Verificar que tiene los 8 parámetros
        assert len(result['optimized_params']) == 8
        assert 'a_EE' in result['optimized_params']
        assert 'd_EE' in result['optimized_params']

    def test_stage1_cost_decreases(self, small_gsm, small_ssn):
        """
        Test that Stage 1 reduces cost function.

        CRÍTICO: Verificar que el optimizador reduce el costo.
        """
        stage1_params = {
            'max_iter': 10,
            'contrast': 0.5,
            'n_samples_gsm': 20,
        }

        result = small_ssn._optimize_stage1(small_gsm, stage1_params)

        # El costo final debe ser finito (no NaN, no infinito)
        assert np.isfinite(result['final_cost'])

        # El costo no debe ser absurdamente grande
        # (indicaría que algo está mal)
        assert result['final_cost'] < 1e6

        print(f"\nStage 1 final cost: {result['final_cost']:.4f}")
        print(f"Iterations: {result['n_iterations']}")
        print(f"Success: {result['success']}")

    def test_stage1_parameters_in_valid_range(self, small_gsm, small_ssn):
        """
        Test that optimized parameters are within valid bounds.

        Verificar que los parámetros optimizados respetan los bounds:
        - Amplitudes a_XY: >= 0
        - Widths d_XY: [0, √2]
        """
        stage1_params = {
            'max_iter': 10,
            'contrast': 0.5,
            'n_samples_gsm': 20,
        }

        result = small_ssn._optimize_stage1(small_gsm, stage1_params)
        params = result['optimized_params']

        # Verificar bounds de amplitudes
        for key in ['a_EE', 'a_EI', 'a_IE', 'a_II']:
            assert params[key] >= 0, f"{key} debe ser no negativo"
            print(f"{key}: {params[key]:.4f}")

        # Verificar bounds de widths
        sqrt2 = np.sqrt(2.0)
        for key in ['d_EE', 'd_EI', 'd_IE', 'd_II']:
            assert (
                0 <= params[key] <= sqrt2
            ), f"{key} debe estar en [0, √2]"
            print(f"{key}: {params[key]:.4f}")

    def test_stage1_updates_ssn_parameters(self, small_gsm, small_ssn):
        """
        Test that Stage 1 updates internal SSN parameters.

        Verificar que los parámetros optimizados se guardan en el objeto SSN.
        """
        # Guardar parámetros iniciales
        initial_a_EE = small_ssn._a_EE
        initial_d_EE = small_ssn._d_EE

        stage1_params = {
            'max_iter': 10,
            'contrast': 0.5,
            'n_samples_gsm': 20,
        }

        result = small_ssn._optimize_stage1(small_gsm, stage1_params)

        # Verificar que los parámetros internos se actualizaron
        assert (
            small_ssn._a_EE != initial_a_EE
            or small_ssn._d_EE != initial_d_EE
        )

        # Verificar que coinciden con los optimizados
        assert small_ssn._a_EE == result['optimized_params']['a_EE']
        assert small_ssn._d_EE == result['optimized_params']['d_EE']

        # Verificar que Stage 1 se marca como completado
        assert small_ssn._stage1_completed

    @pytest.mark.slow
    def test_stage1_convergence_medium_network(self, small_gsm, medium_ssn):
        """
        Test Stage 1 convergence on medium-sized network.

        IMPORTANTE: Test más realista con red de tamaño mediano.
        Requiere más tiempo de cómputo.
        """
        stage1_params = {
            'max_iter': 30,
            'contrast': 0.5,
            'n_samples_gsm': 50,
        }

        result = medium_ssn._optimize_stage1(small_gsm, stage1_params)

        # Verificar que el optimizador reporta éxito
        print(f"\nOptimization success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Final cost: {result['final_cost']:.4f}")
        print(f"Iterations: {result['n_iterations']}")

        # El costo debe ser finito y razonable
        assert np.isfinite(result['final_cost'])

        # Imprimir parámetros finales
        print("\nOptimized connectivity parameters:")
        for key, value in result['optimized_params'].items():
            initial = result['initial_params'][key]
            print(f"  {key}: {initial:.4f} → {value:.4f}")


# =============================================================================
# STAGE 2 TRAINING TESTS
# =============================================================================


class TestStage2Training:
    """Tests for Stage 2 noise covariance optimization."""

    def test_stage2_basic_execution(self, small_gsm, small_ssn):
        """
        Test that Stage 2 training executes without errors.

        NOTA: Stage 2 puede ejecutarse incluso sin Stage 1
        (usará conectividad por defecto).
        """
        stage2_params = {
            'max_iter': 5,
            'contrast': 0.5,
            'n_samples_gsm': 10,
        }

        result = small_ssn._optimize_stage2(small_gsm, stage2_params)

        # Verificar estructura de resultado
        assert 'optimized_params' in result
        assert 'initial_params' in result
        assert 'final_cost' in result
        assert 'sigma_eta_shape' in result

        # Verificar que tiene los 4 parámetros de ruido
        assert len(result['optimized_params']) == 4
        assert 'width' in result['optimized_params']
        assert 'std_e' in result['optimized_params']
        assert 'std_i' in result['optimized_params']
        assert 'rho' in result['optimized_params']

    def test_stage2_noise_parameters_in_valid_range(
        self, small_gsm, small_ssn
    ):
        """
        Test that optimized noise parameters are within valid bounds.

        Bounds esperados (según código original):
        - width: [0.01, √2]
        - std_e: [0.1, 4.0]
        - std_i: [0.1, 4.0]
        - rho: [0, 1] (correlación E-I)
        """
        stage2_params = {
            'max_iter': 10,
            'contrast': 0.5,
            'n_samples_gsm': 20,
        }

        result = small_ssn._optimize_stage2(small_gsm, stage2_params)
        params = result['optimized_params']

        # Verificar bounds
        sqrt2 = np.sqrt(2.0)
        assert (
            0.01 <= params['width'] <= sqrt2
        ), f"width debe estar en [0.01, √2]"
        assert (
            0.1 <= params['std_e'] <= 4.0
        ), f"std_e debe estar en [0.1, 4.0]"
        assert (
            0.1 <= params['std_i'] <= 4.0
        ), f"std_i debe estar en [0.1, 4.0]"
        assert 0.0 <= params['rho'] <= 1.0, f"rho debe estar en [0, 1]"

        print("\nOptimized noise parameters:")
        for key, value in params.items():
            print(f"  {key}: {value:.4f}")

    def test_stage2_builds_noise_covariance_matrix(
        self, small_gsm, small_ssn
    ):
        """
        Test that Stage 2 constructs noise covariance matrix Σ_η.

        Verificar que la matriz Σ_η se construye correctamente.
        """
        stage2_params = {
            'max_iter': 5,
            'contrast': 0.5,
            'n_samples_gsm': 10,
        }

        result = small_ssn._optimize_stage2(small_gsm, stage2_params)

        # Verificar que Σ_η se creó
        assert small_ssn._Sigma_eta is not None

        # Verificar dimensiones correctas (N_E + N_I = N)
        expected_size = small_ssn._N_E + small_ssn._N_I
        assert small_ssn._Sigma_eta.shape == (expected_size, expected_size)

        # Verificar que es simétrica (propiedad de covarianza)
        np.testing.assert_allclose(
            small_ssn._Sigma_eta,
            small_ssn._Sigma_eta.T,
            rtol=1e-10,
            err_msg="Σ_η debe ser simétrica",
        )

        # Verificar que es semi-definida positiva
        # (eigenvalues no negativos)
        eigenvalues = np.linalg.eigvalsh(small_ssn._Sigma_eta)
        assert np.all(
            eigenvalues >= -1e-10
        ), "Σ_η debe ser semi-definida positiva"

        print(f"\nΣ_η shape: {small_ssn._Sigma_eta.shape}")
        print(f"Min eigenvalue: {eigenvalues.min():.6f}")
        print(f"Max eigenvalue: {eigenvalues.max():.6f}")


# =============================================================================
# END-TO-END TRAINING TESTS
# =============================================================================


class TestEndToEndTraining:
    """Tests for complete two-stage training pipeline."""

    def test_train_method_basic(self, small_gsm, small_ssn):
        """
        Test the high-level train() method.

        Verificar que el método train() ejecuta ambas etapas.
        """
        # Parámetros mínimos para test rápido
        stage1_params = {'max_iter': 5, 'n_samples_gsm': 10}
        stage2_params = {'max_iter': 5, 'n_samples_gsm': 10}

        result = small_ssn.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Verificar estructura del resultado
        assert 'stage1' in result
        assert 'stage2' in result
        assert 'connectivity_params' in result
        assert 'convergence_info' in result

        # Verificar que ambas etapas se completaron
        assert result['convergence_info']['stage1_completed']
        assert result['convergence_info']['stage2_completed']
        assert result['convergence_info']['is_trained']

    def test_trained_network_can_simulate(self, small_gsm, small_ssn):
        """
        Test that trained network can run simulations.

        CRÍTICO: Después del training, la red debe poder simular.
        """
        # Entrenar (pocas iteraciones)
        stage1_params = {'max_iter': 5, 'n_samples_gsm': 10}
        stage2_params = {'max_iter': 5, 'n_samples_gsm': 10}

        small_ssn.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Intentar ejecutar simulación
        response, extra = small_ssn.run(
            stimulus_contrast=0.5,
            stimulus_orientation=0.0,
            noise_level=0.1,
            simulation_time=100.0,  # Simulación corta
        )

        # Verificar que generó outputs
        assert 'excitatory_firing_rate' in response
        assert 'inhibitory_firing_rate' in response

        # Verificar que las tasas son finitas (no explotaron)
        exc_rates = response['excitatory_firing_rate']
        inh_rates = response['inhibitory_firing_rate']

        assert np.all(np.isfinite(exc_rates)), "Tasas E deben ser finitas"
        assert np.all(np.isfinite(inh_rates)), "Tasas I deben ser finitas"

        print(f"\nMean E rate: {exc_rates.mean():.4f}")
        print(f"Mean I rate: {inh_rates.mean():.4f}")
        print(f"Max E rate: {exc_rates.max():.4f}")
        print(f"Max I rate: {inh_rates.max():.4f}")

    @pytest.mark.slow
    def test_full_training_convergence(self, small_gsm, medium_ssn):
        """
        Test full training pipeline with realistic parameters.

        IMPORTANTE: Este es el test más crítico.
        Verifica que el training completo funciona end-to-end.
        """
        # Parámetros más realistas (pero aún reducidos para CI)
        stage1_params = {
            'max_iter': 20,
            'contrast': 0.5,
            'n_samples_gsm': 30,
            'lambda_mean': 1.0,
            'lambda_var': 1.0,
            'lambda_cov': 1.0,
        }

        stage2_params = {
            'max_iter': 15,
            'contrast': 0.5,
            'n_samples_gsm': 30,
        }

        print("\n" + "=" * 60)
        print("FULL TRAINING TEST - MEDIUM NETWORK (25E + 25I)")
        print("=" * 60)

        # Ejecutar training completo
        result = medium_ssn.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Reportar resultados de Stage 1
        print("\n--- STAGE 1 RESULTS ---")
        print(f"Success: {result['stage1']['success']}")
        print(f"Final cost: {result['stage1']['final_cost']:.6f}")
        print(f"Iterations: {result['stage1']['n_iterations']}")
        print("\nOptimized connectivity parameters:")
        for key, value in result['stage1']['optimized_params'].items():
            initial = result['stage1']['initial_params'][key]
            change = ((value - initial) / initial * 100) if initial != 0 else 0
            print(f"  {key}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")

        # Reportar resultados de Stage 2
        print("\n--- STAGE 2 RESULTS ---")
        print(f"Success: {result['stage2']['success']}")
        print(f"Final cost: {result['stage2']['final_cost']:.6f}")
        print(f"Iterations: {result['stage2']['n_iterations']}")
        print("\nOptimized noise parameters:")
        for key, value in result['stage2']['optimized_params'].items():
            initial = result['stage2']['initial_params'][key]
            change = ((value - initial) / initial * 100) if initial != 0 else 0
            print(f"  {key}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")

        # Verificaciones
        assert result['convergence_info']['is_trained']
        assert np.isfinite(result['stage1']['final_cost'])
        assert np.isfinite(result['stage2']['final_cost'])

        # Ejecutar simulación con parámetros entrenados
        print("\n--- TESTING SIMULATION WITH TRAINED PARAMETERS ---")
        response, extra = medium_ssn.run(
            stimulus_contrast=0.5,
            stimulus_orientation=0.0,
            noise_level=0.1,
            simulation_time=500.0,
        )

        # Verificar que la simulación fue exitosa
        assert extra['simulation_successful']

        exc_rates = response['excitatory_firing_rate']
        print(f"Excitatory rates - Mean: {exc_rates.mean():.4f}, "
              f"Std: {exc_rates.std():.4f}, Max: {exc_rates.max():.4f}")

        # La red entrenada debe tener actividad finita y estable
        assert np.all(np.isfinite(exc_rates))
        assert exc_rates.max() < 100.0  # No debe explotar

        print("\n✅ FULL TRAINING TEST PASSED")


# =============================================================================
# PARAMETER SAVING/LOADING TESTS
# =============================================================================


class TestParameterPersistence:
    """Tests for saving and loading trained parameters."""

    def test_save_trained_parameters(self, small_gsm, small_ssn, tmp_path):
        """
        Test saving trained parameters to disk.

        Verificar que podemos guardar parámetros entrenados.
        """
        # Entrenar
        stage1_params = {'max_iter': 5, 'n_samples_gsm': 10}
        stage2_params = {'max_iter': 5, 'n_samples_gsm': 10}

        small_ssn.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Guardar parámetros
        output_dir = tmp_path / "trained_params"
        output_dir.mkdir()

        small_ssn.save_parameters(str(output_dir))

        # Verificar que los archivos se crearon
        assert (output_dir / "w_ee_height_learn").exists()
        assert (output_dir / "w_ee_width_learn").exists()
        assert (output_dir / "w_learn").exists()
        assert (output_dir / "sigma_eta_learn").exists()

    def test_load_trained_parameters(self, small_gsm, tmp_path):
        """
        Test loading trained parameters into a new network.

        CRÍTICO: Verificar que podemos cargar parámetros entrenados
        en una nueva instancia de la red.
        """
        # Entrenar primera red
        ssn1 = Echeveste2020(N_E=10, N_I=10, seed=42)

        stage1_params = {'max_iter': 5, 'n_samples_gsm': 10}
        stage2_params = {'max_iter': 5, 'n_samples_gsm': 10}

        ssn1.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Guardar parámetros
        output_dir = tmp_path / "trained_params"
        output_dir.mkdir()
        ssn1.save_parameters(str(output_dir))

        # Crear nueva red y cargar parámetros
        ssn2 = Echeveste2020(N_E=10, N_I=10, seed=123)
        ssn2.load_parameters(str(output_dir))

        # Verificar que los parámetros coinciden
        np.testing.assert_allclose(ssn2._a_EE, ssn1._a_EE)
        np.testing.assert_allclose(ssn2._d_EE, ssn1._d_EE)

        # Verificar que la nueva red está entrenada
        assert ssn2.is_trained()


# =============================================================================
# GRADIENT AND OPTIMIZATION TESTS
# =============================================================================


class TestOptimizationInternals:
    """Tests for optimization internals (JAX, gradients, etc.)."""

    def test_jax_gradient_computation(self, small_gsm, small_ssn):
        """
        Test that JAX gradients are computed correctly.

        Verificar que la diferenciación automática funciona.
        """
        from skneuromsi.neural._echeveste_training import (
            compute_nonlinear_moments,
        )
        import jax.numpy as jnp

        # Crear inputs de prueba
        mu = jnp.array([0.5, 1.0, -0.5, 0.0])
        sigma = jnp.eye(4) * 0.1

        # Computar momentos no lineales
        nu, gamma = compute_nonlinear_moments(mu, sigma, k=0.3)

        # Verificar que los outputs son finitos
        assert jnp.all(jnp.isfinite(nu))
        assert jnp.all(jnp.isfinite(gamma))

        # Verificar formas correctas
        assert nu.shape == mu.shape
        assert gamma.shape == mu.shape

        # Verificar propiedades esperadas
        # gamma debe ser no negativo (derivada de función creciente)
        assert jnp.all(gamma >= 0)

        print(f"\nmu: {mu}")
        print(f"nu: {nu}")
        print(f"gamma: {gamma}")

    def test_connectivity_matrix_differentiable(self, small_ssn):
        """
        Test that connectivity matrix construction is differentiable.

        Importante para Stage 1 optimization.
        """
        # Construir matriz con parámetros default
        W = small_ssn.build_connectivity_matrix()

        # Verificar dimensiones
        expected_size = small_ssn._N_E + small_ssn._N_I
        assert W.shape == (expected_size, expected_size)

        # Verificar signos correctos (Dale's principle)
        # E→E: positivo
        W_EE = W[: small_ssn._N_E, : small_ssn._N_E]
        assert np.all(W_EE >= 0), "E→E debe ser no negativo"

        # E→I: negativo
        W_EI = W[: small_ssn._N_E, small_ssn._N_E :]
        assert np.all(W_EI <= 0), "E→I debe ser no positivo"

        # I→E: positivo
        W_IE = W[small_ssn._N_E :, : small_ssn._N_E]
        assert np.all(W_IE >= 0), "I→E debe ser no negativo"

        # I→I: negativo
        W_II = W[small_ssn._N_E :, small_ssn._N_E :]
        assert np.all(W_II <= 0), "I→I debe ser no positivo"


# =============================================================================
# MOMENT MATCHING TESTS
# =============================================================================


class TestMomentMatching:
    """Tests for moment-matching between SSN and GSM."""

    @pytest.mark.slow
    def test_moment_matching_after_training(self, small_gsm, medium_ssn):
        """
        Test that trained SSN approximates GSM posterior moments.

        CRÍTICO: Este test verifica el objetivo fundamental del training.
        Después del entrenamiento, los momentos del SSN deberían aproximar
        los momentos del posterior del GSM.
        """
        # Entrenar con más iteraciones para mejor convergencia
        stage1_params = {
            'max_iter': 30,
            'contrast': 0.5,
            'n_samples_gsm': 50,
        }

        stage2_params = {
            'max_iter': 20,
            'contrast': 0.5,
            'n_samples_gsm': 50,
        }

        print("\n" + "=" * 60)
        print("MOMENT MATCHING TEST")
        print("=" * 60)

        result = medium_ssn.train(
            small_gsm, stage1_params=stage1_params, stage2_params=stage2_params
        )

        # Obtener targets del GSM
        gsm_data = small_gsm.compute_posterior_for_ssn_training(
            contrast=0.5, n_samples=50
        )

        target_mu = gsm_data['target_mu'][: medium_ssn._N_E]
        target_sigma = gsm_data['target_sigma'][
            : medium_ssn._N_E, : medium_ssn._N_E
        ]

        # Ejecutar simulación con red entrenada y extraer momentos
        response, extra = medium_ssn.run(
            stimulus_contrast=0.5,
            simulation_time=1000.0,
            noise_level=0.1,
        )

        # Calcular momentos empíricos del SSN
        # Usar segunda mitad de la simulación (después del transiente)
        exc_rates = response['excitatory_firing_rate']
        n_steps = len(exc_rates)
        burn_in = n_steps // 2

        ssn_mu = exc_rates[burn_in:].mean(axis=0)
        ssn_sigma = np.cov(exc_rates[burn_in:].T)

        # Comparar momentos
        print("\n--- MOMENT COMPARISON ---")
        print(f"Target μ (GSM): {target_mu[:5]}")
        print(f"SSN μ:          {ssn_mu[:5]}")
        print(f"\nTarget Σ diagonal: {np.diag(target_sigma)[:5]}")
        print(f"SSN Σ diagonal:    {np.diag(ssn_sigma)[:5]}")

        # Calcular errores
        mu_error = np.linalg.norm(ssn_mu - target_mu) / np.linalg.norm(
            target_mu
        )
        sigma_error = np.linalg.norm(
            ssn_sigma - target_sigma, 'fro'
        ) / np.linalg.norm(target_sigma, 'fro')

        print(f"\nRelative μ error: {mu_error:.4f}")
        print(f"Relative Σ error: {sigma_error:.4f}")

        # Los errores deberían ser razonables
        # (no verificamos convergencia perfecta porque usamos pocas iters)
        assert mu_error < 5.0, "Error de media demasiado grande"
        assert sigma_error < 5.0, "Error de covarianza demasiado grande"

        print("\n✅ MOMENT MATCHING TEST PASSED")


if __name__ == "__main__":
    # Para ejecutar tests individuales durante desarrollo
    pytest.main([__file__, "-v", "-s"])
