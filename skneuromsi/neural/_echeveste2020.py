#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

from dataclasses import dataclass

import brainpy as bp

import numpy as np
from scipy import stats

from ..core import SKNMSIMethodABC


@dataclass
class SSNIntegrator:
    """
    Integrator for the Stabilized Supralinear Network (SSN) model.

    Similar to Cuppini2017Integrator but with supralinear activation
    instead of sigmoid.

    - Función matemática pura que define las ecuaciones diferenciales
    - NO guarda estado, solo calcula derivadas
    - Fórmula que calcula cómo cambian las cosas por unidad de tiempo

    IMPORTANT: All time constants must be in SECONDS to match BrainPy's dt.
    The original Echeveste code uses: tau_e = 20.0e-3 s, tau_i = 10.0e-3 s
    """

    #: Time constants for excitatory and inhibitory neurons IN SECONDS
    #: tau_e: 0.02s (20ms), tau_i: 0.01s (10ms)
    #: From Echeveste et al. (2020) Supplementary Material Table S1
    tau_e: float
    tau_i: float

    #: Time constant for correlated noise η IN SECONDS
    #: tau_n: 0.02s (20ms) by Echeveste original implementation
    tau_n: float

    #: Supralinear exponent for Excitatory neurons, n=2.0
    # from Echeveste et al. (2020) Supplementary Material
    n: float

    #: Scaling factor for firing rates, k=0.3
    #: from Echeveste et al. (2020) Supplementary Material
    k: float

    #: Name of the integrator
    name: str = "SSNIntegrator"

    @property
    def __name__(self):
        """Return the name of the Integrator."""
        return self.name

    def supralinear_activation(self, u):
        """
        Supralinear activation: r = k * [u]_+^n where [x]_+ = max(0, x).

        Mathematical foundation:
        - Echeveste et al. (2020), Equations 8-9: The firing rate is
          r_α = k * [u_α]_+^n, where u_α is the membrane potential
        - Echeveste et al. (2020) Supplementary Material: The supralinear
          exponent
          n = 2.0 produces the required nonlinearity for sampling-based
          inference and k = 0.3 (scaling factor from SSN parameters)

        Note: Uses BrainPy's math backend which supports GPU acceleration.
        """
        return self.k * bp.math.power(bp.math.maximum(0, u), self.n)

    def __call__(self, u_e, u_i, t, W, h, eta):
        """
        Compute the SSN dynamics based on Echeveste et al. (2020).

        Mathematical foundation:
        - Main paper, Eq. 8: τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        - Main paper, Eq. 9: r_α = k * [u_α]_+^n (supralinear activation)
        - Echeveste et al. (2020) Supplementary Material, Table S1:
          τ_E = 20ms, τ_I = 10ms
        - Main paper, Section 2.2: E-I network structure with W_EE, W_EI,
            W_IE, W_II blocks
        - Main paper, Eq. 6-7: GSM generative model provides h (external input)
        - Echeveste et al. (2020) Supplementary Material: η represents
          inference noise

        INPUTS
        ------
        - u_e: Excitatory membrane potentials (shape: [N_E])
            variable u_α for α ∈ E
        - u_i: Inhibitory membrane potentials (shape: [N_I])
            variable u_α for α ∈ I
        - t: Current time (scalar)
        - W: Connectivity matrix (shape: [N, N]) - W_αβ from Eq. 8
        - h: External inputs (shape: [N]) - h_α from GSM model (Eq. 6-7)
        - eta: Noise inputs (shape: [N]) - η_α for sampling-based inference

        Return
        ------
        - (du_e_dt, du_i_dt): Temporal derivatives from Equation 8
        """
        # Calcula firing rates usando activación supralineal (Eq. 9)
        r_e = self.supralinear_activation(u_e)  # Tasas excitatorias
        r_i = self.supralinear_activation(u_i)  # Tasas inhibitorias

        # Número de neuronas excitatorias (Supp. Table S1: N_E=50)
        n_e = len(u_e)

        # IMPORTANTE: En entrenamiento los pesos W no son matrices almacenadas
        # sino que se calculan dinámicamente usando conectividad paramétrica
        # Ahora usamos W completa directamente como en código original

        # Implementa dinámicas SSN (Ecuación 8):
        # τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        # IMPORTANTE: W ya contiene los signos correctos (W_EI < 0, W_II < 0)

        # Concatenar firing rates para usar con W completa
        # Note: Uses BrainPy's math backend for GPU acceleration
        r = bp.math.concatenate([r_e, r_i])  # r_β del término Σ_β W_αβ r_β

        # Aplicar ecuación exacta como en código original: W @ r
        W_times_r = bp.math.matmul(W, r)  # Σ_β W_αβ r_β (signos correctos)

        # Para neuronas excitatorias (α ∈ E):
        du_e_dt = (
            -u_e  # Potencial de membrana
            + h[:n_e]  # Input externo del modelo GSM
            + W_times_r[:n_e]  # Σ_β W_αβ r_β (con signos de W)
            + eta[:n_e]  # Ruido de inferencia: η_α
        ) / self.tau_e  # Divide por constante de tiempo τ_E

        # Para neuronas inhibitorias (α ∈ I):
        du_i_dt = (
            -u_i  # Potencial de membrana
            + h[n_e:]  # Input externo del modelo GSM
            + W_times_r[n_e:]  # Σ_β W_αβ r_β (con signos de W)
            + eta[n_e:]  # Ruido de inferencia: η_α
        ) / self.tau_i

        return du_e_dt, du_i_dt  # Retorna derivadas temporales de Eq. 8


class Echeveste2020(SKNMSIMethodABC):
    """
    Cortical-like dynamics model from Echeveste et al. (2020).

    Implementation of "Cortical-like dynamics in recurrent
    circuits optimized for sampling-based probabilistic inference"
    - Echeveste et al., Nature Neuroscience 2020

    Mathematical foundation:
    - Main paper, Abstract: "optimized recurrent cortical circuits
    that implement sampling-based probabilistic inference"
    - Main paper, Eq. 1-7: Gaussian Scale Mixture (GSM) generative model
    - Main paper, Eq. 8-9: Stabilized Supralinear Network (SSN) dynamics
    - Main paper, Figure 1: Ring topology with circular symmetry for
        orientation selectivity
    - Echeveste et al. (2020) Supplementary Material, Table S1:
        Network parameters
        (N_E=50, N_I=50, etc.)

    Key features:
    - Ring topology with E-I populations (Main paper, Section 2.2)
    - Supralinear activation r = k[u]_+^n (Eq. 9, n=2.0)
    - Parametric connectivity:
        W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²] (Eq. 10)
    - Solo 8 parámetros optimizados:
        {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
    - GSM generative model for natural image patches (Eq. 1-7)
    - Sampling-based Bayesian inference (Main paper, Section 2.1)
    - Cortical-like gamma oscillations (Main paper, Figure 3)
    """

    # Implementación del modelo de dinámicas
    # corticales de Echeveste et al. (2020)

    _model_name = "Echeveste2020"
    _model_type = "Neural"
    _run_input = [
        {"target": "stimulus_contrast", "template": "stimulus_contrast"},
        {"target": "stimulus_orientation", "template": "stimulus_orientation"},
        {"target": "noise_level", "template": "noise_level"},
    ]
    _run_output = [
        {"target": "excitatory_firing_rate",
         "template": "excitatory_firing_rate"},
        {"target": "inhibitory_firing_rate",
         "template": "inhibitory_firing_rate"},
        {"target": "excitatory_potential", "template": "excitatory_potential"},
        {"target": "inhibitory_potential", "template": "inhibitory_potential"},
    ]
    _output_mode = "excitatory_firing_rate"  # Primary output mode

    # (Supp. Table S1)
    def __init__(
        self,
        *,
        N_E=50,  # Número de neuronas excitatorias
        N_I=50,  # Número de neuronas inhibitorias
        tau_e=20.0,  # ms - Excitatory time constant (→ s internally)
        tau_i=10.0,  # ms - Inhibitory time constant (→ s internally)
        tau_n=20.0,  # ms - Noise timescale η (→ s internally)
        n=2.0,  # Exponente supralineal de Eq. 9
        k=0.3,  # Factor de escala de Eq. 9
        seed=None,  # Semilla para generador aleatorio
        position_range=(
            0,
            180,
        ),  # Orientaciones ring topology (Main paper, Figure 1B)
        position_res=None,  # Calculated based on N_E for exact matching
        time_range=(0, 1000),  # Rango temporal de simulación (ms)
        time_res=0.2,  # Paso temporal dt
        device="auto",  # Device selection: 'cpu', 'gpu', or 'auto'
        **integrator_kws,  # Keywords adicionales para integrador
    ):
        """
        Initialize the SSN model with parameters from Echeveste et al. (2020).

        Mathematical foundation:
        - Echeveste et al. (2020) Supplementary Material, Table S1:
          Complete parameter specification
        - Main paper, Section 2.2: Network architecture with ring topology
        - Main paper, Eq. 8-9: SSN dynamics with specified time constants
        - Echeveste et al. (2020) Supplementary Material: Numerical
          integration details

        All default values match the optimized parameters from the paper's
        sampling-based inference optimization procedure.

        Parameters
        ----------
        device : {'cpu', 'gpu', 'auto'}, default='auto'
            Computational device to use:
            - 'cpu': Force CPU usage
            - 'gpu': Force GPU usage (requires JAX with CUDA support)
            - 'auto': Automatically select GPU if available, else CPU
        """
        # Configurar dispositivo computacional (GPU/CPU)
        from ._device_config import configure_device
        self._device_config = configure_device(device)

        # Estructura básica de la red
        self._N_E = N_E  # Número de neuronas excitatorias
        self._N_I = N_I  # Número de neuronas inhibitorias

        # Validación: Ring topology del paper asume N_E = N_I
        # (Ver Supp. Material: "ring of m sites, each with 1E + 1I neuron")
        # El input feedforward W_ff = [A A]^T requiere N_E = N_I para que
        # ambas poblaciones reciban la proyección completa de los filtros
        if self._N_E != self._N_I:
            raise ValueError(
                f"Echeveste2020 paper assumes N_E = N_I (ring topology). "
                f"Got N_E={N_E}, N_I={N_I}. "
                f"The feedforward input W_ff = [A A]^T / 15.0 requires "
                f"equal numbers of E and I neurons to properly duplicate "
                f"the GSM projection h = A.T @ x for both populations."
            )

        # Parámetros de conectividad parametrica - Eq. 10 del paper
        # Stage 1 optimization: optimize these 8 connectivity parameters
        self._a_EE = None  # Amplitud excitatorio-excitatorio
        self._a_EI = None  # Amplitud excitatorio-inhibitorio
        self._a_IE = None  # Amplitud inhibitorio-excitatorio
        self._a_II = None  # Amplitud inhibitorio-inhibitorio
        self._d_EE = None  # Ancho excitatorio-excitatorio
        self._d_EI = None  # Ancho excitatorio-inhibitorio
        self._d_IE = None  # Ancho inhibitorio-excitatorio
        self._d_II = None  # Ancho inhibitorio-inhibitorio

        # Storage for computed connectivity matrices
        self._W_EE = None  # Matriz de conectividad E-E
        self._W_EI = None  # Matriz de conectividad E-I
        self._W_IE = None  # Matriz de conectividad I-E
        self._W_II = None  # Matriz de conectividad I-I
        self._W_exact = None  # Exact original matrix from w_learn file

        # Storage for noise covariance matrix
        self._Sigma_eta = (
            None  # Matriz de covarianza del ruido η (optimizada en Stage 2)
        )

        # Input transformation parameters (Stage 2 optimization)
        self._input_scaling = None  # α_h
        self._input_baseline = None  # β_h
        self._input_nl_pow = None  # γ_h

        # Noise covariance parameters (Stage 2 optimization)
        self._noise_params = None  # Dict con var_e, var_i, var_width, rho

        # Training state tracking
        self._is_trained = False
        self._stage1_completed = False
        self._stage2_completed = False
        self._N = N_E + N_I  # Total de neuronas en ring topology

        # Parámetros espaciales y temporales
        self._position_range = position_range  # Rango orientaciones [0°, 180°]
        # Calculate position_res to match number of neurons for exact grid
        if position_res is None:
            # For ring topology: resolution = range / number_of_neurons
            # This ensures grid positions exactly match neuron count
            range_span = position_range[1] - position_range[0]
            self._position_res = (
                range_span / self._N_E if self._N_E > 0 else range_span
            )
        else:
            self._position_res = position_res  # Use provided value
        self._time_range = time_range  # Duración simulación (1000ms)
        self._time_res = time_res  # dt = 0.2ms

        # Crea integrador numérico para dinámicas SSN (Supp. Sec. 2.1)
        # Método Euler con dt = 0.2ms asegura estabilidad para dinámicas
        # inhibitorias rápidas
        integrator_kws.setdefault(
            "method", "euler"
        )  # Integración Euler forward
        integrator_kws.setdefault(
            "dt", time_res / 1000.0
        )  # Convierte ms a segundos para BrainPy

        # Remove seed from integrator_kws if present
        integrator_kws.pop("seed", None)
        integrator_kws.pop("random_seed", None)

        # Inicializa integrador SSN con parámetros optimizados (Supp. Table S1)
        # CRITICAL: Convert time constants from ms to seconds for consistency
        # BrainPy uses dt in seconds, so tau must also be in seconds
        # Original code: tau_e = 20.0e-3 s, tau_i = 10.0e-3 s, dt = 0.2e-3 s
        integrator_model = SSNIntegrator(
            tau_e=tau_e / 1000.0,  # Convert ms to seconds: 20ms → 0.02s
            tau_i=tau_i / 1000.0,  # Convert ms to seconds: 10ms → 0.01s
            tau_n=tau_n / 1000.0,  # Convert ms to seconds: 20ms → 0.02s
            n=n,  # Exponente supralineal n = 2.0 (Eq. 9)
            k=k,  # Factor de escala k = 0.3 (Eq. 9)
        )

        # Integrador ODE de BrainPy para dinámicas de Ecuación 8
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        self.set_random(
            np.random.default_rng(seed=seed)
        )  # RNG para ruido η (Supp. Sec. 2.3)

    def train(self, gsm_model, stage1_params=None, stage2_params=None):
        """
        Train the SSN model following the two-stage optimization procedure.

        Based on Echeveste et al. (2020) Methods section, page 18:

        **Stage 1: ADAM with stochastic sampling (N_trial = 50)**
        "During the first stage, we employed a stochastic gradient method
        using N_trial = 50 trials for each training stimulus to estimate
        the corresponding moments of network responses, and performed 250
        iterations of the ADAM optimizer. Both the network's initial
        conditions and the process noise were re-sampled for each trial
        and iteration. [...] T_min was systematically changed ('annealed')
        from T_min = 0 ms to T_max - 50 ms."

        **Stage 2: L-BFGS-B with deterministic ADF**
        "In the second stage, we continued optimization using the L-BFGS-B
        optimizer, now using the ADF method to (deterministically) compute
        the moments of the network's response distribution. We kept the
        cost-integration time window at its minimum (T_max - T_min = 50 ms)."

        The GSM (Gaussian Scale Mixture) generative model must be pre-trained.
        The SSN learns to perform sampling-based inference on this GSM.

        Parameters
        ----------
        gsm_model: object
            Pre-trained GSM generative model containing:
            - Gabor filters (receptive fields A)
            - Prior parameters for z and G
            - Trained parameters from natural image statistics
        stage1_params: dict, optional
            Parameters specific to Stage 1 (ADAM) optimization:
            - max_iter: number of ADAM iterations
                (default: 250, recommendation max_iter >= 50)
            - n_trials: number of samples per iteration (default: 50)
            - eta: ADAM learning rate (default: 0.002)
            - beta1, beta2: ADAM momentum parameters
            - t_min_initial, t_min_final: annealing range
            - lambda_mean: cost weight for mean matching
                (default: 4.0e-5, from Suppl. Table S1)
            - lambda_var: cost weight for variance matching
                (default: 8.0e-5, from Suppl. Table S1)
            - lambda_cov: cost weight for covariance matching
                (default: 8.0e-7, from Suppl. Table S1)
        stage2_params: dict, optional
            Parameters specific to Stage 2 (L-BFGS-B) optimization:
            - max_iter: number of L-BFGS-B iterations (default: 50)
            - min_time: fixed evaluation window start (default: T_max - 50ms)
            - lambda_mean: cost weight for mean matching
                (default: 4.0e-5, from Suppl. Table S1)
            - lambda_var: cost weight for variance matching
                (default: 8.0e-5, from Suppl. Table S1)
            - lambda_cov: cost weight for covariance matching
                (default: 8.0e-7, from Suppl. Table S1)
            - lambda_slow: slowness penalty weight
                (default: 4.0e-8, from Suppl. Table S1, ADF only)

        Returns
        -------
        dict
            Training results with optimized parameters and convergence metrics

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Methods page 18
        .. [2] ssn_inference_optimizer/train.ml
        """
        # Validate GSM model structure
        if not self._validate_gsm_model(gsm_model):
            raise ValueError("Invalid pre-trained GSM model")

        # Stage 1: Optimize connectivity parameters (Eq. 10)
        # Main paper, página 15: First optimize recurrent connectivity
        stage1_results = self._optimize_stage1(gsm_model, stage1_params)

        # Solo marcar completado si tuvo éxito
        if stage1_results.get('success', False):
            self._stage1_completed = True
            print("✓ Stage 1 completed successfully")
        else:
            msg = stage1_results.get('message', 'Unknown error')
            print(f"✗ Stage 1 failed: {msg}")
            # No continuar si Stage 1 falla
            return {
                "stage1": stage1_results,
                "stage2": None,
                "success": False,
                "error": "Stage 1 optimization failed",
                "stage1_completed": False,
                "stage2_completed": False,
                "is_trained": False,
            }

        # Stage 2: Optimize remaining parameters
        # Main paper, página 16: Then optimize stimulus and noise parameters
        stage2_results = self._optimize_stage2(gsm_model, stage2_params)

        # Solo marcar completado si tuvo éxito
        if stage2_results.get('success', False):
            self._stage2_completed = True
            print("✓ Stage 2 completed successfully")
        else:
            msg = stage2_results.get('message', 'Unknown error')
            print(f"✗ Stage 2 failed: {msg}")
            return {
                "stage1": stage1_results,
                "stage2": stage2_results,
                "success": False,
                "error": "Stage 2 optimization failed",
                "stage1_completed": True,
                "stage2_completed": False,
                "is_trained": False,
            }

        # Build final connectivity matrices from optimized parameters
        self._build_connectivity_matrices()

        # Marcar como entrenado solo si ambos stages fueron completados
        self._is_trained = self._stage1_completed and self._stage2_completed

        print("✓ Training completed successfully")

        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "connectivity_params": self._get_connectivity_parameters(),
            "convergence_info": self._get_convergence_info(),
            "success": True,
            "stage1_completed": self._stage1_completed,
            "stage2_completed": self._stage2_completed,
            "is_trained": self._is_trained,
        }

    def _validate_gsm_model(self, gsm_model):
        """
        Validate pre-trained GSM generative model structure.

        Based on Main paper, Eq. 1-7: GSM generative model must be
        pre-trained on natural images before SSN inference optimization.
        The GSM provides the generative model that the SSN learns to invert.
        """
        # Check for essential GSM attributes
        # A: Gabor filter matrix (from generative model)
        # C: Prior covariance matrix
        required_attributes = ["A", "C"]
        return all(hasattr(gsm_model, attr) for attr in required_attributes)

    def _optimize_stage1(self, gsm_model, params):
        """
        Stage 1: Optimize connectivity using ADAM and sample-based inference.

        Based on Echeveste et al. (2020) Methods section, page 18:
        "During the first stage, we employed a stochastic gradient method
        using N_trial = 50 trials for each training stimulus to estimate
        the corresponding moments of network responses, and performed 250
        iterations of the ADAM optimizer. Both the network's initial
        conditions and the process noise were re-sampled for each trial
        and iteration."

        This stage optimizes the 8 connectivity kernel parameters:
        {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
        which define the connectivity matrix W via Eq. 10:
            W_XY(θi, θj) = a_XY * exp((cos(2(θi - θj)) - 1) / d_XY²)

        Key features:
        1. ADAM optimizer with stochastic gradients
        2. Sample-based estimation (N_trial = 50 trials per iteration)
        3. Temporal annealing: T_min goes from 0ms → (T_max - 50ms)
        4. Re-sampling of initial conditions and noise each iteration

        Parameters
        ----------
        gsm_model: object
            Pre-trained GSM generative model defining target statistics
        params: dict
            Stage 1 specific parameters (optimizer settings, etc.)

        Returns
        -------
        dict
            Optimization results for Stage 1

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Methods page 18
        .. [2] ssn_inference_optimizer/train.ml lines 280-293
        """
        from ._echeveste_training import compute_costs_with_samples
        import jax
        import jax.numpy as jnp

        # Parámetros por defecto siguiendo el paper exactamente
        # Paper Methods page 18
        if params is None:
            params = {}

        # Parámetros de ADAM optimizer
        # train.ml lines 288-292
        max_iter = params.get('max_iter', 250)  # Paper: 250 iterations
        n_trials = params.get('n_trials', 50)   # Paper: N_trial = 50
        eta = params.get('eta', 0.002)          # Learning rate (default 0.002)
        beta1 = params.get('beta1', 0.9)        # ADAM β1
        beta2 = params.get('beta2', 0.999)      # ADAM β2
        epsilon_adam = params.get('epsilon', 1e-8)  # ADAM ε

        # Parámetros de simulación
        dt = params.get('dt', 0.2e-3)           # 0.2ms timestep
        t_max = params.get('t_max', 0.1)        # 100ms total simulation
        t_subsamp = params.get('t_subsamp', 10.0e-3)  # 10ms subsampling

        # Paper: "T_min was systematically changed ('annealed') from
        # T_min = 0 ms (stimulus onset) to T_max - 50 ms"
        t_min_initial = params.get('t_min_initial', 0.0)      # 0ms
        t_min_final = params.get('t_min_final', t_max - 0.05)  # T_max - 50ms

        # Parámetros de costo
        lambda_mean = params.get('lambda_mean', 1.0)
        lambda_var = params.get('lambda_var', 1.0)
        lambda_cov = params.get('lambda_cov', 1.0)

        # Inicialización de parámetros de conectividad
        # train.ml lines 92-100
        initial_params = {
            'a_EE': params.get('a_EE', 0.02),
            'a_EI': params.get('a_EI', 0.02),
            'a_IE': params.get('a_IE', 0.02),
            'a_II': params.get('a_II', 0.02),
            'd_EE': params.get('d_EE', 0.8),
            'd_EI': params.get('d_EI', 0.8),
            'd_IE': params.get('d_IE', 0.8),
            'd_II': params.get('d_II', 0.8),
        }

        # Pack parámetros iniciales
        x0 = self._pack_parameters(initial_params)

        # NOTE: Parameter bounds will be enforced manually in ADAM loop
        # objective.ml lines 235-238:
        # - Amplitudes (params 0-3): lower bound 0, no upper bound
        # - Widths (params 4-7): bounded by [0.01, sqrt(2)]
        #   Lower bound 0.01 prevents division by zero in d²

        # Construir inverse time constants
        inv_taus = jnp.concatenate([
            jnp.full(self._N_E, 1.0 / self._integrator.f.tau_e),
            jnp.full(self._N_I, 1.0 / self._integrator.f.tau_i)
        ])

        # Extraer h_vec y targets del GSM model
        # Generar targets reales desde el posterior del GSM
        print("Computing GSM posterior targets...")
        contrast = params.get('contrast', 0.5)
        n_samples_gsm = params.get('n_samples_gsm', 50)

        gsm_data = gsm_model.compute_posterior_for_ssn_training(
            contrast=contrast,
            n_samples=n_samples_gsm
        )

        # Usar el promedio de los h_inputs como estímulo base
        # Shape: (n_gsm_orientations,) - típicamente 50 del GSM pre-trained
        h_full = gsm_data['h_inputs'].mean(axis=0)

        # IMPORTANTE: Siguiendo el código original de Echeveste
        # (generalization.py:116), el input h se duplica para neuronas E e I:
        # h = np.concatenate((h,h)). Las neuronas inhibitorias reciben el
        # mismo input que las excitatorias (ref: generalization.py:115-116)
        h_e = h_full[:self._N_E]  # Primeras N_E para excitatorias
        h_i = h_e[:self._N_I]     # Duplicar para inhibitorias (mismo input)
        h_vec = jnp.concatenate([h_e, h_i])  # Shape: (N_E + N_I,)

        # Targets del posterior GSM (solo neuronas excitatorias)
        target_mu = jnp.array(gsm_data['target_mu'][:self._N_E])
        target_sigma = jnp.array(
            gsm_data['target_sigma'][:self._N_E, :self._N_E]
        )

        print(f"  Contrast: {contrast}")
        print(f"  GSM samples: {n_samples_gsm}")
        print(f"  h_vec shape: {h_vec.shape}")
        print(f"  Target mu shape: {target_mu.shape}")
        print(f"  Target sigma shape: {target_sigma.shape}")

        # Parámetros fijos de ruido para Stage 1
        noise_width = 0.8
        noise_std_e = 2.0
        noise_std_i = 2.0
        noise_rho = 0.8

        def build_w_from_params(param_values):
            """Construir matriz W desde vector de parámetros."""
            params_dict = self._unpack_parameters(param_values)

            # Orientaciones en el ring
            theta = jnp.linspace(0, jnp.pi, self._N_E, endpoint=False)

            # Función auxiliar para construir bloques
            # Usa _spatial_kernel_jax como single source of truth
            def connectivity_block(theta_pre, theta_post, a, d, sign):
                delta = theta_pre[:, None] - theta_post[None, :]
                kernel = Echeveste2020._spatial_kernel_jax(delta, d)
                return sign * a * kernel

            # Construir bloques
            W = jnp.zeros((self._N, self._N))
            W = W.at[:self._N_E, :self._N_E].set(
                connectivity_block(
                    theta, theta, params_dict['a_EE'],
                    params_dict['d_EE'], 1.0
                )
            )
            W = W.at[:self._N_E, self._N_E:].set(
                connectivity_block(
                    theta, theta, params_dict['a_EI'],
                    params_dict['d_EI'], -1.0
                )
            )
            W = W.at[self._N_E:, :self._N_E].set(
                connectivity_block(
                    theta, theta, params_dict['a_IE'],
                    params_dict['d_IE'], 1.0
                )
            )
            W = W.at[self._N_E:, self._N_E:].set(
                connectivity_block(
                    theta, theta, params_dict['a_II'],
                    params_dict['d_II'], -1.0
                )
            )

            return W

        def build_sigma_eta():
            """Construir matriz de covarianza de ruido."""
            var_e = noise_std_e ** 2
            var_i = noise_std_i ** 2

            theta = jnp.linspace(0, jnp.pi, self._N_E, endpoint=False)
            delta = theta[:, None] - theta[None, :]

            # Usa _spatial_kernel_jax como single source of truth
            spatial_kernel = Echeveste2020._spatial_kernel_jax(
                delta, noise_width
            )

            # Construir bloques
            Sigma_ee = var_e * spatial_kernel
            Sigma_ii = var_i * spatial_kernel
            Sigma_ei = (
                noise_rho * jnp.sqrt(var_e * var_i) * spatial_kernel
            )

            # Ensamblar matriz completa
            Sigma_eta = jnp.block([
                [Sigma_ee, Sigma_ei],
                [Sigma_ei.T, Sigma_ii]
            ])

            # Agregar término diagonal para estabilidad
            # objective.ml:305
            Sigma_eta = Sigma_eta + 0.01 * jnp.eye(self._N)

            return Sigma_eta

        sigma_eta = build_sigma_eta()

        # =====================================================================
        # ADAM Optimizer with Sample-Based Optimization
        # =====================================================================
        # Paper Methods page 18: "performed 250 iterations of the ADAM
        # optimizer. Both the network's initial conditions and the process
        # noise were re-sampled for each trial and iteration."
        # train.ml lines 280-293

        print("=" * 60)
        print("Stage 1: ADAM with sample-based optimization")
        print("=" * 60)
        print("Parameters:")
        print("  Optimizer: ADAM")
        print(f"  Max iterations: {max_iter}")
        print(f"  N_trials per iteration: {n_trials}")
        print(f"  Learning rate (η): {eta}")
        print(f"  Beta1: {beta1}, Beta2: {beta2}")
        print(f"  T_min annealing: {t_min_initial*1000:.0f}ms → "
              f"{t_min_final*1000:.0f}ms")
        print(f"Initial parameters: {initial_params}")
        print("=" * 60)

        # Inicializar temporal_weighting para annealing
        # Ref: train.ml line 281, objective.ml line 528
        n_time_bins = int(t_max / dt)
        min_time_bins = int(0.05 / dt)  # Fijo: últimos 50ms
        min_time_fixed = 0.05  # Fijo: 50ms
        temporal_weighting = np.ones(n_time_bins)

        # Función de costo con sampling estocástico
        def compute_cost_and_grad(x, iteration, key):
            """
            Compute cost and gradient using stochastic sampling.

            Paper: "using N_trial = 50 trials for each training stimulus
            to estimate the corresponding moments"

            TEMPORAL ANNEALING:
            -------------------
            En lugar de cambiar min_time, usamos temporal_weighting que
            se actualiza en cada iteración (ver código original OCaml).

            Ref: train.ml lines 240-243, 286
            - Line 240-243: update_temporal_weighting función
            - Line 286: update_temporal_weighting k (en cada iteración)

            Paper Methods página 18:
            "the beginning of the averaging time window, T_min in Eqs.  26-28,
            was systematically changed ('annealed') from T_min = 0 ms to
            T_max - 50 ms during the initial 200 iterations"
            """
            # Construir matriz W desde parámetros
            w = build_w_from_params(x)

            # Calcular costo con N_trial samples y temporal_weighting
            # Paper: "Both the network's initial conditions and the process
            # noise were re-sampled for each trial and iteration"
            # Ref: objective.ml lines 544-550 (usa wt en cada timestep)
            cost, _, _ = compute_costs_with_samples(
                w, h_vec, sigma_eta, inv_taus, dt,
                self._integrator.f.tau_n, t_max, t_subsamp,
                target_mu, target_sigma, self._integrator.f.k,
                lambda_mean, lambda_var, lambda_cov,
                min_time_fixed,  # ← FIJO, no cambia (50ms)
                n_trials, key,
                temporal_weighting=temporal_weighting  # ← Cambia cada iter
            )

            return cost

        # Función para calcular costo Y gradiente en una sola pasada
        # OPTIMIZACIÓN: value_and_grad es más eficiente que dos llamadas
        # separadas (cost + grad) porque hace forward+backward una sola vez
        # Ref: DIAGNOSTICO_OOM_STAGE1.md
        cost_and_grad_fn = jax.value_and_grad(
            compute_cost_and_grad, argnums=0
        )

        # Inicializar parámetros
        x = x0.copy()

        # =================================================================
        # Inicializar ADAM optimizer usando Optax
        # Ref: Código original usa biblioteca externa de OCaml
        # (train.ml:288-293), nosotros usamos Optax (estándar JAX)
        # =================================================================
        import optax

        # Crear optimizador ADAM con hiperparámetros del paper
        # train.ml lines 288-292:
        # - eta = 0.002 (learning rate)
        # - beta1 = 0.9, beta2 = 0.999 (ADAM betas)
        # - epsilon = 1e-8 (ADAM epsilon)
        optimizer = optax.adam(
            learning_rate=eta,
            b1=beta1,
            b2=beta2,
            eps=epsilon_adam
        )
        opt_state = optimizer.init(x)

        # TODELETE: Manual ADAM implementation (reemplazado por Optax)
        # # Inicializar momentos de ADAM
        # # ADAM mantiene promedios móviles exponenciales de gradientes
        # # (m) y gradientes al cuadrado (v)
        # m = jnp.zeros_like(x)  # First moment (mean of gradients)
        # v = jnp.zeros_like(x)  # Second moment (uncentered variance)

        # Gradient clipping
        # train.ml line 292: clip = sqrt(n_prms) * 5
        grad_clip = np.sqrt(len(x)) * 5.0

        # Tracking
        cost_history = []
        best_cost = float('inf')
        best_x = x.copy()

        # Random key para reproducibilidad
        key = jax.random.PRNGKey(params.get('seed', 42))

        # Inicializar temporal_weighting antes del loop
        # Ref: train.ml line 281
        from ._echeveste_training import update_temporal_weighting
        update_temporal_weighting(
            temporal_weighting, n_time_bins, min_time_bins,
            iteration=1, max_iterations=200
        )

        # ADAM optimization loop
        # Paper: "performed 250 iterations of the ADAM optimizer"
        for iteration in range(1, max_iter + 1):
            # Actualizar temporal_weighting (annealing progresivo)
            # Ref: train.ml line 286
            # "update_temporal_weighting k" en cada iteración
            update_temporal_weighting(
                temporal_weighting, n_time_bins, min_time_bins,
                iteration, max_iterations=200
            )

            # Generar nuevo key para este iteration
            # Paper: "re-sampled for each trial and iteration"
            key, subkey = jax.random.split(key)

            # Calcular costo Y gradiente en una sola pasada
            # OPTIMIZACIÓN: Evita doble evaluación forward+backward
            cost, grad = cost_and_grad_fn(x, iteration, subkey)

            # Convertir a numpy para obtener escalar
            cost_val = float(cost)

            # =============================================================
            # ADAM update usando Optax
            # Ref: train.ml:288-293 usa Adam.min (biblioteca externa OCaml)
            # Nosotros usamos optax.adam (biblioteca estándar JAX)
            # =============================================================

            # Gradient clipping para estabilidad
            # train.ml line 292: clip = sqrt(n_prms) * 5
            grad_norm = jnp.linalg.norm(grad)
            if grad_norm > grad_clip:
                grad = grad * (grad_clip / grad_norm)

            # ADAM update con Optax
            # Optax maneja internamente:
            # - First moment estimation (m)
            # - Second moment estimation (v)
            # - Bias correction
            # - Parameter update
            # Equivalente a Kingma & Ba (2015) Algorithm 1
            updates, opt_state = optimizer.update(grad, opt_state, x)
            x = optax.apply_updates(x, updates)

            # TODELETE: Manual ADAM update (reemplazado por Optax)
            # # Convertir a numpy para manipulación
            # # OPTIMIZACIÓN: Libera arrays JAX intermediate
            # grad = np.array(grad)
            #
            # # Gradient clipping para estabilidad
            # # train.ml line 261
            # grad_norm = np.linalg.norm(grad)
            # if grad_norm > grad_clip:
            #     grad = grad * (grad_clip / grad_norm)
            #
            # # ADAM update en numpy para evitar acumulación de JAX arrays
            # # OPTIMIZACIÓN: Trabajar en numpy reduce presión de memoria JAX
            # # Ref: DIAGNOSTICO_OOM_STAGE1.md
            # m_np = np.array(m)
            # v_np = np.array(v)
            # x_np = np.array(x)
            #
            # # Paper referencias: Kingma & Ba (2015)
            # m_np = beta1 * m_np + (1 - beta1) * grad
            # v_np = beta2 * v_np + (1 - beta2) * (grad ** 2)
            #
            # # Bias correction
            # m_hat = m_np / (1 - beta1 ** iteration)
            # v_hat = v_np / (1 - beta2 ** iteration)
            #
            # # Parameter update
            # x_np = x_np - eta * m_hat / (np.sqrt(v_hat) + epsilon_adam)
            #
            # # Convertir de vuelta a JAX (necesario para bounds)
            # x = jnp.array(x_np)
            # m = jnp.array(m_np)
            # v = jnp.array(v_np)

            # Apply bounds
            # train.ml lines 235-238: width parameters bounded by sqrt(2)
            for i in range(8):
                if i < 4:  # Amplitudes: lower bound 0
                    x = x.at[i].set(jnp.maximum(0, x[i]))
                else:  # Widths: [0.01, sqrt(2)]
                    x = x.at[i].set(jnp.clip(x[i], 0.01, np.sqrt(2.0)))

            # Track best
            cost_history.append(cost_val)
            if cost_val < best_cost:
                best_cost = cost_val
                best_x = x.copy()

            # Progress reporting
            # train.ml line 284: "iteration %5i | cost = %.5f"
            if iteration % 10 == 0 or iteration == 1:
                current_t_min = (
                    t_min_initial +
                    (iteration / max_iter) * (t_min_final - t_min_initial)
                )
                print(f"Iteration {iteration:5d} | "
                      f"Cost: {cost_val:.5f} | "
                      f"T_min: {current_t_min*1000:.1f}ms | "
                      f"||grad||: {float(grad_norm):.3f}")

            # Limpieza periódica de memoria para evitar OOM
            # OPTIMIZACIÓN: Cada 10 iteraciones limpiamos cachés de JAX
            # Aumentada frecuencia para 250 iter (antes era cada 20)
            # Ref: DIAGNOSTICO_OOM_STAGE1.md
            if iteration % 10 == 0:
                import gc
                jax.clear_caches()  # Limpiar cachés de compilación JAX
                gc.collect()        # Garbage collection de Python

        # Use best parameters found
        x_final = best_x
        optimized_params = self._unpack_parameters(x_final)

        # Actualizar parámetros internos
        self._a_EE = optimized_params['a_EE']
        self._a_EI = optimized_params['a_EI']
        self._a_IE = optimized_params['a_IE']
        self._a_II = optimized_params['a_II']
        self._d_EE = optimized_params['d_EE']
        self._d_EI = optimized_params['d_EI']
        self._d_IE = optimized_params['d_IE']
        self._d_II = optimized_params['d_II']

        print("=" * 60)
        print("Stage 1 completed:")
        print(f"  Final cost: {best_cost:.6f}")
        print(f"  Iterations: {max_iter}")
        print(f"  Optimized parameters: {optimized_params}")
        print("=" * 60)

        return {
            'optimized_params': optimized_params,
            'initial_params': initial_params,
            'final_cost': float(best_cost),
            'n_iterations': max_iter,
            'max_iterations': max_iter,
            'success': True,  # ADAM siempre completa las iteraciones
            'message': 'ADAM optimization completed',
            'cost_history': cost_history,
            'optimizer': 'ADAM',
            'n_trials': n_trials,
        }

    def _optimize_stage2(self, gsm_model, params):
        """
        Stage 2: Optimize ALL 15 parameters using L-BFGS-B with ADF method.

        IMPORTANTE: Esta implementación replica EXACTAMENTE el comportamiento
        de Echeveste et al. (2020), optimizando los 15 parámetros
        simultáneamente:
        - 3 input transformation: α_h, β_h, γ_h
        - 8 connectivity: a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II
        - 4 noise covariance: width, σ_E, σ_I, ρ

        Based on Echeveste et al. (2020) Methods section, page 18:
        "In the second stage, we continued optimization using the L-BFGS-B
        optimizer, now using the ADF method to (deterministically) compute
        the moments of the network's response distribution."

        Key characteristics:
        1. L-BFGS-B optimizer (quasi-Newton method)
        2. Deterministic ADF method (no sampling, N_trial = None)
        3. Fixed time window: T_min = T_max - 50ms
        4. Slowness penalty cost (lambda_slow)
        5. Optimizes ALL 15 parameters (unlike Stage 1)
        6. Reuses Stage 1 results as initialization for connectivity params

        Parameters
        ----------
        gsm_model: object
        Pre-trained GSM generative model
        params: dict
        Stage 2 specific parameters

        Returns
        -------
        dict
        Optimization results for Stage 2

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Methods page 18
        .. [2] ssn_inference_optimizer/train.ml lines 249-278
        .. [3] ssn_inference_optimizer/objective.ml lines 101-105 (n_prms=15)
        """
        from scipy.optimize import minimize
        from ._echeveste_training import create_objective_function
        import jax.numpy as jnp
        import numpy as np

        # Parámetros por defecto de Stage 2
        if params is None:
            params = {}

        print("="*70)
        print("STAGE 2: L-BFGS-B with ADF (15 parameters)")
        print("="*70)

        # Parámetros de optimización
        max_iter = params.get('max_iter', 250)
        dt = params.get('dt', 0.2e-3)
        t_max = params.get('t_max', 0.1)
        t_subsamp = params.get('t_subsamp', 10.0e-3)
        min_time = params.get('min_time', 0.05)

        # Parámetros de costo (multiplicadores pre-normalización)
        # Ref: train.ml lines 36-39, objective.ml:110-113
        lambda_mean = params.get('lambda_mean', 1.0)
        lambda_var = params.get('lambda_var', 2.0)
        lambda_cov = params.get('lambda_cov', 1.0)
        lambda_slow = params.get('lambda_slow', 0.01)

        print("\nOptimization parameters:")
        print(f"  max_iter: {max_iter}")
        print(f"  dt: {dt} s")
        print(f"  t_max: {t_max} s")
        print(f"  min_time: {min_time} s")
        print(f"  lambda weights: mean={lambda_mean}, var={lambda_var}, "
              f"cov={lambda_cov}, slow={lambda_slow}")

        # PASO 1: Generar targets del GSM con h_vec RAW
        # ================================================
        print("\nGenerating GSM targets (apply_transformation=False)...")
        contrast = params.get('contrast', 0.5)
        n_samples_gsm = params.get('n_samples_gsm', 50)

        gsm_data = gsm_model.compute_posterior_for_ssn_training(
            contrast=contrast,
            n_samples=n_samples_gsm,
            apply_transformation=False  # ← Clave: obtener h_vec raw
        )

        print(f"  Generated {n_samples_gsm} samples at contrast={contrast}")
        print(f"  h_inputs shape: {gsm_data['h_inputs'].shape}")

        # PASO 2: Inicializar los 15 parámetros
        # ========================================
        print("\nInitializing 15 parameters...")

        # 2a. Input transformation parameters (train.ml:93-95)
        input_baseline_lb = gsm_model.compute_input_baseline_lb(n_samples=100)
        print(f"  Computed input_baseline_lb: {input_baseline_lb:.6f}")

        initial_params = {}

        # Valores iniciales de transformación
        initial_params['input_baseline'] = input_baseline_lb + 1.0  # ~1.6
        initial_params['input_scaling'] = 1.0
        initial_params['input_nl_pow'] = 1.0

        # 2b. Connectivity parameters
        # Si Stage 1 completado: usar resultados
        # Si no: usar defaults (objective.ml:92-94)
        if self._stage1_completed and hasattr(self, '_connectivity_params'):
            # Reusar parámetros de Stage 1
            conn_params = self._connectivity_params
            print("  Reusing connectivity from Stage 1:")
        else:
            # Usar defaults
            conn_params = {
                'a_EE': 0.02, 'a_EI': 0.02, 'a_IE': 0.02, 'a_II': 0.02,
                'd_EE': 0.8, 'd_EI': 0.8, 'd_IE': 0.8, 'd_II': 0.8,
            }
            print("  Using default connectivity (Stage 1 not completed):")

        initial_params.update(conn_params)

        # 2c. Noise parameters (objective.ml:174-179)
        initial_params['sigma_eta_width'] = params.get('noise_width', 0.8)
        initial_params['sigma_eta_std_e'] = params.get('noise_std_e', 2.0)
        initial_params['sigma_eta_std_i'] = params.get('noise_std_i', 2.0)
        initial_params['sigma_eta_rho'] = params.get('noise_rho', 0.8)

        # Imprimir valores iniciales
        print("\n  Input transformation:")
        print(f"    α_h = {initial_params['input_scaling']:.4f}")
        print(f"    β_h = {initial_params['input_baseline']:.4f}")
        print(f"    γ_h = {initial_params['input_nl_pow']:.4f}")
        print("  Connectivity (sample):")
        print(f"    a_EE = {initial_params['a_EE']:.4f}")
        print(f"    d_EE = {initial_params['d_EE']:.4f}")
        print("  Noise:")
        print(f"    width = {initial_params['sigma_eta_width']:.4f}")
        print(f"    σ_E = {initial_params['sigma_eta_std_e']:.4f}")
        print(f"    σ_I = {initial_params['sigma_eta_std_i']:.4f}")
        print(f"    ρ = {initial_params['sigma_eta_rho']:.4f}")

        # PASO 3: Crear función objetivo con 15 parámetros
        # ==================================================
        print("\nCreating objective function (15 parameters)...")

        objective_fn, gradient_fn, pack_fn, unpack_fn = \
            create_objective_function(
                gsm_model,
                self._N_E,
                self._N_I,
                self._integrator.f.tau_e,
                self._integrator.f.tau_i,
                self._integrator.f.tau_n,
                self._integrator.f.k,
                dt,
                t_max,
                t_subsamp,
                min_time,
                lambda_mean,
                lambda_var,
                lambda_cov
            )

        print("  ✓ Objective function created")
        print("  ✓ Gradient function (JAX autodiff)")
        print("  ✓ Pack/unpack functions for 15 parameters")

        # PASO 4: Pack parámetros iniciales
        # ===================================
        x0 = pack_fn(initial_params)
        print(f"\n  Packed parameters: x0.shape = {x0.shape}")

        # PASO 5: Configurar bounds (objective.ml:216-237)
        # =================================================
        print("\nConfiguring bounds for 15 parameters...")

        # Bounds específicos según objective.ml
        sqrt2 = np.sqrt(2.0)
        bounds = [
            # Input transformation (0-2)
            (0.0, None),              # input_baseline - lb (≥ 0)
            (0.0, None),              # input_scaling (≥ 0)
            (0.0, None),              # input_nl_pow (≥ 0)
            # Weight heights (3-6)
            (0.0, None),              # a_EE - 0.01 (≥ 0)
            (0.0, None),              # a_EI - 0.01
            (0.0, None),              # a_IE - 0.01
            (0.0, None),              # a_II - 0.01
            # Weight widths (7-10)
            (0.01, sqrt2),            # d_EE ∈ (0, √2)
            (0.01, sqrt2),            # d_EI
            (0.01, sqrt2),            # d_IE
            (0.01, sqrt2),            # d_II
            # Noise parameters (11-14)
            (0.01, sqrt2),            # width ∈ (0, √2)
            (0.1, 4.0),               # std_e ∈ (0.1, 4)
            (0.1, 4.0),               # std_i ∈ (0.1, 4)
            (-5.0, 5.0),              # rho_transformed ∈ (-∞, ∞)
        ]

        print(f"  Configured {len(bounds)} bounds")

        # PASO 6: Wrappers para scipy
        # =============================
        def objective_scipy(x):
            """Wrapper que retorna float para scipy."""
            return float(objective_fn(jnp.array(x)))

        def gradient_scipy(x):
            """Wrapper que retorna np.array para scipy."""
            return np.array(gradient_fn(jnp.array(x)))

        # PASO 7: Optimización L-BFGS-B
        # ==============================
        print("\nStarting L-BFGS-B optimization...")
        print(f"  Initial cost: {objective_scipy(x0):.6f}")

        lbfgs_options = {
            'maxiter': max_iter,
            'disp': True,
            'maxcor': 20,        # Hessian history size
            'ftol': 1e-10,       # Function tolerance
            'gtol': 1e-8,        # Gradient tolerance
        }

        result = minimize(
            objective_scipy,
            x0,
            method='L-BFGS-B',
            jac=gradient_scipy,
            bounds=bounds,
            options=lbfgs_options
        )

        # PASO 8: Procesar resultados
        # =============================
        print("\nOptimization complete:")
        print(f"  Success: {result.success}")
        print(f"  Message: {result.message}")
        print(f"  Final cost: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}/{max_iter}")
        print(f"  Function evaluations: {result.nfev}")

        # Unpack parámetros optimizados
        optimized_params = unpack_fn(result.x)

        # Comparar inicial vs final
        print("\nParameter changes:")
        print("  Input transformation:")
        print(f"    α_h: {initial_params['input_scaling']:.4f} → "
              f"{optimized_params['input_scaling']:.4f}")
        print(f"    β_h: {initial_params['input_baseline']:.4f} → "
              f"{optimized_params['input_baseline']:.4f}")
        print(f"    γ_h: {initial_params['input_nl_pow']:.4f} → "
              f"{optimized_params['input_nl_pow']:.4f}")

        print("  Connectivity (sample):")
        print(f"    a_EE: {initial_params['a_EE']:.4f} → "
              f"{optimized_params['a_EE']:.4f}")
        print(f"    d_EE: {initial_params['d_EE']:.4f} → "
              f"{optimized_params['d_EE']:.4f}")

        print("  Noise:")
        print(f"    width: {initial_params['sigma_eta_width']:.4f} → "
              f"{optimized_params['sigma_eta_width']:.4f}")
        print(f"    σ_E: {initial_params['sigma_eta_std_e']:.4f} → "
              f"{optimized_params['sigma_eta_std_e']:.4f}")
        print(f"    σ_I: {initial_params['sigma_eta_std_i']:.4f} → "
              f"{optimized_params['sigma_eta_std_i']:.4f}")
        print(f"    ρ: {initial_params['sigma_eta_rho']:.4f} → "
              f"{optimized_params['sigma_eta_rho']:.4f}")

        # PASO 9: Actualizar modelo si exitoso
        # ======================================
        if result.success:
            # Actualizar conectividad
            self._connectivity_params = {
                'a_EE': optimized_params['a_EE'],
                'a_EI': optimized_params['a_EI'],
                'a_IE': optimized_params['a_IE'],
                'a_II': optimized_params['a_II'],
                'd_EE': optimized_params['d_EE'],
                'd_EI': optimized_params['d_EI'],
                'd_IE': optimized_params['d_IE'],
                'd_II': optimized_params['d_II'],
            }
            self._W = self.build_connectivity_matrix(self._connectivity_params)

            # Actualizar noise covariance
            self._Sigma_eta = self._build_noise_covariance(
                optimized_params['sigma_eta_width'],
                optimized_params['sigma_eta_std_e'],
                optimized_params['sigma_eta_std_i'],
                optimized_params['sigma_eta_rho']
            )

            # Actualizar parámetros de transformación de input
            self._input_scaling = optimized_params['input_scaling']
            self._input_baseline = optimized_params['input_baseline']
            self._input_nl_pow = optimized_params['input_nl_pow']

            # Guardar parámetros de ruido (para reconstrucción)
            self._noise_params = {
                'var_e': optimized_params['sigma_eta_std_e']**2,
                'var_i': optimized_params['sigma_eta_std_i']**2,
                'var_width': optimized_params['sigma_eta_width'],
                'rho': optimized_params['sigma_eta_rho'],
            }

            print("\n✓ Model parameters updated successfully")
        else:
            print("\n✗ Optimization failed, parameters NOT updated")

            # PASO 10: Retornar resultados
            # =============================
            return {
                'optimized_params': optimized_params,
                'initial_params': initial_params,
                'final_cost': float(result.fun),
                'initial_cost': float(objective_scipy(x0)),
                'n_iterations': int(result.nit),
                'max_iterations': max_iter,
                'success': bool(result.success),
                'message': str(result.message),
                'nfev': int(result.nfev),
                'optimizer': 'L-BFGS-B',
                'n_parameters': 15,
                'lambda_slow': lambda_slow,
            }

    def _build_noise_covariance(self, width, std_e, std_i, rho):
        """
        Build noise covariance matrix.

        Now uses vectorized implementation for efficiency.

        Parameters
        ----------
        width : float
            Width parameter for spatial correlation (radians)
        std_e : float
            Standard deviation for excitatory noise
        std_i : float
            Standard deviation for inhibitory noise
        rho : float
            Cross-correlation coefficient between E and I populations

        Returns
        -------
        np.ndarray
            Noise covariance matrix of shape (2*N_E, 2*N_E)

        Notes
        -----
        Delegates to _build_noise_covariance_vectorized for efficient
        vectorized computation.

        References
        ----------
        - objective.ml lines 290-305 (sigma_eta function)
        """
        return self._build_noise_covariance_vectorized(
            width, std_e, std_i, rho
        )

    def _build_connectivity_matrices(self):
        """
        Build full connectivity matrices from optimized parameters.

        Uses the optimized 8 parameters to compute full W matrices via Eq. 10.
        This follows the hybrid approach: store both parameters and matrices.
        """
        if not self._stage1_completed:
            raise ValueError(
                "Stage 1 must be completed before building matrices"
            )

        # Generate orientation vectors for ring topology
        orientations = np.linspace(0, np.pi, self._N, endpoint=False)
        # Alternativa de cálculo:
        # orientations = np.pi * np.arange(self._N) / self._N

        # Build matrices using parametric connectivity
        self._W_EE = self._build_parametric_matrix(
            orientations[:self._N_E],
            orientations[:self._N_E],
            self._a_EE,
            self._d_EE,
        )
        self._W_EI = -self._build_parametric_matrix(
            orientations[:self._N_E],
            orientations[self._N_E:],
            self._a_EI,
            self._d_EI,
        )
        self._W_IE = self._build_parametric_matrix(
            orientations[self._N_E:],
            orientations[:self._N_E],
            self._a_IE,
            self._d_IE,
        )
        self._W_II = -self._build_parametric_matrix(
            orientations[self._N_E:],
            orientations[self._N_E:],
            self._a_II,
            self._d_II,
        )

    def _build_parametric_matrix(self, theta_pre, theta_post, a_xy, d_xy):
        """
        Build connectivity matrix using parametric formula from Eq. 10.

        Now uses vectorized implementation for efficiency.

        Parameters
        ----------
        theta_pre: np.ndarray
            Orientations of pre-synaptic neurons (radians)
        theta_post: np.ndarray
            Orientations of post-synaptic neurons (radians)
        a_xy: float
            Amplitude parameter
        d_xy: float
            Width parameter (radians)

        Returns
        -------
        np.ndarray
            Connectivity matrix of shape (len(theta_post), len(theta_pre))

        Notes
        -----
        Delegates to _build_parametric_matrix_vectorized for efficient
        vectorized computation.
        """
        return self._build_parametric_matrix_vectorized(
            theta_pre, theta_post, a_xy, d_xy, sign=1.0
        )

    def _get_connectivity_parameters(self):
        """Return dictionary of 8 connectivity parameters."""
        return {
            "a_EE": self._a_EE,
            "a_EI": self._a_EI,
            "a_IE": self._a_IE,
            "a_II": self._a_II,
            "d_EE": self._d_EE,
            "d_EI": self._d_EI,
            "d_IE": self._d_IE,
            "d_II": self._d_II,
        }

    def _get_convergence_info(self):
        """Return convergence information from training stages."""
        return {
            "stage1_completed": self._stage1_completed,
            "stage2_completed": self._stage2_completed,
            "is_trained": self._is_trained,
        }

    def _compute_nonlinear_moments(self, mu, sigma):
        """
        Compute nonlinear moments for the supralinear activation.

        Based on objective.ml lines 254-268.
        Implements the calculation of nonlinear moments required
        for the evolution of the network under supralinear activation.
        Used for the ADF training method

        Parameters
        ----------
        mu : np.ndarray
            Mean vector of neuron inputs
        sigma : np.ndarray
            Covariance matrix of neuron inputs

        Returns
        -------
        tuple
            (nu, gamma) - nonlinear moment functions
        """
        sigma2 = np.diag(sigma)  # Variances
        sigma_std = np.sqrt(sigma2)  # Standard deviations

        # Ratio mu/sigma
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = mu / sigma_std
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

        # Normal PDF and CDF
        phi = stats.norm.pdf(ratio)  # φ(mu/σ)
        psi = stats.norm.cdf(ratio)  # Ψ(mu/σ)

        # Acceder a k desde el integrador
        k = self._integrator.f.k

        # Basado en objective.ml 261-267 y elementwise_harcoded
        # methods.py:87
        # La fórmula para n=2 es:
        # nu = k * ((μ² + σ²)·Ψ + μ·σ·φ)
        #    = k * (μ²·Ψ + μ·σ·φ + σ²·Ψ)
        #
        # Factorizando como en objective.ml:
        # nu1 = μ·Ψ + σ·φ  (sin factor k)
        # nu = k * (μ·nu1 + σ²·Ψ)

        nu1 = mu * psi + sigma_std * phi  # Intermedio SIN factor k
        nu = k * (mu * nu1 + sigma2 * psi)  # Aplicar k solo UNA vez

        # From objective.ml line 268: gamma_fun
        # gamma = 2*k * nu1  (gamma necesita el factor 2*k)
        gamma = 2 * k * nu1

        return nu, gamma

    def _pack_parameters(self, params_dict):
        """
        Pack parameters into optimization vector.

        Based on objective.ml lines 185-214 (pack function).
        Convierte parámetros físicos en vector de optimización
        aplicando transformaciones para garantizar positividad y bounds.

        Parameters
        ----------
        params_dict : dict
            Dictionary with keys: a_EE, a_EI, a_IE, a_II,
            d_EE, d_EI, d_IE, d_II, plus optional noise params

        Returns
        -------
        np.ndarray
            Packed parameter vector for optimization
        """
        packed = []

        # Height parameters (amplitudes) a_XY
        # Transformación: a_XY = 0.01 + x²  => x = sqrt(a_XY - 0.01)
        # Basado en objective.ml líneas 192
        for key in ['a_EE', 'a_EI', 'a_IE', 'a_II']:
            a = params_dict[key]
            if a <= 0.011:
                raise ValueError(
                    f"Initial {key}={a} too small, "
                    f"must be > 0.011"
                )
            packed.append(np.sqrt(a - 0.01))

        # Width parameters d_XY (sin transformación adicional)
        for key in ['d_EE', 'd_EI', 'd_IE', 'd_II']:
            packed.append(params_dict[key])

        return np.array(packed)

    def _unpack_parameters(self, x):
        """
        Unpack optimization vector into physical parameters.

        Based on objective.ml lines 153-180 (unpack function).
        Recupera parámetros físicos desde vector de optimización.

        Parameters
        ----------
        x : np.ndarray
            Optimization vector

        Returns
        -------
        dict
            Dictionary with unpacked parameters
        """
        params = {}

        # Amplitudes: a_XY = 0.01 + x²
        params['a_EE'] = 0.01 + x[0]**2
        params['a_EI'] = 0.01 + x[1]**2
        params['a_IE'] = 0.01 + x[2]**2
        params['a_II'] = 0.01 + x[3]**2

        # Widths: d_XY (directo)
        params['d_EE'] = x[4]
        params['d_EI'] = x[5]
        params['d_IE'] = x[6]
        params['d_II'] = x[7]

        return params

    def is_trained(self):
        """Check if model has been trained with both stages completed."""
        return self._stage1_completed and self._stage2_completed

    # PROPERTY ============================================================

    @property
    def N_E(self):
        """Number of excitatory neurons."""
        return self._N_E

    @property
    def N_I(self):
        """Number of inhibitory neurons."""
        return self._N_I

    @property
    def position_range(self):
        """Range of orientations."""
        return self._position_range

    @property
    def position_res(self):
        """Angular resolution."""
        return self._position_res

    @property
    def time_range(self):
        """Time range for simulation."""
        return self._time_range

    @property
    def time_res(self):
        """Time resolution."""
        return self._time_res

    # Ejecución del modelo
    def set_random(self, rng):
        """Set random number generator."""
        self._random = rng

    def burn_in(
        self,
        W,
        h,
        u_init,
        Sigma_eta,
        burn_in_time=10000.0,
        eta_init=None,
    ):
        """
        Evolve network to equilibrium WITHOUT recording samples.

        This implements the burn-in procedure from Echeveste et al. (2020)
        original code (methods.py:383-404, network_evolution function).
        The burn-in eliminates transient effects from arbitrary initial
        conditions, ensuring the network starts at steady state.

        Mathematical foundation:
        - Echeveste et al. (2020), Main paper Eq. 8:
          τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        - Main paper Eq. 9: r_α = k * [u_α]_+^n (supralinear activation)
        - Main paper Eq. 11: ⟨η(t)η(t+s)ᵀ⟩ = Σᶯexp(-s/τᶯ)
        - Original code reference: transients.py line 128 uses 100,000 steps
          (20,000ms), activity_example.py line 99 uses 50,000 steps (10,000ms)

        Parameters
        ----------
        W : array (N, N)
            Connectivity matrix
        h : array (N,)
            Input to network (typically h_baseline for spontaneous activity)
        u_init : array (N,)
            Initial membrane potentials
        Sigma_eta : array (N, N)
            Noise covariance matrix
        burn_in_time : float, default=10000.0
            Time in milliseconds for burn-in evolution
            Default 10,000ms matches activity_example.py (50,000 steps * 0.2ms)
            For transients experiments, use 20,000ms (transients.py uses
            100,000 steps)
        eta_init : array (N,) or None, default=None
            Initial noise state. If None, samples from N(0, Sigma_eta)

        Returns
        -------
        u_equilibrium : array (N,)
            Equilibrated membrane potential after burn-in
        eta_equilibrium : array (N,)
            Equilibrated noise state after burn-in

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience
        .. [2] ssn_inference_numerical_experiments/SSN/methods.py:383-404
        .. [3] ssn_inference_numerical_experiments/SSN/transients.py:128
        .. [4] ssn_inference_numerical_experiments/SSN/activity_example.py:99
        """
        # Convertir tiempo a pasos
        dt = self._time_res / 1000.0  # ms a segundos
        steps_max = int(burn_in_time / self._time_res)

        # Inicializar estados
        u_old = np.copy(u_init)

        # Inicializar ruido
        if eta_init is None:
            eta_old = self._random.multivariate_normal(
                np.zeros(self._N), Sigma_eta
            )
        else:
            eta_old = np.copy(eta_init)

        # Descomposición Cholesky para ruido correlacionado
        # Original code: methods.py:388
        L = np.linalg.cholesky(Sigma_eta)

        # Coeficientes para actualización temporal del ruido
        # Original code: methods.py:390-391
        tau_n_inv = 1.0 / self._integrator.f.tau_n
        eps_1 = 1.0 - dt * tau_n_inv
        eps_2 = np.sqrt(2.0 * dt * tau_n_inv)

        # Loop de evolución (NO guarda muestras)
        # Original code: methods.py:393-400
        steps = 0
        while steps < steps_max:
            steps += 1

            # Calcular actividad (Eq. 9)
            r_e = self._integrator.f.supralinear_activation(
                u_old[:self._N_E]
            )
            r_i = self._integrator.f.supralinear_activation(
                u_old[self._N_E:]
            )
            _ = bp.math.concatenate([r_e, r_i])  # noqa: F841

            # Generar ruido (methods.py:396)
            temp = bp.math.matmul(
                L, self._random.normal(loc=0.0, scale=1.0, size=self._N)
            )

            # Actualizar ruido correlacionado (methods.py:397)
            eta_new = eps_1 * eta_old + eps_2 * temp

            # Actualizar potenciales (Eq. 8, methods.py:398)
            u_e_old, u_i_old = u_old[:self._N_E], u_old[self._N_E:]
            u_e_new, u_i_new = self._integrator(
                u_e_old, u_i_old, steps * dt, W, h, eta_old
            )
            u_new = bp.math.concatenate([u_e_new, u_i_new])

            # Actualizar estados
            u_old = np.array(u_new)
            eta_old = np.array(eta_new)

            # Verificar estabilidad (methods.py:401-402)
            if np.linalg.norm(u_new) > 1000:
                print(f"Warning: Activity exploding during burn-in "
                      f"at step {steps}")
                break

        print(f"Burn-in completed: {steps} steps "
              f"({steps * self._time_res:.1f}ms)")

        return u_old, eta_old

    def run(
        self,
        *,
        stimulus_contrast=0.019,  # Contraste del estímulo z en GSM
        stimulus_orientation=0.0,  # Orientación del estímulo G en GSM
        noise_level=0.1,  # Nivel de ruido η para inferencia
        simulation_time=1000.0,  # Tiempo de simulación en ms
        # Parámetros de burn-in
        burn_in_time=10000.0,  # ms: Tiempo de burn-in (default 10s)
        # Parámetros para estructura de tres fases (transientes)
        use_three_phases=False,  # Si True, usa transientes como código orig
        t_init=225.0,  # ms: Pre-stimulus spontaneous activity
        t_stimulus=100.0,  # ms: Stimulus presentation duration
        t_final=125.0,  # ms: Post-stimulus spontaneous activity
        use_delays=False,  # Si True, usa delays por neurona en transiciones
        mean_delay=45.0,  # ms: Delay promedio para transiciones
        delay_sd=5.0,  # ms: Desviación estándar de delays
        h_input=None,  # Custom h_input vector (shape: N_E or N_E+N_I)
        **kwargs,  # Parámetros adicionales
    ):
        """
        Run the SSN simulation.

        Following Echeveste et al. (2020), this method supports two modes:

        1. **Single-phase mode** (use_three_phases=False, default):
           - Stimulus present throughout entire simulation
           - Used for steady-state analysis and training validation
           - Consistent with training procedure (objective.ml)

        2. **Three-phase mode** (use_three_phases=True):
           - Phase 1: Spontaneous activity (no stimulus, duration t_init)
           - Phase 2: Stimulus presentation (duration t_stimulus)
           - Phase 3: Post-stimulus spontaneous (no stimulus, duration t_final)
           - Used for transient dynamics and temporal response analysis
           - Consistent with simulation code (transients.py)
           - Optionally supports per-neuron delays in stimulus transitions

        Parameters
        ----------
        stimulus_contrast : float, default=0.019
            Contrast z of the GSM stimulus (Main paper Eq. 1-7)
        stimulus_orientation : float, default=0.0
            Orientation G of the GSM stimulus (radians)
        noise_level : float, default=0.1
            Noise level η for inference.
            IMPORTANT: When trained parameters are loaded (i.e.,
            self._Sigma_eta is not None), this parameter is IGNORED for
            the noise covariance matrix, as Σᶯ is already optimized and
            stored with the correct scale. The parameter only affects the
            fallback case when no trained Σᶯ is available (uses σ²I)
        simulation_time : float, default=1000.0
            Total simulation time in ms (for single-phase mode)
            Ignored in three-phase mode (uses t_init+t_stimulus+t_final)
        burn_in_time : float, default=10000.0
            Time in milliseconds for burn-in evolution
            Default 10,000ms matches activity_example.py
            For transients experiments (use_three_phases=True), consider
            using 20,000ms to match transients.py
            Reference: methods.py:383 (network_evolution)
        use_three_phases : bool, default=False
            If True, uses three-phase temporal structure (transients)
            If False, uses single-phase with constant stimulus (training)
        t_init : float, default=225.0
            Duration of pre-stimulus spontaneous activity in ms
            (Only used if use_three_phases=True)
            Reference: transients.py line 41
        t_stimulus : float, default=100.0
            Duration of stimulus presentation in ms
            (Only used if use_three_phases=True)
            Reference: transients.py line 42
        t_final : float, default=125.0
            Duration of post-stimulus spontaneous activity in ms
            (Only used if use_three_phases=True)
            Reference: transients.py line 43
        use_delays : bool, default=False
            If True, applies per-neuron delays in stimulus transitions
            (Only used if use_three_phases=True)
            Reference: transients.py lines 50-60, methods.py:746-776
        mean_delay : float, default=45.0
            Mean delay time for stimulus arrival at neurons in ms
            Reference: transients.py line 51
        delay_sd : float, default=5.0
            Standard deviation of delays in ms
            Reference: transients.py line 52
        h_input : array_like or None, default=None
            Custom stimulus input vector. If provided, overrides the
            automatic generation via _generate_gsm_stimulus().
            Shape can be either:
            - (N_E,): will be duplicated for inhibitory neurons [h, h]
            - (N_E + N_I,): used directly as full stimulus vector
            When using h_input, stimulus_contrast and stimulus_orientation
            are ignored for the stimulus generation (but still recorded
            in the output metadata).
            Useful for testing custom stimuli like varying width_factor
            in generate_bump_stimulus().

        Returns
        -------
        dict
            Simulation results with firing rates and membrane potentials

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience
        .. [2] ssn_inference_numerical_experiments/SSN/transients.py
        .. [3] ssn_inference_optimizer/objective.ml
        """
        # Verificar que el modelo esté entrenado antes de la simulación
        if not self.is_trained():
            raise ValueError(
                """Model must be trained before simulation.
                Use train() or load_parameters() method first."""
            )

        # =====================================================================
        # PREPARACIÓN DE ESTÍMULOS Y CONECTIVIDAD
        # =====================================================================
        # Generate stimuli following Echeveste et al. (2020)
        # Reference: Main paper Eq. 1-7, transients.py lines 88-102

        # Siempre generar baseline (contraste 0, actividad espontánea)
        h_baseline = self._generate_gsm_stimulus(0.0, 0.0)

        # Determinar h_stimulus: usar h_input personalizado o generar desde GSM
        if h_input is not None:
            # Usar h_input personalizado
            h_input = np.asarray(h_input)

            # Validar y ajustar dimensiones
            if len(h_input) == self._N_E:
                # Solo neuronas E: duplicar para I (como en código original)
                # Ref: generalization.py:115-116 h = np.concatenate((h,h))
                h_stimulus = np.concatenate([h_input, h_input])
                print(f"Using custom h_input (shape {self._N_E}), "
                      f"duplicated for I neurons")
            elif len(h_input) == self._N:
                # Vector completo E+I: usar directamente
                h_stimulus = h_input
                print(f"Using custom h_input (shape {self._N})")
            else:
                raise ValueError(
                    f"h_input must have shape ({self._N_E},) or ({self._N},), "
                    f"got ({len(h_input)},)"
                )
        else:
            # Generar h_stimulus desde GSM (comportamiento original)
            h_stimulus = self._generate_gsm_stimulus(
                stimulus_contrast, stimulus_orientation
            )

        # Validar dimensiones de estímulos
        self._validate_stimulus_dimensions(h_stimulus, "Stimulus")
        self._validate_stimulus_dimensions(h_baseline, "Baseline")

        # Obtener matriz de conectividad
        # Reference: objective.ml, usa w_learn pre-computada
        W = (self._W_exact if hasattr(self, "_W_exact")
             and self._W_exact is not None
             else self.build_connectivity_matrix())

        # Validar dimensiones de conectividad
        expected_W_shape = (self._N, self._N)
        if W.shape != expected_W_shape:
            raise ValueError(
                f"Connectivity matrix dimension mismatch. Expected "
                f"{expected_W_shape}, got {W.shape}."
            )

        # =====================================================================
        # PREPARACIÓN DE MATRICES DE RUIDO
        # =====================================================================
        # Reference: Main paper Eq. 11, Table S1 (τᶯ = 20ms)
        # methods.py:388 usa Sigma_eta directamente sin escalado
        Sigma_eta = (self._Sigma_eta if self._Sigma_eta is not None
                     else noise_level * np.eye(self._N))

        dt = self._time_res / 1000.0  # Convierte ms a segundos

        # =====================================================================
        # BURN-IN: Evolucionar red a equilibrio antes de registrar
        # =====================================================================
        # Reference: transients.py:128, activity_example.py:99
        # Condiciones iniciales: u(0) ~ N(0, 4I)
        u_init = np.random.multivariate_normal(
            np.zeros(self._N), 4.0 * np.eye(self._N)
        )

        print(f"Starting burn-in: {burn_in_time:.1f}ms...")
        u_0, eta_0 = self.burn_in(
            W=W, h=h_baseline, u_init=u_init,
            Sigma_eta=Sigma_eta, burn_in_time=burn_in_time, eta_init=None
        )

        # =====================================================================
        # PREPARACIÓN DE COEFICIENTES PARA INTEGRACIÓN TEMPORAL
        # =====================================================================
        # Reference: Main paper Eq. 11, methods.py:402-403
        # Proceso Ornstein-Uhlenbeck para ruido correlacionado
        has_noise = noise_level > 0
        if has_noise:
            L = np.linalg.cholesky(Sigma_eta)
            tau_n_inv = 1.0 / self._integrator.f.tau_n  # τ_η = 20ms
            eps_1 = 1.0 - dt * tau_n_inv  # e^(-dt/τ_η) ≈ 1-dt/τ_η
            eps_2 = np.sqrt(2.0 * dt * tau_n_inv)  # Mantener varianza
        else:
            L = None
            eps_1 = eps_2 = 0.0
            eta_0 = np.zeros(self._N)

        # =====================================================================
        # CONFIGURACIÓN DE ESTRUCTURA TEMPORAL
        # =====================================================================
        # Reference: transients.py lines 41-43
        if use_three_phases:
            steps_init = int(t_init / self._time_res)
            steps_stimulus = int(t_stimulus / self._time_res)
            steps_final = int(t_final / self._time_res)
            simulation_steps = steps_init + steps_stimulus + steps_final
            total_time_ms = t_init + t_stimulus + t_final

            # Generar delays por neurona si se solicitan
            if use_delays:
                delays = self._random.normal(
                    mean_delay / 1000.0, delay_sd / 1000.0, self._N_E
                )
                delays[delays < 0] = 0.0
                delays = np.concatenate((delays, delays))  # Duplicar para I
                print(f"Using per-neuron delays: "
                      f"mean={mean_delay}ms, sd={delay_sd}ms")
            else:
                delays = np.zeros(self._N)
        else:
            # Modo single-phase
            simulation_steps = int(simulation_time / self._time_res)
            total_time_ms = simulation_time
            steps_init = steps_final = 0
            steps_stimulus = simulation_steps
            delays = np.zeros(self._N)

        print(f"Simulation: {total_time_ms:.1f}ms ({simulation_steps} steps)")
        if use_three_phases:
            print(f"  Phase 1 (spontaneous): {t_init:.1f}ms "
                  f"({steps_init} steps)")
            print(f"  Phase 2 (stimulus): {t_stimulus:.1f}ms "
                  f"({steps_stimulus} steps)")
            print(f"  Phase 3 (post-stimulus): {t_final:.1f}ms "
                  f"({steps_final} steps)")

        # =====================================================================
        # LOOP TEMPORAL PRINCIPAL
        # =====================================================================
        # Reference: methods.py network_evolution(), Main paper Eq. 8
        u_old = np.copy(u_0)
        eta_old = np.copy(eta_0)
        u_trajectory = np.zeros((simulation_steps, self._N))
        simulation_successful = True
        actual_steps = simulation_steps

        for step in range(simulation_steps):
            # Determinar estímulo actual según fase y delays
            h_current = self._get_current_stimulus(
                step, dt, use_three_phases, use_delays,
                steps_init, steps_stimulus,
                h_baseline, h_stimulus, delays
            )

            # Actualizar ruido correlacionado (Ornstein-Uhlenbeck)
            if has_noise:
                temp = bp.math.matmul(
                    L, self._random.normal(loc=0.0, scale=1.0, size=self._N)
                )
                eta_new = eps_1 * eta_old + eps_2 * temp
            else:
                eta_new = eta_old

            # Integrar dinámica: Main paper Eq. 8
            u_e_new, u_i_new = self._integrator(
                u_old[:self._N_E], u_old[self._N_E:],
                step * dt, W, h_current, eta_old
            )
            u_new = bp.math.concatenate([u_e_new, u_i_new])

            u_trajectory[step] = u_new
            u_old = bp.math.array(u_new)
            eta_old = bp.math.array(eta_new)

            # Verificar estabilidad
            if bp.math.linalg.norm(u_new) > 1000:
                simulation_successful = False
                actual_steps = step
                print(f"Warning: Activity exploded at step "
                      f"{step}/{simulation_steps}. "
                      f"Using trajectory up to step {step-1}.")
                break

        # =====================================================================
        # POST-PROCESAMIENTO Y RESULTADOS
        # =====================================================================
        if not simulation_successful:
            u_trajectory = u_trajectory[:actual_steps]
            if actual_steps == 0:
                raise RuntimeError(
                    f"Simulation failed immediately. Check network "
                    f"parameters: contrast={stimulus_contrast}, "
                    f"noise_level={noise_level}."
                )

        # Calcular firing rates: r_α(t) = k * [u_α(t)]_+^n (Eq. 9)
        activation = self._integrator.f.supralinear_activation
        response = {
            "excitatory_firing_rate": activation(
                u_trajectory[:, :self._N_E]
            ),
            "inhibitory_firing_rate": activation(
                u_trajectory[:, self._N_E:]
            ),
            "excitatory_potential": u_trajectory[:, :self._N_E],
            "inhibitory_potential": u_trajectory[:, self._N_E:],
        }

        extra = {
            "stimulus_contrast": stimulus_contrast,
            "stimulus_orientation": stimulus_orientation,
            "noise_level": noise_level,
            "burn_in_time": burn_in_time,
            "burn_in_steps": int(burn_in_time / self._time_res),
            "simulation_successful": simulation_successful,
            "actual_simulation_steps": actual_steps,
            "requested_simulation_steps": simulation_steps,
            "actual_simulation_time": actual_steps * self._time_res,
            "requested_simulation_time": simulation_time,
            "use_three_phases": use_three_phases,
            "t_init": t_init if use_three_phases else None,
            "t_stimulus": t_stimulus if use_three_phases else None,
            "t_final": t_final if use_three_phases else None,
            "use_delays": use_delays if use_three_phases else None,
            "used_custom_h_input": h_input is not None,
            "h_stimulus": h_stimulus,  # Guardar h usado para referencia
        }

        return response, extra

    def _validate_stimulus_dimensions(self, stimulus, name):
        """
        Validate stimulus dimensions match network size.

        Parameters
        ----------
        stimulus : array_like
            Stimulus array to validate
        name : str
            Name of the stimulus (for error messages)

        Raises
        ------
        ValueError
            If stimulus dimensions don't match network size
        """
        expected_size = self._N
        if len(stimulus) != expected_size:
            raise ValueError(
                f"{name} dimension mismatch. Expected {expected_size}, "
                f"got {len(stimulus)}. Network: {self._N_E}E + {self._N_I}I."
            )

    def _get_current_stimulus(
        self, step, dt, use_three_phases, use_delays,
        steps_init, steps_stimulus,
        h_baseline, h_stimulus, delays
    ):
        """
        Determine current stimulus based on phase and delays.

        Parameters
        ----------
        step : int
            Current time step
        dt : float
            Time resolution in seconds
        use_three_phases : bool
            Whether using three-phase temporal structure
        use_delays : bool
            Whether using per-neuron delays
        steps_init : int
            Steps in initial spontaneous phase
        steps_stimulus : int
            Steps in stimulus presentation phase
        h_baseline : array
            Baseline stimulus (spontaneous activity)
        h_stimulus : array
            Full stimulus
        delays : array
            Per-neuron delays in seconds

        Returns
        -------
        h_current : array
            Stimulus for current timestep

        References
        ----------
        transients.py lines 131-145
        """
        if not use_three_phases:
            return h_stimulus

        current_time_sec = step * dt

        # Fase 1: Actividad espontánea inicial
        if step < steps_init:
            return h_baseline

        # Fase 2: Presentación del estímulo
        if step < (steps_init + steps_stimulus):
            if not use_delays:
                return h_stimulus
            # Con delays: transición gradual
            phase2_time = current_time_sec - (steps_init * dt)
            h_current = h_baseline.copy()
            arrived = delays <= phase2_time
            h_current[arrived] = h_stimulus[arrived]
            return h_current

        # Fase 3: Retorno a actividad espontánea
        if not use_delays:
            return h_baseline
        # Con delays: transición gradual inversa
        phase3_time = current_time_sec - ((steps_init + steps_stimulus) * dt)
        h_current = h_stimulus.copy()
        arrived = delays <= phase3_time
        h_current[arrived] = h_baseline[arrived]
        return h_current

    def _make_result(
        self,
        modes_dict,
        time_range,
        position_range,
        time_res,
        position_res,
        causes,
        run_parameters,
        extra,
        **kwargs,
    ):
        """
        Custom result maker that handles truncated simulations.

        This method is called by the parent class to create the NDResult.
        It adjusts temporal dimensions if the simulation was truncated.
        """
        from ..core.ndresult import NDResult

        # ALWAYS adjust time_range based on actual data length
        # The time_range parameter comes from self._time_range which is fixed
        # but the actual simulation may be shorter (simulation_time parameter)
        if hasattr(modes_dict, "values") and modes_dict:
            actual_length = len(next(iter(modes_dict.values())))

            # Handle explosion cases by removing problematic last data point
            if isinstance(extra, dict) and not extra.get(
                "simulation_successful", True
            ):
                # For exploded simulations, remove the last data point
                truncated_modes_dict = {}
                for key, data in modes_dict.items():
                    if hasattr(data, "__len__") and len(data) > 0:
                        truncated_modes_dict[key] = data[:-1]  # Remove last
                    else:
                        truncated_modes_dict[key] = data
                modes_dict = truncated_modes_dict
                actual_length = len(next(iter(modes_dict.values())))

            # Fix floating point precision issue in time_range calculation
            # The validation uses: expected = int(time_span/time_res)
            # To ensure int(time_span / time_res) == actual_length,
            # we calculate time_max directly from actual data length
            epsilon = 1e-10
            new_time_max = actual_length * time_res + epsilon
            time_range = (0, new_time_max)

        # Use standard creation (validation should pass now)
        return NDResult.from_modes_dict(
            modes_dict=modes_dict,
            time_range=time_range,
            position_range=position_range,
            time_res=time_res,
            position_res=position_res,
            causes=causes,
            run_parameters=run_parameters,
            extra=extra,
            **kwargs,
        )

    # =========================================================================
    # FUNCIONES KERNEL CENTRALIZADAS (SINGLE SOURCE OF TRUTH)
    # =========================================================================

    @staticmethod
    def _spatial_kernel(angular_diff, width):
        """
        Spatial kernel for connectivity and noise covariance.

        This is the SINGLE SOURCE OF TRUTH for the spatial kernel formula.
        All other functions should use this.

        Formula: exp[(cos(2*Δθ) - 1) / w²]

        Parameters
        ----------
        angular_diff : float or array
            Angular difference θi - θj (in radians)
        width : float
            Width parameter (in radians)

        Returns
        -------
        float or array
            Kernel value

        Notes
        -----
        Factor 2 in cosine is needed because we use θ ∈ [0, π] for
        orientations (non-directional), while Echeveste's OCaml code
        uses θ ∈ [0, 2π]. The factor 2 compensates for this difference.

        References
        ----------
        - Echeveste et al. 2020, Equation 10
        - objective.ml lines 285-287 (uses θ ∈ [0, 2π] without factor 2)
        """
        return np.exp((np.cos(2 * angular_diff) - 1) / (width**2))

    @staticmethod
    def _spatial_kernel_jax(angular_diff, width):
        """
        JAX version of spatial kernel for efficient training.

        Same formula as _spatial_kernel but using JAX operations.

        Parameters
        ----------
        angular_diff : jax.Array
            Angular difference θi - θj (in radians)
        width : float
            Width parameter (in radians)

        Returns
        -------
        jax.Array
            Kernel value
        """
        import jax.numpy as jnp
        return jnp.exp((jnp.cos(2 * angular_diff) - 1) / (width**2))

    def _build_parametric_matrix_vectorized(self, theta_pre, theta_post,
                                            a_xy, d_xy, sign=1.0):
        """
        Build parametric connectivity matrix (vectorized, efficient).

        This replaces ALL loop-based constructions of connectivity blocks.

        Parameters
        ----------
        theta_pre : array_like
            Orientations of pre-synaptic neurons (radians)
        theta_post : array_like
            Orientations of post-synaptic neurons (radians)
        a_xy : float
            Connectivity amplitude
        d_xy : float
            Connectivity width (radians)
        sign : float, optional
            Sign of connection (+1 for excitatory, -1 for inhibitory)

        Returns
        -------
        np.ndarray
            Connectivity matrix of shape (len(theta_post), len(theta_pre))
            where W[i,j] = connection from pre[j] to post[i]
        """
        theta_pre = np.asarray(theta_pre)
        theta_post = np.asarray(theta_post)

        # Compute all pairwise angular differences
        # Shape: (len(theta_post), len(theta_pre))
        delta = theta_post[:, None] - theta_pre[None, :]

        # Apply spatial kernel
        kernel = self._spatial_kernel(delta, d_xy)

        # Apply amplitude and sign
        return sign * a_xy * kernel

    def _build_noise_covariance_vectorized(self, width, std_e, std_i, rho):
        """
        Build noise covariance matrix (vectorized, efficient).

        This replaces the loop-based construction in _build_noise_covariance.

        Parameters
        ----------
        width : float
            Width parameter for spatial correlation (radians)
        std_e : float
            Standard deviation for excitatory noise
        std_i : float
            Standard deviation for inhibitory noise
        rho : float
            Cross-correlation coefficient between E and I populations

        Returns
        -------
        np.ndarray
            Noise covariance matrix of shape (N, N) where N = 2*N_E

        Notes
        -----
        Uses _spatial_kernel for consistency with connectivity matrices.

        References
        ----------
        - objective.ml lines 290-305 (sigma_eta function)
        """
        N_E = self._N_E
        N = 2 * N_E

        # Generar orientaciones para población E
        theta = np.linspace(0, np.pi, N_E, endpoint=False)

        # Compute pairwise angular differences
        delta = theta[:, None] - theta[None, :]

        # Apply spatial kernel
        spatial_kernel = self._spatial_kernel(delta, width)

        # Compute variances
        var_e = std_e ** 2
        var_i = std_i ** 2

        # Build blocks
        Sigma_ee = var_e * spatial_kernel
        Sigma_ii = var_i * spatial_kernel
        Sigma_ei = rho * np.sqrt(var_e * var_i) * spatial_kernel
        Sigma_ie = Sigma_ei.T  # Symmetric

        # Assemble full matrix
        Sigma = np.zeros((N, N))
        Sigma[:N_E, :N_E] = Sigma_ee
        Sigma[N_E:, N_E:] = Sigma_ii
        Sigma[:N_E, N_E:] = Sigma_ei
        Sigma[N_E:, :N_E] = Sigma_ie

        # Add diagonal noise for numerical stability
        # Ref: objective.ml line 305
        Sigma += 0.01 * np.eye(N)

        return Sigma

    def parametric_connectivity(self, theta_i, theta_j, a_xy, d_xy):
        """
        Calculate parametric connectivity using the formula in Equation 10.

        Implement: W_XY(θi, θj) = a_XY * exp[(cos(2*(θi - θj)) - 1) / d_XY²]

        Parameters
        ----------
        theta_i : float
            Preferred orientation of neuron i (in radians)
        theta_j : float
            Preferred orientation of neuron j (in radians)
        a_xy : float
            Connectivity amplitude between X→Y populations
        d_xy : float
            Connectivity width (dispersion parameter, in radians)

        Returns
        -------
        float
            Connectivity strength from j to i

        Notes
        -----
        Uses _spatial_kernel as single source of truth for the formula.

        References
        ----------
        - Echeveste et al. 2020, Equation 10
        - objective.ml lines 285-287
        """
        angular_diff = theta_i - theta_j
        kernel = self._spatial_kernel(angular_diff, d_xy)
        return a_xy * kernel

    def build_connectivity_matrix(self, connectivity_params=None):
        """
        Construct connectivity matrix using parametric formulation (Eq.10).

        Mathematical foundation:
        - Main paper, Eq. 10: W_XY(θi,θj) = a_XY * exp[(cos(θi-θj)-1)/d_XY²]
        - Only 8 parameters: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
        - Ref: objective.ml lines 285-287 (SIN factor 2 en coseno)
        - Echeveste et al. (2020) Supplementary Material: Connectivity
          Parameter Optimization

        Parameters
        ----------
        connectivity_params: dict, optional
            Dictionary with the 8 connectivity parameters:
            - 'a_EE', 'a_EI', 'a_IE', 'a_II': connectivity amplitudes
            - 'd_EE', 'd_EI', 'd_IE', 'd_II': connectivity dispersions
            If None, uses stored parameters from training.

        Returns
        -------
        W: np.ndarray, shape(N, N)
            Complete connectivity matrix for use in SSNIntegrator
        """
        # Use trained parameters if available, otherwise use provided params
        if connectivity_params is None:
            if not self._stage1_completed:
                raise ValueError(
                    """Stage 1 must be completed or
                    connectivity_params must be provided"""
                )

            # Check if we have the full matrix loaded as fallback
            if hasattr(self, "_W_full") and self._W_full is not None:
                return self._W_full
            # Use stored trained parameters
            params = self._get_connectivity_parameters()
        else:
            params = connectivity_params

        # If matrices already built and size matches exactly, return assembled
        # matrix
        if (
            self._W_EE is not None
            and self._W_EI is not None
            and self._W_IE is not None
            and self._W_II is not None
            and self._W_EE.shape == (self._N_E, self._N_E)
            and self._W_EI.shape == (self._N_E, self._N_I)
            and self._W_IE.shape == (self._N_I, self._N_E)
            and self._W_II.shape == (self._N_I, self._N_I)
        ):
            W_full = np.zeros((self._N, self._N))
            W_full[:self._N_E, :self._N_E] = self._W_EE
            W_full[:self._N_E, self._N_E:] = self._W_EI
            W_full[self._N_E:, :self._N_E] = self._W_IE
            W_full[self._N_E:, self._N_E:] = self._W_II
            return W_full

        # Construir W exactamente como objective.ml líneas 278-288
        # IMPORTANTE: Replicar exactamente la lógica de indexación de OCaml
        # para obtener la matriz idéntica a w_learn
        #
        # Código original OCaml (objective.ml:278-288):
        #   M.init n n (fun i j ->
        #     let p, s = if i<=m
        #       then (if j<=m then ee,1 else ei,-1)
        #       else (if j<=m then ie,1 else ii,-1) in
        #     let i = (pred i) mod m and j = (pred j) mod m in
        #     let theta_i = 2*pi*float(i)/float(m) in
        #     let theta_j = 2*pi*float(j)/float(m) in
        #     (s*p.height) * exp(cos_diff_minus_one / width²))
        #
        # Diferencia clave: OCaml usa índices 1-based (1..n), luego
        # hace (i-1) mod m para obtener índice angular 0..(m-1)

        W = np.zeros((self._N, self._N))

        # Determinar m (número de orientaciones por población)
        # En el caso original: N_E = N_I = m = 50, n = 100
        m = self._N_E  # Asumimos N_E = N_I como en el original

        # Iterar con índices estilo OCaml (1..n) para replicar exactamente
        for i_ocaml in range(1, self._N + 1):
            for j_ocaml in range(1, self._N + 1):
                # Determinar qué bloque y signo según lógica OCaml
                # if i<=m then (if j<=m then ee,1 else ei,-1)
                #         else (if j<=m then ie,1 else ii,-1)
                if i_ocaml <= m:  # i es E
                    if j_ocaml <= m:  # j es E → bloque E←E
                        a, d, sign = (params["a_EE"], params["d_EE"], 1.0)
                    else:  # j es I → bloque E←I
                        a, d, sign = (params["a_EI"], params["d_EI"], -1.0)
                else:  # i es I
                    if j_ocaml <= m:  # j es E → bloque I←E
                        a, d, sign = (params["a_IE"], params["d_IE"], 1.0)
                    else:  # j es I → bloque I←I
                        a, d, sign = (params["a_II"], params["d_II"], -1.0)

                # Convertir índices OCaml (1-based) a índices angulares
                # let i = (pred i) mod m → i_angular = (i_ocaml - 1) mod m
                i_angular = (i_ocaml - 1) % m
                j_angular = (j_ocaml - 1) % m

                # Calcular ángulos: theta = 2π * i / m
                theta_i = 2.0 * np.pi * i_angular / m
                theta_j = 2.0 * np.pi * j_angular / m

                # Diferencia angular
                cos_diff_minus_one = np.cos(theta_i - theta_j) - 1.0

                # Fórmula de conectividad: (sign * a) * exp(cos_diff/d²)
                W[i_ocaml - 1, j_ocaml - 1] = (
                    (sign * a) * np.exp(cos_diff_minus_one / (d ** 2))
                )

        return W

    def _generate_simple_stimulus(self, contrast):
        """
        Generate simple uniform stimulus matching original code.

        The original Echeveste code uses uniform input h = constant
        for all neurons, not complex GSM-generated oriented stimuli.
        """
        # Evidencia del código original Echeveste2020:
        # 1. activity_example.py:55: h[alpha] = np.loadtxt("h_true_"+str(α))
        # 2. h_true_0_learn: UNIFORME = [0.01928, 0.01928, ...] (100 iguales)
        # 3. h_true_1-4_learn: ORIENTADOS con patrón espacial (gabor)
        # 4. Casos GSM: h = h_scale*x_proj del modelo generativo
        # Patrón 0 es uniforme (baseline), patrones 1-4 son orientados

        stimulus = np.full(self._N_E + self._N_I, contrast)
        return stimulus

    def _generate_gsm_stimulus(self, contrast, orientation=0.0,
                               use_precomputed=True):
        """
        Generate stimulus from Gaussian Scale Mixture (GSM) generative model.

        IMPORTANT: This method loads precomputed h_true files that ALREADY
        have the nonlinear transformation applied. No additional
        transformation is needed.

        Mathematical foundation:
        - Main paper, Eq. 1-3: GSM generative model I = z * G
        - Main paper, Eq. 6-7: Linear projection h_vec = (A^T / 15) @ I
        - Trained transformation: h = α_h * max(h_vec + β_h, 0)^γ_h
        - Echeveste et al. (2020) Supplementary Material Table S1

        Parameters
        ----------
        contrast : float
            Contrast level (ignored if use_precomputed=True)
        orientation : float, optional
            Orientation in degrees (ignored if use_precomputed=True)
        use_precomputed : bool, optional
            If True (default), loads precomputed h_true from files.
            This is the recommended mode to replicate paper results.
            If False, raises NotImplementedError (future feature).

        Returns
        -------
        h_full : np.ndarray, shape (N_E + N_I,)
            Transformed SSN input for full network.

        Notes
        -----
        The precomputed h_true files
        were generated by div_norm_data_GSM.py WITH the transformation
        already applied. The files contain:

        1. Raw filter response: h_vec = (A.T / 15.0) @ x
        2. Nonlinear transformation: h = α_h * (h_vec + β_h)^γ_h
        3. Duplication for E+I: h_full = [h; h]

        This matches the original Echeveste code (full_net.py line 45):
            h[alpha] = np.loadtxt("h_true_"+str(alpha)+"_learn")

        Contrast mapping (based on available h_true files):
        - contrast < 0.06  → h_true_0 (spontaneous, z=0.0)
        - 0.06 ≤ contrast < 0.19 → h_true_1 (low, z≈0.125)
        - 0.19 ≤ contrast < 0.38 → h_true_2 (medium, z≈0.25)
        - 0.38 ≤ contrast < 0.75 → h_true_3 (high, z≈0.5)
        - contrast ≥ 0.75 → h_true_4 (very high, z≈1.0)

        References
        ----------
        .. [1] div_norm_data_GSM.py lines 236-241, 352-353:
               get_true_h_from_x_proj() generates h_true WITH transformation
        .. [2] full_net.py line 45:
               Original code loads h_true directly WITHOUT transformation
        .. [3] parameters.md lines 32-34:
               Transformation parameters (α_h=1.96, β_h=0.10, γ_h=2.03)
        """
        try:
            # Initialize GSM if not already created
            if not hasattr(self, "_gsm"):
                self._gsm = self._create_gsm()

            if use_precomputed:
                # Map contrast level to file index
                # Echeveste's contrast levels: [0.0, 0.125, 0.25, 0.5, 1.0]
                if contrast < 0.06:
                    contrast_idx = 0  # Spontaneous activity (z=0.0)
                elif contrast < 0.19:
                    contrast_idx = 1  # Low contrast (z≈0.125)
                elif contrast < 0.38:
                    contrast_idx = 2  # Medium contrast (z≈0.25)
                elif contrast < 0.75:
                    contrast_idx = 3  # High contrast (z≈0.5)
                else:
                    contrast_idx = 4  # Very high contrast (z≈1.0)

                # Load precomputed h_true directly from files
                # IMPORTANT: h_true files already have transformation applied
                # No additional transformation is needed
                # Ref: full_net.py line 45 loads h_true directly
                h_full = self._gsm.load_precomputed_h_transformed(
                    contrast_idx
                )

                # Validate dimensions
                expected_size = self._N_E + self._N_I
                if len(h_full) != expected_size:
                    raise ValueError(
                        f"Stimulus dimension mismatch. "
                        f"Expected {expected_size}, got {len(h_full)}. "
                        f"Network: {self._N_E}E + {self._N_I}I = "
                        f"{expected_size} total. "
                        f"h_true loaded for contrast_idx={contrast_idx}"
                    )

                return h_full

            else:
                # Future feature: Generate new stimulus with GSM
                # This would involve:
                # 1. Generate image x with GSM: x = z * (A @ y)
                # 2. Project with filters: h_vec = W_ff @ x
                # 3. Apply transformation: h = α_h * max(h_vec + β_h, 0)^γ_h
                # 4. Duplicate for I neurons
                raise NotImplementedError(
                    "Dynamic GSM stimulus generation not yet implemented. "
                    "Use use_precomputed=True to load from files. "
                    f"Requested contrast={contrast}, orientation={orientation}"
                )

        except Exception as e:
            if "dimension mismatch" in str(e).lower():
                raise  # Re-raise our custom dimension errors
            elif isinstance(e, NotImplementedError):
                raise  # Re-raise NotImplementedError
            else:
                raise RuntimeError(
                    f"Failed to generate GSM stimulus: {e}. "
                    f"Network size: {self._N_E}E + {self._N_I}I = "
                    f"{self._N_E + self._N_I} total. "
                    f"Check that GSM data files exist and are accessible."
                ) from e

    def _create_gsm(self):
        """Create GSM instance for stimulus generation.

        Returns
        -------
        Echeveste2020GSM
            GSM instance configured for this SSN model.
        """
        try:
            from ..generative import GSM

            # Create GSM with parameters matching this SSN
            gsm = GSM(
                patch_size=16,
                n_orientations=self._N,  # Match SSN neuron count
                h_scale=1.0 / 15.0,  # Echeveste's scaling factor
                gamma=0.0,  # No contrast-proportional term
                spatial_freq=0.2,
                bandwidth=1.0,
                correlation_strength=0.5,
                noise_variance=0.01,
                random_seed=getattr(self, "_random_seed", None),
                use_pretrained=True,  # Use internal data loader
            )

            return gsm

        except ImportError:
            raise ImportError(
                "GSM implementation not available. "
                "Please ensure the generative "
                "module is properly installed."
            )

    def generate_stimulus_from_gsm(self, contrast=0.5, n_samples=1):
        """Generate stimulus using GSM model.

        Parameters
        ----------
        contrast: float, optional
            Contrast level for stimulus generation (0-1). Default: 0.5.
        n_samples: int, optional
            Number of stimulus samples to generate. Default: 1.

        Returns
        -------
        numpy.ndarray
            Generated stimulus h vectors of shape (n_samples, N_neurons).
        """
        if not hasattr(self, "_gsm"):
            self._gsm = self._create_gsm()

        # Map contrast to Echeveste's levels and generate samples
        contrast_level = contrast * 4.0
        h_samples = self._gsm.get_h_for_contrast_level(
            contrast_level, n_samples
        )

        return h_samples

    def calculate_causes(
        self,
        network_activity=None,
        stimulus=None,
        contrast_range=None,
        **kwargs,
    ):
        """
        Extract causal inference from SSN sampling-based dynamics.

        This method implements the core causal inference mechanism from
        Echeveste et al. (2020), analyzing network activity patterns to
        infer the number and properties of underlying causes.

        Mathematical foundation:
        - Main paper, Section 2.1: "Networks approximate
          Bayesian inference by sampling from the posterior
          distribution P(z,G|I)"
        - Main paper, Eq. 10: Posterior sampling through
          network dynamics
        - Main paper, Figure 4: Population activity reflects
          posterior statistics
        - Main paper, Figure 5: Causal inference from multi-modal posterior
        - Echeveste et al. (2020) Supplementary Material: Inference analysis
          methods

        Parameters
        ----------
        network_activity : array_like, optional
            Steady-state activity of SSN network (N,) where N = N_E + N_I.
            If None, uses current network state.
        stimulus : array_like, optional
            Input stimulus for context. Used for validation.
        contrast_range : array_like, optional
            Range of contrast values to evaluate.
            Default: np.linspace(0, 5, 201)
        **kwargs : dict
            Additional parameters:
            - peak_threshold : float, minimum peak height (default: 0.1)
            - peak_distance : int, minimum distance between peaks (default: 10)
            - confidence_threshold : float, minimum confidence for
              valid cause (default: 0.5)

        Returns
        -------
        dict
            Dictionary containing causal inference results:
            - 'num_causes' : int, number of detected causes
            - 'cause_positions' : list, orientation angles of detected
              causes (degrees)
            - 'cause_contrasts' : list, estimated contrast levels of
              causes
            - 'confidence' : list, confidence scores for each cause [0,1]
            - 'posterior_distribution' : dict, complete posterior P(z|x)
                - 'contrast_values' : array, contrast grid
                - 'probabilities' : array, posterior probabilities
                - 'map_estimate' : float, maximum a posteriori contrast
            - 'posterior_stats' : dict, posterior statistics
                - 'mean' : float, posterior mean contrast
                - 'std' : float, posterior standard deviation
                - 'modes' : list, all detected modes
        """
        # Extraer parámetros de control para la detección de causas
        # Basado en el análisis de picos de distribución posterior
        # altura mínima de un pico para ser considerado (muy sensible)
        peak_threshold = kwargs.get("peak_threshold", 0.001)
        # distancia mínima entre picos (muy tolerante)
        peak_distance = kwargs.get("peak_distance", 3)
        # confianza mínima comparado con la probabilidad máxima (muy tolerante)
        confidence_threshold = kwargs.get("confidence_threshold", 0.1)

        # Establecer rango de contraste por defecto
        # Coincide con el rango usado en Echeveste et al. (2020) - Fig. 3
        # 201 puntos → paso de 0.025 en contraste
        if contrast_range is None:
            contrast_range = np.linspace(0.0, 5.0, 201)

        # Obtener o generar actividad de red
        # Paso fundamental: necesitamos la respuesta de estado estable del SSN
        # que representa muestras de la distribución posterior P(z,G|I)
        if network_activity is None:
            # Try to extract network activity from kwargs
            # (when called by parent class)
            if ("excitatory_firing_rate" in kwargs and
                    "inhibitory_firing_rate" in kwargs):
                # Extract final timestep from simulation results
                exc_activity = kwargs["excitatory_firing_rate"]
                inh_activity = kwargs["inhibitory_firing_rate"]

                # Use the final timestep as steady-state activity
                if exc_activity.ndim > 1:
                    final_exc = exc_activity[-1]  # Last timestep
                else:
                    final_exc = exc_activity

                if inh_activity.ndim > 1:
                    final_inh = inh_activity[-1]  # Last timestep
                else:
                    final_inh = inh_activity

                network_activity = np.concatenate([final_exc, final_inh])

            elif stimulus is not None:
                # Generar respuesta de red al estímulo usando dinámica SSN
                network_activity = self._generate_network_response(stimulus)
            else:
                raise ValueError(
                    "Either network_activity or stimulus must be provided, "
                    "or excitatory/inhibitory activity must be in kwargs"
                )

        # Validar dimensiones de actividad de red
        # Debe tener N = N_E + N_I elementos
        # (50 excitatorias + 50 inhibitorias)
        if len(network_activity) != self._N:
            raise ValueError(
                f"Network activity must have length "
                f"{self._N}, got {len(network_activity)}"
            )

        # PASO 1: Extraer distribución posterior de la actividad de red
        # Fundamento teórico: Echeveste et al. Fig. 2 - la actividad de red
        # refleja muestras de P(z|x) donde z es contraste y x es input visual
        posterior_dist = self._extract_posterior_distribution(
            network_activity, contrast_range
        )

        # PASO 2: Detectar picos/modos en la posterior (causas potenciales)
        # Fundamento: Echeveste et al. Fig. 5 - distribuciones multi-modales
        # indican múltiples causas en la escena visual
        peaks_info = self._detect_posterior_peaks(
            posterior_dist, peak_threshold, peak_distance
        )

        # PASO 3: Extraer estadísticas causales y filtrar por confianza
        # Calcula confianza basada en altura de picos relativos
        causal_results = self._extract_causal_statistics(
            peaks_info, posterior_dist, confidence_threshold
        )

        # PASO 4: Extraer orientaciones de las causas
        orientation_results = self._extract_cause_orientations(
            network_activity, causal_results["cause_contrasts"]
        )

        # Return full results dictionary
        return {
            "num_causes": causal_results["num_causes"],
            "cause_positions": orientation_results["orientations"],
            "cause_contrasts": causal_results["cause_contrasts"],
            "contrast_confidence": causal_results["confidence_scores"],
            "orientation_confidence": orientation_results["confidence_scores"],
            "posterior_distribution": {
                "contrast_values": posterior_dist["contrast_values"],
                "probabilities": posterior_dist["probabilities"],
                "map_estimate": posterior_dist["map_estimate"],
            },
            "posterior_stats": causal_results["posterior_stats"],
        }

    def _generate_network_response(self, stimulus):
        """
        Generate SSN network response to a given stimulus.

        This is a simplified implementation that maps stimulus to network
        activity.
        In the full implementation, this would involve running the SSN dynamics
        to steady state.

        Parameters
        ----------
        stimulus : array_like
            Input stimulus (typically from GSM)

        Returns
        -------
        network_activity : ndarray
            Steady-state activity of network (N_E + N_I,)
        """
        # Simplified implementation: use supralinear activation on stimulus
        if len(stimulus) != self._N:
            # If stimulus dimension doesn't match network, extend it
            h_extended = np.zeros(self._N)
            h_extended[: min(len(stimulus), self._N)] = stimulus[
                : min(len(stimulus), self._N)
            ]
        else:
            h_extended = stimulus.copy()

        # Apply network nonlinearity to get activity
        network_activity = self._integrator.f.supralinear_activation(
            h_extended
        )

        return network_activity

    def _extract_posterior_distribution(
        self, network_activity, contrast_range
    ):
        """
        Extract posterior distribution P(z|x) using EXACT Echeveste et al.

        This implements the exact same formula as the original Echeveste code:
        P(z|x) = P(z) * P(x|z) / P(x)

        Where:
        - P(z) is gamma prior over contrasts
        - P(x|z) is multivariate normal likelihood
        - Covariance matrix: Cov = z^2 * A*C*A^T + s_x^2 * I

        This is the same implementation from GSM.py line 210-227.
        """
        # Load GSM data using the correct data loader
        from ..data.gsm_data_loader import GSMDataLoader

        gsm_loader = GSMDataLoader()
        A = gsm_loader.load_gabor_filters()  # Gabor filters (256, 50)
        C = gsm_loader.load_prior_covariance()  # Covariance matrix (50, 50)

        # Map network activity to GSM observation space
        # Network activity (100,) -> GSM observation x (256,)
        # The network has N_E=50 excitatory neurons representing orientations
        # We need to create a 256-dimensional observation that matches GSM

        excitatory_activity = network_activity[: self._N_E]  # (50,)

        # Create synthetic GSM observation from network activity
        # This maps SSN activity back to visual observation space
        # Using the transpose: if x = A*y + noise, then y ~ A^T * x

        # Method 1: Direct mapping using transpose
        # We construct a 256D observation that would
        # produce this activity pattern
        # Note: Using JAX for GPU acceleration
        import jax.numpy as jnp

        x_reconstructed = jnp.dot(A, excitatory_activity)  # (256,)

        # Add realistic noise level based on GSM parameters
        if jnp.std(x_reconstructed) > 0:
            noise_scale = 0.1 * jnp.std(x_reconstructed)
        else:
            noise_scale = 0.1
        noise = np.random.normal(0, noise_scale, len(x_reconstructed))
        x_observation = x_reconstructed + noise

        # EXACT ECHEVESTE FORMULA - GSM.py line 210-227
        # Compute P(z|x) for each contrast value in contrast_range

        n_contrasts = len(contrast_range)
        D_x = len(x_observation)
        log_p = jnp.zeros(n_contrasts)

        # Pre-compute matrices for efficiency (GPU accelerated)
        ACA_T = jnp.dot(A, jnp.dot(C, A.T))  # A*C*A^T
        mean_x = jnp.zeros(D_x)  # Mean is always zero in GSM

        # GSM noise variance (from original code parameters)
        s_x_2 = 100.0  # This matches original Echeveste parameters

        # Gamma prior parameters (from original code)
        k_gamma = 2.0  # Shape parameter
        theta_gamma = 0.5  # Scale parameter

        dz = (
            contrast_range[1] - contrast_range[0]
            if len(contrast_range) > 1
            else 0.1
        )

        for i, z in enumerate(contrast_range):
            # Likelihood: P(x|z) ~ N(0, z^2 * A*C*A^T + s_x^2 * I)
            covariance = z * z * ACA_T + s_x_2 * jnp.eye(D_x)

            try:
                # Log prior: P(z) ~ Gamma(k, theta)
                if z > 0:
                    from scipy.stats import gamma

                    log_prior = gamma.logpdf(z, k_gamma, scale=theta_gamma)
                else:
                    log_prior = -jnp.inf  # Zero prior for negative contrasts

                # Log likelihood: P(x|z) ~ N(0, Cov)
                from scipy.stats import multivariate_normal

                log_likelihood = multivariate_normal.logpdf(
                    x_observation, mean_x, covariance
                )

                # Log posterior = log prior + log likelihood
                log_p = log_p.at[i].set(log_prior + log_likelihood)

            except (np.linalg.LinAlgError, ValueError):
                # Handle numerical issues with singular covariance
                log_p = log_p.at[i].set(-jnp.inf)

        # Normalize probabilities (exactly as in original code)
        max_log_p = jnp.max(log_p)
        p_unnorm = jnp.exp(log_p - max_log_p)
        norm = jnp.sum(p_unnorm) * dz

        if norm > 0:
            probabilities = p_unnorm / norm
        else:
            # Fallback: uniform distribution
            probabilities = jnp.ones(n_contrasts) / n_contrasts

        # Find MAP estimate
        map_idx = jnp.argmax(probabilities)
        map_estimate = contrast_range[map_idx]

        return {
            "contrast_values": contrast_range,
            "probabilities": probabilities,
            "map_estimate": map_estimate,
        }

    def _detect_peaks(self, probabilities, **kwargs):
        """Alias for compatibility with debug functions."""
        from scipy.signal import find_peaks

        peak_threshold = kwargs.get("peak_threshold", 0.1)
        peak_distance = kwargs.get("peak_distance", 10)

        peaks, _ = find_peaks(
            probabilities, height=peak_threshold, distance=peak_distance
        )
        return peaks

    def _detect_posterior_peaks(
        self, posterior_dist, peak_threshold, peak_distance
    ):
        """
        Detect peaks in posterior distribution that correspond to
        potential causes.

        Parameters
        ----------
        posterior_dist : dict
            Posterior distribution from _extract_posterior_distribution
        peak_threshold : float
            Minimum peak height
        peak_distance : int
            Minimum distance between peaks

        Returns
        -------
        peaks_info : dict
            Information about detected peaks:
            - 'peak_indices': indices of peaks in contrast_values array
            - 'peak_heights': heights of peaks (probability values)
            - 'peak_contrasts': contrast values at peaks
        """
        from scipy.signal import find_peaks

        # Extraer valores de probabilidad y contraste
        # de la distribución posterior
        # Fundamento teórico: Echeveste et al. Fig. 5 -
        # análisis de distribuciones
        # multi-modales para inferencia causal
        probabilities = posterior_dist["probabilities"]
        contrast_values = posterior_dist["contrast_values"]

        # Detectar picos en la distribución posterior usando scipy.signal
        # Fundamento: Echeveste et al. Suplementario Sec. 3.2 - "peaks in the
        # posterior distribution correspond to likely cause configurations"
        # Los picos representan modos de la distribución P(z|x) que indican
        # diferentes causas potenciales en la escena visual

        # CRITICAL FIX: Exclude zero contrast from peak detection
        # Zero contrast is not a meaningful "cause" - we need to find peaks
        # in the positive contrast range that represent actual visual stimuli
        non_zero_mask = (
            contrast_values > 0.001
        )  # Small threshold to avoid numerical issues
        non_zero_indices = np.where(non_zero_mask)[0]

        if len(non_zero_indices) == 0:
            # No non-zero contrasts available
            peaks = np.array([], dtype=int)
            properties = {}
        else:
            # Apply peak detection only to non-zero contrast range
            non_zero_probs = probabilities[non_zero_indices]

            peaks_relative, properties = find_peaks(
                non_zero_probs,
                # umbral mínimo de altura para considerar un pico
                height=peak_threshold,
                # distancia mínima entre picos para evitar ruido
                distance=peak_distance,
            )

            # Convert relative indices back to absolute indices
            peaks = (
                non_zero_indices[peaks_relative]
                if len(peaks_relative) > 0
                else np.array([], dtype=int)
            )

        # Extraer información específica de cada pico detectado
        # Las alturas representan la probabilidad posterior de cada causa
        # Los contrastes representan los valores de z (variable latente)
        # más probables
        peak_heights = probabilities[peaks]
        peak_contrasts = contrast_values[peaks]

        return {
            "peak_indices": peaks,
            "peak_heights": peak_heights,
            "peak_contrasts": peak_contrasts,
            "properties": properties,
        }

    def _extract_causal_statistics(
        self, peaks_info, posterior_dist, confidence_threshold
    ):
        """
        Extract causal statistics from detected peaks.

        Parameters
        ----------
        peaks_info : dict
            Peak information from _detect_posterior_peaks
        posterior_dist : dict
            Posterior distribution
        confidence_threshold : float
            Minimum confidence for valid causes

        Returns
        -------
        causal_results : dict
            Causal inference results:
            - 'num_causes': number of valid causes
            - 'cause_contrasts': contrast values of causes
            - 'confidence_scores': confidence for each cause
            - 'posterior_stats': posterior statistics
        """
        # Importar con compatibilidad hacia atrás
        try:
            from scipy.integrate import trapezoid as trapz
        except ImportError:
            from scipy.integrate import trapz

        probabilities = posterior_dist["probabilities"]
        contrast_values = posterior_dist["contrast_values"]

        # Calcular estadísticas de la distribución posterior P(z|x)
        # Fundamento teórico: Echeveste et al. (2020) - inferencia bayesiana
        # Estas estadísticas caracterizan la incertidumbre sobre las causas

        # Media posterior: E[z|x] = ∫ z P(z|x) dz
        # (ecuación estándar de esperanza condicional)
        # Representa el contraste esperado dado el input visual
        posterior_mean = trapz(
            contrast_values * probabilities, contrast_values
        )

        # Varianza posterior: Var[z|x] = ∫ (z - E[z|x])² P(z|x) dz
        # Cuantifica la incertidumbre sobre el contraste inferido
        posterior_var = trapz(
            (contrast_values - posterior_mean) ** 2 * probabilities,
            contrast_values,
        )
        posterior_std = np.sqrt(posterior_var)

        # Procesar picos detectados para extraer causas válidas
        # Fundamento: Echeveste et al. Fig. 5 - cada pico representa una causa
        # potencial con su respectiva probabilidad
        peak_contrasts = peaks_info["peak_contrasts"]
        peak_heights = peaks_info["peak_heights"]

        # Calcular scores de confianza basados en altura relativa del pico
        # Fundamento teórico: la altura normalizada del pico indica qué tan
        # probable es esa causa comparada con la más probable CAUSA REAL
        #
        # Calculate confidence relative to non-zero contrast peaks only
        # Zero contrast is not a meaningful "cause" - it just means no stimulus
        # We need to compare real causes against each other,
        # not against "no cause"

        # Find the maximum probability among non-zero contrasts only
        contrast_values = posterior_dist["contrast_values"]
        non_zero_mask = contrast_values > 0.001

        if np.any(non_zero_mask):
            non_zero_probs = probabilities[non_zero_mask]
            max_prob_non_zero = (
                np.max(non_zero_probs) if len(non_zero_probs) > 0 else 1.0
            )
        else:
            max_prob_non_zero = 1.0  # Fallback

        confidence_scores = (
            peak_heights / max_prob_non_zero
            if max_prob_non_zero > 0
            else np.zeros_like(peak_heights)
        )

        # Filtrar picos por umbral de confianza
        # Solo consideramos causas con confianza suficientemente alta
        # Fundamento: Echeveste et al. - filtrado de hipótesis
        # causales por evidencia
        valid_peaks = confidence_scores >= confidence_threshold
        valid_contrasts = peak_contrasts[valid_peaks]
        valid_confidence = confidence_scores[valid_peaks]

        # Ordenar por confianza (mayor confianza primero)
        # Esto prioriza las causas más probables según la inferencia Bayesiana
        sort_indices = np.argsort(valid_confidence)[::-1]
        valid_contrasts = valid_contrasts[sort_indices]
        valid_confidence = valid_confidence[sort_indices]

        return {
            "num_causes": len(valid_contrasts),
            "cause_contrasts": valid_contrasts.tolist(),
            "confidence_scores": valid_confidence.tolist(),
            "posterior_stats": {
                "mean": posterior_mean,
                "std": posterior_std,
                "modes": peak_contrasts.tolist(),
            },
        }

    def _extract_cause_orientations(self, network_activity, cause_contrasts):
        """
        Extract orientation angles of causes from spatial activity patterns.

        IMPORTANT: This implements proper orientation extraction based on
        spatial peaks in excitatory activity, not weighted by contrast.

        The number of detected causes should match the number of spatial
        peaks in the excitatory activity pattern.

        Parameters
        ----------
        network_activity : array_like
            Network activity (N,)
        cause_contrasts : list
            Contrast values of detected causes

        Returns
        -------
        dict
            Dictionary containing:
            - 'orientations': list of orientation angles in degrees
              for each cause
            - 'confidence_scores': list of confidence scores for
              each orientation
        """
        from scipy.signal import find_peaks

        # Extraer actividad excitatoria (sintonizada a orientación)
        # Fundamento teórico: Echeveste et al. Sec. 2.1 - solo las neuronas
        # excitatorias representan las variables latentes del modelo GSM
        excitatory_activity = network_activity[: self._N_E]

        # Calcular orientaciones preferidas para neuronas excitatorias
        # Fundamento: topología de anillo con neuronas organizadas según
        # su orientación preferida (ring topology)
        orientation_range = np.linspace(
            self._position_range[0], self._position_range[1], self._N_E
        )

        # CORRECT APPROACH: Find spatial peaks in excitatory activity
        # Each peak corresponds to a different oriented stimulus/cause
        # The HEIGHT of peaks relates to contrast, the POSITION to orientation

        # Find peaks in spatial activity pattern
        # Use relatively permissive parameters to detect multiple causes
        min_peak_height = 0.1 * np.max(excitatory_activity)
        min_peak_distance = max(
            1, self._N_E // 10
        )  # At least N_E/10 neurons apart

        peaks, properties = find_peaks(
            excitatory_activity,
            height=min_peak_height,
            distance=min_peak_distance,
        )

        # Extract orientations from peak positions
        peak_orientations = [orientation_range[peak] for peak in peaks]

        # Handle mismatch between detected contrasts and spatial peaks
        num_contrasts = len(cause_contrasts)
        num_spatial_peaks = len(peak_orientations)

        if num_spatial_peaks == 0:
            # Fallback: use global maximum if no peaks found
            global_peak = np.argmax(excitatory_activity)
            _ = excitatory_activity[global_peak]  # noqa: F841
            return {
                "orientations": [orientation_range[global_peak]]
                * num_contrasts,
                "confidence_scores": [1.0]
                * num_contrasts,  # Full confidence for single peak
            }

        elif num_spatial_peaks == num_contrasts:
            # Perfect match: each contrast gets one orientation
            peak_strengths = excitatory_activity[peaks]
            max_strength = np.max(peak_strengths)
            confidence_scores = (
                peak_strengths / max_strength
                if max_strength > 0
                else np.ones_like(peak_strengths)
            )
            return {
                "orientations": peak_orientations,
                "confidence_scores": confidence_scores.tolist(),
            }

        elif num_spatial_peaks < num_contrasts:
            # More contrasts than spatial peaks: some causes share orientations
            peak_strengths = excitatory_activity[peaks]
            max_strength = np.max(peak_strengths)
            confidence_scores = (
                peak_strengths / max_strength
                if max_strength > 0
                else np.ones_like(peak_strengths)
            )

            # Replicate orientations and confidences to
            # match number of contrasts
            repeated_orientations = []
            repeated_confidences = []
            for i in range(num_contrasts):
                idx = i % num_spatial_peaks
                repeated_orientations.append(peak_orientations[idx])
                repeated_confidences.append(confidence_scores[idx])

            return {
                "orientations": repeated_orientations,
                "confidence_scores": repeated_confidences,
            }

        else:
            # More spatial peaks than contrasts: select strongest peaks
            # Sort peaks by activity strength and take top N
            peak_strengths = excitatory_activity[peaks]
            strongest_indices = np.argsort(peak_strengths)[-num_contrasts:]
            strongest_peaks = peaks[strongest_indices]
            strongest_strengths = peak_strengths[strongest_indices]

            # Calculate confidence scores for selected peaks
            max_strength = np.max(strongest_strengths)
            confidence_scores = (
                strongest_strengths / max_strength
                if max_strength > 0
                else np.ones_like(strongest_strengths)
            )

            # Sort by strength (strongest first) for consistent ordering
            sort_order = np.argsort(strongest_strengths)[::-1]

            return {
                "orientations": [
                    orientation_range[strongest_peaks[i]] for i in sort_order
                ],
                "confidence_scores": confidence_scores[sort_order].tolist(),
            }

    def load_parameters(self, load_exact_matrices=True):
        """
        Load all pre-trained parameters from Echeveste et al. (2020).

        This method loads the 15 optimized parameters from the original
        paper and updates the model's internal state. The parameters can
        be loaded as exact matrices (w_learn, sigma_eta_learn) or
        constructed from the parametric form.

        Parameters
        ----------
        load_exact_matrices : bool, optional
            If True (default), loads exact pre-computed matrices
            (w_learn, sigma_eta_learn).
            If False, loads individual parameters and constructs matrices.

        Returns
        -------
        dict
            Dictionary with all loaded parameters grouped by category

        Raises
        ------
        FileNotFoundError
            If required parameter files are not found

        Examples
        --------
        >>> from skneuromsi.neural import Echeveste2020
        >>> model = Echeveste2020()
        >>> params = model.load_parameters()
        >>> # Model is now ready for simulation
        >>> results = model.simulate_three_phases(
        ...     stimulus_contrast=0.5,
        ...     stimulus_orientation=0.0
        ... )

        Notes
        -----
        After calling this method:
        - Model is marked as trained (_is_trained = True)
        - Both Stage 1 and Stage 2 are marked complete
        - All 15 parameters are loaded and stored
        - Connectivity matrix W is available
        - Noise covariance Sigma_eta is available

        References
        ----------
        .. [1] Echeveste et al. (2020) Supplementary Table S1
        .. [2] scikit-neuromsi/skneuromsi/data/echeveste2020/
        """
        from ..data import EchevesteDataLoader

        loader = EchevesteDataLoader()

        print("Loading pre-trained parameters from Echeveste et al. (2020)...")

        # 1. Load connectivity parameters (8 params)
        print("  [1/4] Loading connectivity parameters...")
        conn_params = loader.load_ssn_connectivity_parameters()
        self._a_EE = conn_params['a_EE']
        self._a_EI = conn_params['a_EI']
        self._a_IE = conn_params['a_IE']
        self._a_II = conn_params['a_II']
        self._d_EE = conn_params['d_EE']
        self._d_EI = conn_params['d_EI']
        self._d_IE = conn_params['d_IE']
        self._d_II = conn_params['d_II']

        # Store for future use
        self._connectivity_params = conn_params

        # 2. Load input transformation parameters (3 params)
        print("  [2/4] Loading input transformation parameters...")
        input_params = loader.load_input_transformation_parameters()
        self._input_scaling = input_params['alpha_h']
        self._input_baseline = input_params['beta_h']
        self._input_nl_pow = input_params['gamma_h']

        # 3. Load noise covariance (4 params + matrix)
        print("  [3/4] Loading noise covariance...")
        try:
            noise_params = loader.load_noise_variance_parameters()
            self._noise_params = noise_params
        except FileNotFoundError:
            print("    Warning: Noise parameters not found, "
                  "will use sigma_eta_learn only")
            self._noise_params = None

        # Load or construct sigma_eta
        self._Sigma_eta = loader.load_noise_covariance(
            fallback_to_construction=True
        )

        # Mark model as trained BEFORE building matrices
        # (build methods check these flags)
        self._stage1_completed = True
        self._stage2_completed = True
        self._is_trained = True

        # 4. Load or construct connectivity matrix W
        print("  [4/4] Loading connectivity matrix...")
        if load_exact_matrices:
            self._W_exact = loader.load_connectivity_matrix()
            # Also build individual blocks for consistency
            self._build_connectivity_matrices()
        else:
            # Build from parameters
            self._build_connectivity_matrices()
            self._W_exact = None

        print("✓ Pre-trained parameters loaded successfully")
        print("  - Connectivity: 8 parameters")
        print("  - Input transformation: 3 parameters")
        noise_msg = (
            '4 parameters + matrix'
            if self._noise_params else 'matrix only'
        )
        print(f"  - Noise covariance: {noise_msg}")
        print("  - Total: 15 parameters")

        return {
            'connectivity': conn_params,
            'input_transformation': input_params,
            'noise': self._noise_params,
            'loaded_exact_matrices': load_exact_matrices,
            'is_trained': self._is_trained,
        }

    def save_parameters(self, output_dir, save_matrices=True):
        """
        Save all trained parameters to files.

        Saves the 15 optimized parameters and optionally the derived
        matrices (W, Sigma_eta) to the specified directory.

        Parameters
        ----------
        output_dir : str or Path
            Directory where to save the parameter files
        save_matrices : bool, optional
            If True (default), also saves the full matrices W and Sigma_eta

        Returns
        -------
        dict
            Dictionary with paths to all saved files

        Raises
        ------
        ValueError
            If model is not trained (no parameters to save)
        OSError
            If output directory cannot be created or files cannot be written

        Examples
        --------
        >>> from skneuromsi.neural import Echeveste2020
        >>> model = Echeveste2020()
        >>> # ... train the model ...
        >>> saved_files = model.save_parameters('my_trained_model/')

        Notes
        -----
        Saved files match the naming convention from Echeveste et al. (2020):
        - input_scaling, input_baseline, input_nl_pow
        - w_ee_height_learn, w_ee_width_learn, etc.
        - var_e_learn, var_i_learn, var_width_learn, rho_learn
        - w_learn (if save_matrices=True)
        - sigma_eta_learn (if save_matrices=True)
        """
        from pathlib import Path
        import numpy as np

        if not self._is_trained:
            raise ValueError(
                "Model is not trained. Train the model first using "
                "model.train(gsm_model) or load pre-trained parameters "
                "using model.load_pretrained_parameters()"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        print(f"Saving parameters to {output_path}...")

        # 1. Save input transformation parameters (3 files)
        print("  [1/4] Saving input transformation parameters...")
        if self._input_scaling is not None:
            np.savetxt(
                output_path / "input_scaling",
                [self._input_scaling]
            )
            saved_files['input_scaling'] = str(output_path / "input_scaling")

        if self._input_baseline is not None:
            np.savetxt(
                output_path / "input_baseline",
                [self._input_baseline]
            )
            saved_files['input_baseline'] = str(
                output_path / "input_baseline"
            )

        if self._input_nl_pow is not None:
            np.savetxt(
                output_path / "input_nl_pow",
                [self._input_nl_pow]
            )
            saved_files['input_nl_pow'] = str(output_path / "input_nl_pow")

        # 2. Save connectivity parameters (8 files)
        print("  [2/4] Saving connectivity parameters...")
        conn_param_mapping = {
            'a_EE': ('w_ee_height_learn', self._a_EE),
            'a_EI': ('w_ei_height_learn', self._a_EI),
            'a_IE': ('w_ie_height_learn', self._a_IE),
            'a_II': ('w_ii_height_learn', self._a_II),
            'd_EE': ('w_ee_width_learn', self._d_EE),
            'd_EI': ('w_ei_width_learn', self._d_EI),
            'd_IE': ('w_ie_width_learn', self._d_IE),
            'd_II': ('w_ii_width_learn', self._d_II),
        }

        for param_key, (filename, value) in conn_param_mapping.items():
            if value is not None:
                np.savetxt(output_path / filename, [value])
                saved_files[filename] = str(output_path / filename)

        # 3. Save noise parameters (4 files)
        print("  [3/4] Saving noise covariance parameters...")
        if self._noise_params is not None:
            np.savetxt(
                output_path / "var_e_learn",
                [self._noise_params['var_e']]
            )
            saved_files['var_e_learn'] = str(output_path / "var_e_learn")

            np.savetxt(
                output_path / "var_i_learn",
                [self._noise_params['var_i']]
            )
            saved_files['var_i_learn'] = str(output_path / "var_i_learn")

            np.savetxt(
                output_path / "var_width_learn",
                [self._noise_params['var_width']]
            )
            saved_files['var_width_learn'] = str(
                output_path / "var_width_learn"
            )

            np.savetxt(
                output_path / "rho_learn",
                [self._noise_params['rho']]
            )
            saved_files['rho_learn'] = str(output_path / "rho_learn")

        # 4. Save derived matrices (optional)
        if save_matrices:
            print("  [4/4] Saving derived matrices...")

            # Save W
            if self._W_exact is not None:
                np.savetxt(output_path / "w_learn", self._W_exact)
                saved_files['w_learn'] = str(output_path / "w_learn")
            elif hasattr(self, '_W') and self._W is not None:
                np.savetxt(output_path / "w_learn", self._W)
                saved_files['w_learn'] = str(output_path / "w_learn")

            # Save Sigma_eta
            if self._Sigma_eta is not None:
                np.savetxt(
                    output_path / "sigma_eta_learn",
                    self._Sigma_eta
                )
                saved_files['sigma_eta_learn'] = str(
                    output_path / "sigma_eta_learn"
                )

        print(f"✓ Saved {len(saved_files)} parameter files")
        return saved_files

    def get_all_parameters(self):
        """
        Get all current parameter values as a dictionary.

        Returns
        -------
        dict
            Dictionary with all 15 parameters grouped by category

        Raises
        ------
        ValueError
            If model is not trained and parameters are not available

        Examples
        --------
        >>> params = model.get_all_parameters()
        >>> print(params['input_transformation'])
        >>> print(params['connectivity'])
        >>> print(params['noise'])
        """
        if not self._is_trained:
            raise ValueError(
                "Model is not trained. No parameters available."
            )

        return {
            'input_transformation': {
                'alpha_h': self._input_scaling,
                'beta_h': self._input_baseline,
                'gamma_h': self._input_nl_pow,
            },
            'connectivity': {
                'a_EE': self._a_EE,
                'a_EI': self._a_EI,
                'a_IE': self._a_IE,
                'a_II': self._a_II,
                'd_EE': self._d_EE,
                'd_EI': self._d_EI,
                'd_IE': self._d_IE,
                'd_II': self._d_II,
            },
            'noise': self._noise_params,
            'is_trained': self._is_trained,
            'stage1_completed': self._stage1_completed,
            'stage2_completed': self._stage2_completed,
        }
