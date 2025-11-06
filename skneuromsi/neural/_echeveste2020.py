#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

import os
from dataclasses import dataclass

import brainpy as bp

import numpy as np
from scipy import stats

from ..core import SKNMSIMethodABC
from ..data import EchevesteDataLoader


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
        """
        return self.k * np.power(np.maximum(0, u), self.n)

    # TODELETE
    def parametric_connectivity(self, theta_i, theta_j, a_xy, d_xy):
        """
        Calculate parametric connectivity using the formula in Equation 10.

        Implement: W_XY(θi, θj) = a_XY * exp[(cos(2(θi - θj)) - 1) / d_XY²]
        where:
        - θi, θj: preferred orientations of neurons i, j (in radians)
        - a_XY: connectivity amplitude between X→Y populations
        - d_XY: connectivity width (dispersion parameter)

        Mathematical foundation:
        - Main paper, Eq. 10: Parametric connectivity with angular differences
        - Only 8 parameters: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
        - Circular topology: angular differences θi - θj determine
            connection strength

        """
        angular_diff = theta_i - theta_j  # Diferencia θi - θj

        exp_term = np.exp((np.cos(2 * angular_diff) - 1) / (d_xy**2))

        # Conectividad final: amplitud × perfil espacial
        W_xy = a_xy * exp_term

        return W_xy

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
        r = np.concatenate([r_e, r_i])  # r_β del término Σ_β W_αβ r_β

        # Aplicar ecuación exacta como en código original: W @ r
        W_times_r = W @ r  # Σ_β W_αβ r_β (incluye signos correctos)

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
        """
        # Estructura básica de la red
        self._N_E = N_E  # Número de neuronas excitatorias
        self._N_I = N_I  # Número de neuronas inhibitorias

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
            - max_iter: number of ADAM iterations (default: 250)
            - n_trials: number of samples per iteration (default: 50)
            - eta: ADAM learning rate (default: 0.002)
            - beta1, beta2: ADAM momentum parameters
            - t_min_initial, t_min_final: annealing range
        stage2_params: dict, optional
            Parameters specific to Stage 2 (L-BFGS-B) optimization:
            - max_iter: number of L-BFGS-B iterations (default: 50)
            - min_time: fixed evaluation window start (default: T_max - 50ms)

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
        self._is_trained = True

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
            def connectivity_block(theta_pre, theta_post, a, d, sign):
                delta = theta_pre[:, None] - theta_post[None, :]
                return sign * a * jnp.exp(
                    (jnp.cos(2 * delta) - 1) / (d**2)
                )

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

            # Kernel espacial
            spatial_kernel = jnp.exp(
                (jnp.cos(delta) - 1) / (noise_width**2)
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

        # Función de costo con sampling estocástico
        def compute_cost_and_grad(x, iteration, key):
            """
            Compute cost and gradient using stochastic sampling.

            Paper: "using N_trial = 50 trials for each training stimulus
            to estimate the corresponding moments"
            """
            # Temporal annealing: T_min increases linearly over iterations
            # Paper: "T_min was systematically changed ('annealed') from
            # T_min = 0 ms to T_max - 50 ms"
            # train.ml lines 240-243
            progress = iteration / max_iter
            current_t_min = (
                t_min_initial + progress * (t_min_final - t_min_initial)
            )

            # Construir matriz W desde parámetros
            w = build_w_from_params(x)

            # Calcular costo con N_trial samples
            # Paper: "Both the network's initial conditions and the process
            # noise were re-sampled for each trial and iteration"
            cost, _, _ = compute_costs_with_samples(
                w, h_vec, sigma_eta, inv_taus, dt,
                self._integrator.f.tau_n, t_max, t_subsamp,
                target_mu, target_sigma, self._integrator.f.k,
                lambda_mean, lambda_var, lambda_cov, current_t_min,
                n_trials, key
            )

            return cost

        # Función para calcular gradiente usando JAX autodiff
        grad_fn = jax.grad(compute_cost_and_grad, argnums=0)

        # Inicializar parámetros
        x = x0.copy()

        # Inicializar momentos de ADAM
        # ADAM mantiene promedios móviles exponenciales de gradientes
        # (m) y gradientes al cuadrado (v)
        m = jnp.zeros_like(x)  # First moment (mean of gradients)
        v = jnp.zeros_like(x)  # Second moment (uncentered variance)

        # Gradient clipping
        # train.ml line 292: clip = sqrt(n_prms) * 5
        grad_clip = np.sqrt(len(x)) * 5.0

        # Tracking
        cost_history = []
        best_cost = float('inf')
        best_x = x.copy()

        # Random key para reproducibilidad
        key = jax.random.PRNGKey(params.get('seed', 42))

        # ADAM optimization loop
        # Paper: "performed 250 iterations of the ADAM optimizer"
        for iteration in range(1, max_iter + 1):
            # Generar nuevo key para este iteration
            # Paper: "re-sampled for each trial and iteration"
            key, subkey = jax.random.split(key)

            # Calcular costo y gradiente
            cost = compute_cost_and_grad(x, iteration, subkey)
            grad = grad_fn(x, iteration, subkey)

            # Convertir a numpy para manipulación
            grad = np.array(grad)
            cost_val = float(cost)

            # Gradient clipping para estabilidad
            # train.ml line 261
            grad_norm = np.linalg.norm(grad)
            if grad_norm > grad_clip:
                grad = grad * (grad_clip / grad_norm)

            # ADAM update
            # Paper referencias: Kingma & Ba (2015)
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            # Bias correction
            m_hat = m / (1 - beta1 ** iteration)
            v_hat = v / (1 - beta2 ** iteration)

            # Parameter update
            x = x - eta * m_hat / (jnp.sqrt(v_hat) + epsilon_adam)

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
                      f"||grad||: {grad_norm:.3f}")

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
        Stage 2: Optimize using L-BFGS-B with deterministic ADF method.

        Based on Echeveste et al. (2020) Methods section, page 18:
        "In the second stage, we continued optimization using the L-BFGS-B
        optimizer, now using the ADF method to (deterministically) compute
        the moments of the network's response distribution. We kept the
        cost-integration time window at its minimum (T_max - T_min = 50 ms,
        as reached by the end of the first phase)."

        Key differences from Stage 1:
        1. L-BFGS-B optimizer (quasi-Newton method)
        2. Deterministic ADF method (no sampling, N_trial = None)
        3. Fixed time window: T_min = T_max - 50ms
        4. Can include slowness penalty cost (lambda_slow)

        This stage optimizes network noise covariance parameters while
        keeping connectivity parameters fixed from Stage 1.

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
        """
        from scipy.optimize import minimize
        from ._echeveste_training import compute_evolution_costs
        import jax.numpy as jnp

        # Parámetros por defecto de Stage 2
        # Basado en objective.ml líneas 174-179 (sigma_eta_prms)
        if params is None:
            params = {}

        # Parámetros de optimización
        max_iter = params.get('max_iter', 50)
        dt = params.get('dt', 0.2e-3)
        t_max = params.get('t_max', 0.1)
        t_subsamp = params.get('t_subsamp', 10.0e-3)
        min_time = params.get('min_time', 0.05)  # 50ms evaluation window
        lambda_mean = params.get('lambda_mean', 1.0)
        lambda_var = params.get('lambda_var', 1.0)
        lambda_cov = params.get('lambda_cov', 1.0)

        # Inicializar parámetros de ruido
        # objective.ml lines 174-179
        initial_noise_params = {
            'width': params.get('noise_width', 0.8),
            'std_e': params.get('noise_std_e', 2.0),
            'std_i': params.get('noise_std_i', 2.0),
            'rho': params.get('noise_rho', 0.8),
        }

        print("Stage 2: Optimizing noise covariance parameters...")
        print(f"Initial noise params: {initial_noise_params}")
        print(f"Max iterations: {max_iter}")

        # Extraer targets del GSM (reutilizar del Stage 1 si está disponible)
        contrast = params.get('contrast', 0.5)
        n_samples_gsm = params.get('n_samples_gsm', 50)

        print("Computing GSM posterior targets for Stage 2...")
        gsm_data = gsm_model.compute_posterior_for_ssn_training(
            contrast=contrast,
            n_samples=n_samples_gsm
        )

        # Ajustar h_vec al tamaño de la red (mismo que Stage 1)
        # Duplicar input para neuronas E e I (ver generalization.py:116)
        h_full = gsm_data['h_inputs'].mean(axis=0)
        h_e = h_full[:self._N_E]  # Primeras N_E para excitatorias
        h_i = h_e[:self._N_I]     # Duplicar para inhibitorias (mismo input)
        h_vec = jnp.concatenate([h_e, h_i])  # Shape: (N_E + N_I,)

        # Ajustar targets al tamaño de la red
        target_mu = jnp.array(gsm_data['target_mu'][:self._N_E])
        target_sigma = jnp.array(
            gsm_data['target_sigma'][:self._N_E, :self._N_E]
        )

        # Construir inverse time constants
        inv_taus = jnp.concatenate([
            jnp.full(self._N_E, 1.0 / self._integrator.f.tau_e),
            jnp.full(self._N_I, 1.0 / self._integrator.f.tau_i)
        ])

        # Matriz W ya optimizada en Stage 1
        # Si Stage 1 no se ejecutó, usar parámetros default
        if not self._stage1_completed:
            print("Warning: Stage 1 not completed, using default connectivity")
            default_conn_params = {
                'a_EE': 0.02, 'a_EI': 0.02,
                'a_IE': 0.02, 'a_II': 0.02,
                'd_EE': 0.8, 'd_EI': 0.8,
                'd_IE': 0.8, 'd_II': 0.8,
            }
            W_fixed = self.build_connectivity_matrix(default_conn_params)
        else:
            W_fixed = self.build_connectivity_matrix()

        def pack_noise_parameters(noise_params):
            """
            Pack noise parameters into optimization vector.

            Based on objective.ml lines 205-211.
            Parámetros de ruido: width, std_e, std_i, rho
            """
            # Transformaciones para garantizar positividad y bounds
            x = np.zeros(4)
            x[0] = noise_params['width']  # width (directo)
            x[1] = noise_params['std_e']  # std_e (directo)
            x[2] = noise_params['std_i']  # std_i (directo)
            # rho ∈ (0,1): usar transformación tanh inversa
            # rho = 0.5 * (1 + tanh(x)) => x = atanh(2*rho - 1)
            rho = noise_params['rho']
            if rho <= 0.01:
                rho = 0.01
            if rho >= 0.99:
                rho = 0.99
            z = 2.0 * rho - 1.0
            x[3] = 0.5 * (np.log(1.0 + z) - np.log(1.0 - z))
            return x

        def unpack_noise_parameters(x):
            """Unpack optimization vector to noise parameters."""
            noise_params = {}
            noise_params['width'] = x[0]
            noise_params['std_e'] = x[1]
            noise_params['std_i'] = x[2]
            # rho = 0.5 * (1 + tanh(x[3]))
            # IMPORTANTE: Usar jnp.tanh para que sea diferenciable por JAX
            noise_params['rho'] = 0.5 * (1.0 + jnp.tanh(x[3]))
            return noise_params

        def build_sigma_eta_jax(width, std_e, std_i, rho):
            """Construir Σ_η con JAX."""
            var_e = std_e ** 2
            var_i = std_i ** 2

            theta = jnp.linspace(0, jnp.pi, self._N_E, endpoint=False)
            delta = theta[:, None] - theta[None, :]

            # Kernel espacial
            spatial_kernel = jnp.exp(
                (jnp.cos(delta) - 1) / (width**2)
            )

            # Construir bloques
            Sigma_ee = var_e * spatial_kernel
            Sigma_ii = var_i * spatial_kernel
            Sigma_ei = rho * jnp.sqrt(var_e * var_i) * spatial_kernel

            # Ensamblar matriz completa
            Sigma_eta = jnp.block([
                [Sigma_ee, Sigma_ei],
                [Sigma_ei.T, Sigma_ii]
            ])

            # Agregar término diagonal para estabilidad
            Sigma_eta = Sigma_eta + 0.01 * jnp.eye(self._N)

            return Sigma_eta

        # Pack parámetros iniciales
        x0 = pack_noise_parameters(initial_noise_params)

        # Bounds para Stage 2 (objective.ml lines 234-237)
        bounds = [
            (0.01, np.sqrt(2.0)),  # width ∈ (0, √2)
            (0.1, 4.0),            # std_e ∈ (0.1, 4)
            (0.1, 4.0),            # std_i ∈ (0.1, 4)
            (-5.0, 5.0),           # rho_transformed ∈ (-∞, ∞)
        ]

        # Función objetivo con gradientes JAX
        # Mismo patrón que Stage 1: gradientes analíticos son más precisos
        import jax

        def objective_jax(x):
            """Calcular costo (versión JAX pura para autodiff)."""
            noise_params = unpack_noise_parameters(x)

            # Construir Σ_η con parámetros actuales
            sigma_eta = build_sigma_eta_jax(
                noise_params['width'],
                noise_params['std_e'],
                noise_params['std_i'],
                noise_params['rho']
            )

            # Calcular costo usando ADF
            cost, _, _ = compute_evolution_costs(
                jnp.array(W_fixed), h_vec, sigma_eta, inv_taus,
                dt, self._integrator.f.tau_n, t_max, t_subsamp,
                target_mu, target_sigma, self._integrator.f.k,
                lambda_mean, lambda_var, lambda_cov, min_time
            )

            # Manejar NaN/Inf
            cost = jnp.where(
                jnp.isfinite(cost),
                cost,
                1e10
            )

            return cost  # Mantener como JAX array para autodiff

        # Calcular gradiente usando JAX
        grad_fn = jax.grad(objective_jax)

        def objective(x):
            """Wrapper para scipy (retorna float)."""
            return float(objective_jax(jnp.array(x)))

        def gradient(x):
            """Gradiente analítico usando JAX."""
            g = grad_fn(jnp.array(x))
            return np.array(g)

        # Optimización usando L-BFGS-B
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            jac=gradient,  # Usar gradientes analíticos de JAX
            bounds=bounds,
            options={'maxiter': max_iter, 'disp': True}
        )

        # Desempaquetar parámetros optimizados
        optimized_noise_params = unpack_noise_parameters(result.x)

        # Solo actualizar matriz Σ_η si fue exitoso
        if result.success:
            self._Sigma_eta = self._build_noise_covariance(
                optimized_noise_params['width'],
                optimized_noise_params['std_e'],
                optimized_noise_params['std_i'],
                optimized_noise_params['rho']
            )

        # NOTA: NO marcar _stage2_completed aquí, se marca en train()
        # solo si result.success == True

        print("\nStage 2 optimization finished:")
        print(f"  Success: {result.success}")
        print(f"  Final cost: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}/{max_iter}")
        print(f"  Message: {result.message}")
        print(f"  width: {initial_noise_params['width']:.4f} → "
              f"{optimized_noise_params['width']:.4f}")
        print(f"  std_e: {initial_noise_params['std_e']:.4f} → "
              f"{optimized_noise_params['std_e']:.4f}")
        print(f"  std_i: {initial_noise_params['std_i']:.4f} → "
              f"{optimized_noise_params['std_i']:.4f}")
        print(f"  rho:   {initial_noise_params['rho']:.4f} → "
              f"{optimized_noise_params['rho']:.4f}")

        return {
            'optimized_params': optimized_noise_params,
            'initial_params': initial_noise_params,
            'final_cost': float(result.fun),
            'n_iterations': int(result.nit),
            'max_iterations': max_iter,
            'sigma_eta_shape': (
                self._Sigma_eta.shape if result.success else None
            ),
            'success': bool(result.success),
            'message': str(result.message),
            'nfev': int(result.nfev),
            'njev': int(result.njev) if hasattr(result, 'njev') else None
        }

    def _build_noise_covariance(self, width, std_e, std_i, rho):
        """
        Build noise covariance matrix.

        Based on objective.ml lines 290-305 (sigma_eta function).
        Construye la matriz de covarianza del ruido η siguiendo
        la estructura de Kronecker con kernel espacial exponencial.

        Parameters
        ----------
        width : float
            Width parameter for spatial correlation
        std_e : float
            Standard deviation for excitatory noise
        std_i : float
            Standard deviation for inhibitory noise
        rho : float
            Cross-correlation between E and I populations

        Returns
        -------
        np.ndarray
            Noise covariance matrix of shape (2*N_E, 2*N_E)
        """
        n_total = self._N_E * 2  # Total de neuronas E+I
        Sigma = np.zeros((n_total, n_total))

        # Varianzas
        var_e = std_e ** 2
        var_i = std_i ** 2

        # Orientaciones
        orientations = np.linspace(0, np.pi, self._N_E, endpoint=False)

        for i in range(n_total):
            for j in range(n_total):
                # Determinar población (E o I)
                is_i_exc = i < self._N_E
                is_j_exc = j < self._N_E

                # Varianza correspondiente
                if is_i_exc and is_j_exc:
                    var_ij = var_e
                elif not is_i_exc and not is_j_exc:
                    var_ij = var_i
                else:
                    var_ij = rho * np.sqrt(var_e * var_i)

                # Índices en el ring
                idx_i = i % self._N_E
                idx_j = j % self._N_E

                # Diferencia de orientación
                theta_i = orientations[idx_i]
                theta_j = orientations[idx_j]
                cos_diff_minus_one = np.cos(theta_i - theta_j) - 1.0

                # Kernel espacial exponencial
                # objective.ml line 303
                spatial_kernel = np.exp(
                    cos_diff_minus_one / (width ** 2)
                )

                Sigma[i, j] = var_ij * spatial_kernel

        # Agregar pequeño valor a diagonal para estabilidad
        # objective.ml line 305
        Sigma += 0.01 * np.eye(n_total)

        return Sigma

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

        Parameters
        ----------
        theta_pre: np.ndarray
            Orientations of pre-synaptic neurons
        theta_post: np.ndarray
            Orientations of post-synaptic neurons
        a_xy: float
            Amplitude parameter
        d_xy: float
            Width parameter

        Returns
        -------
        np.ndarray
            Connectivity matrix computed from parametric formula
        """
        matrix = np.zeros((len(theta_post), len(theta_pre)))
        for i, theta_i in enumerate(theta_post):
            for j, theta_j in enumerate(theta_pre):
                matrix[i, j] = self.parametric_connectivity(
                    theta_i, theta_j, a_xy, d_xy
                )
        return matrix

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
        Implementa el cálculo de momentos no lineales necesarios
        para la evolución de la red bajo activación supralineal.

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

        # nu1 = μ·Ψ + σ·φ
        nu1 = mu * psi + sigma_std * phi

        # Hard-coded n=2 (supralinear exponent)
        # From objective.ml line 267: nu_fun
        # nu = k * (μ·nu1 + σ²·Ψ)
        nu = self._k * (mu * nu1 + sigma2 * psi)

        # From objective.ml line 268: gamma_fun
        # gamma = 2k * nu1
        gamma = 2 * self._k * nu1

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

    def save_parameters(self, output_path):
        """
        Save optimized parameters following Echeveste's approach.

        Saves both the 8 parametric values and computed full matrices,
        matching the structure found in ssn_inference_numerical_experiments.

        Parameters
        ----------
        output_path: str
            Directory path to save parameters
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before saving parameters")

        # Save 8 parametric connectivity parameters (scalar files)
        params = self._get_connectivity_parameters()
        for param_name, param_value in params.items():
            np.savetxt(
                f"{output_path}/w_{param_name.lower()}_learn", [param_value]
            )

        # Save full connectivity matrices
        full_W = self.build_connectivity_matrix()
        np.savetxt(f"{output_path}/w_learn", full_W)

        # Save noise covariance matrix
        if self._Sigma_eta is not None:
            np.savetxt(f"{output_path}/sigma_eta_learn", self._Sigma_eta)

    def load_parameters(self, input_path=None):
        """
        Load pre-trained parameters from files or internal data loader.

        Can load from:
        1. Internal EchevesteDataLoader (if input_path is None)
        2. External directory path (for custom parameters)
        3. Full matrix fallback

        Sets training stages based on what parameters are successfully loaded:
        - Stage 1: Connectivity parameters (8 parametric values)
        - Stage 2: Noise covariance matrix (Sigma_eta)
        """
        # Reset training state
        self._stage1_completed = False
        self._stage2_completed = False
        self._is_trained = False

        # STAGE 1: Load connectivity parameters
        try:
            if input_path is None:
                # Use internal data loader
                loader = EchevesteDataLoader()
                params = loader.load_ssn_connectivity_parameters()
                # Set parametric values from internal loader
                self._a_EE = params["a_EE"]
                self._a_EI = params["a_EI"]
                self._a_IE = params["a_IE"]
                self._a_II = params["a_II"]
                self._d_EE = params["d_EE"]
                self._d_EI = params["d_EI"]
                self._d_IE = params["d_IE"]
                self._d_II = params["d_II"]
                print("Stage 1 parameters loaded from internal data loader")

                # CRITICAL: Also load exact matrix to avoid instability
                try:
                    self._W_exact = loader.load_exact_connectivity_matrix()
                    print(
                        f"Also loaded exact matrix: "
                        f"shape {self._W_exact.shape}"
                    )
                except FileNotFoundError:
                    print("Warning: No exact matrix found in internal data")
            else:
                # Load from external directory (original behavior)
                param_mapping = {
                    "a_ee": "w_ee_height_learn",
                    "a_ei": "w_ei_height_learn",
                    "a_ie": "w_ie_height_learn",
                    "a_ii": "w_ii_height_learn",
                    "d_ee": "w_ee_width_learn",
                    "d_ei": "w_ei_width_learn",
                    "d_ie": "w_ie_width_learn",
                    "d_ii": "w_ii_width_learn",
                }

                params = {}
                for param_key, file_name in param_mapping.items():
                    params[param_key] = np.loadtxt(
                        os.path.join(input_path, file_name)
                    )

                # Set parametric values
                self._a_EE = params["a_ee"]
                self._a_EI = params["a_ei"]
                self._a_IE = params["a_ie"]
                self._a_II = params["a_ii"]
                self._d_EE = params["d_ee"]
                self._d_EI = params["d_ei"]
                self._d_IE = params["d_ie"]
                self._d_II = params["d_ii"]
                print("Stage 1 parameters loaded successfully (connectivity)")

                # Also try to load exact matrix if available
                try:
                    w_exact = np.loadtxt(os.path.join(input_path, "w_learn"))
                    self._W_exact = w_exact
                    print(
                        f"Also loaded exact matrix w_learn: "
                        f"shape {w_exact.shape}"
                    )
                except FileNotFoundError:
                    print("No w_learn file found - using computed matrix")

            # Mark stage 1 as completed first, then build matrices
            self._stage1_completed = True
            # Build matrices from parameters
            self._build_connectivity_matrices()

        except FileNotFoundError as e:
            if input_path is not None:
                print(f"Warning: Could not load Stage 1 parameters: {e}")
                # Try fallback: load full matrix if parametric files not found
                try:
                    w_full = np.loadtxt(os.path.join(input_path, "w_learn"))
                    print("Loaded full connectivity matrix as fallback")
                    print(f"Full W matrix shape: {w_full.shape}")
                    # Store the full matrix for use in simulations
                    self._W_full = w_full
                    # Store exact original matrix for run() method
                    self._W_exact = w_full
                    # Mark as partially trained
                    self._stage1_completed = True
                except FileNotFoundError:
                    print(
                        """Error: No connectivity parameters found (neither
                        parametric nor full matrix)"""
                    )
            else:
                print(f"Error loading from internal data loader: {e}")

        # STAGE 2: Load noise parameters
        try:
            if input_path is None:
                # Use internal data loader
                loader = EchevesteDataLoader()
                loaded_sigma = loader.load_noise_covariance()
                # Validate dimensions match current network size
                if loaded_sigma.shape[0] == self._N:
                    self._Sigma_eta = loaded_sigma
                    print(
                        "Stage 2 parameters loaded from internal data loader"
                    )
                else:
                    print(
                        f"Warning: Loaded Sigma_eta shape "
                        f"{loaded_sigma.shape} doesn't match "
                        f"network size {self._N}"
                    )
                    print("Using default noise covariance instead")
                    self._Sigma_eta = None
            else:
                # Load from external directory
                loaded_sigma = np.loadtxt(
                    os.path.join(input_path, "sigma_eta_learn")
                )
                # Validate dimensions match current network size
                if loaded_sigma.shape[0] == self._N:
                    self._Sigma_eta = loaded_sigma
                    print(
                        "Stage 2 parameters loaded successfully "
                        "(noise covariance)"
                    )
                else:
                    print(
                        f"Warning: External Sigma_eta shape "
                        f"{loaded_sigma.shape} doesn't match "
                        f"network size {self._N}"
                    )
                    print("Using default noise covariance instead")
                    self._Sigma_eta = None

            self._stage2_completed = True

        except FileNotFoundError:
            print(
                """Warning: noise covariance not found,
                will use default noise in simulations"""
            )
            self._Sigma_eta = None

        # UPDATE TRAINING STATUS
        # Model is considered fully trained if both stages completed
        if self._stage1_completed and self._stage2_completed:
            self._is_trained = True
            print("Model fully trained - both stages completed")
        elif self._stage1_completed:
            print(
                """Model partially trained -
                only Stage 1 (connectivity) completed"""
            )
        elif self._stage2_completed:
            print(
                """Model partially trained -
                only Stage 2 (noise) completed"""
            )
        else:
            print("Model not trained - no parameters loaded successfully")

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

    def run(
        self,
        *,
        stimulus_contrast=0.019,  # Contraste del estímulo z en GSM
        stimulus_orientation=0.0,  # Orientación del estímulo G en GSM
        noise_level=0.1,  # Nivel de ruido η para inferencia
        simulation_time=1000.0,  # Tiempo de simulación en ms
        **kwargs,  # Parámetros adicionales
    ):
        """Run the SSN simulation."""
        # Verificar que el modelo esté entrenado antes de la simulación
        if not self.is_trained():
            raise ValueError(
                """Model must be trained before simulation.
                Use train() or load_parameters() method first."""
            )

        try:
            # Generate GSM stimulus following Echeveste et al. (2020)
            # Mathematical foundation:
            # - Main paper, Eq. 1-7: I = z * G where z is contrast,
            # G is oriented field
            # - Main paper, Section 2.1: "Gaussian Scale Mixture (GSM) model"
            # - Supplementary Material:
            # "Visual input through GSM generative model"
            #
            # Justification for re-enabling GSM:
            # 1. All numerical instabilities have been fixed:
            #    - Time constants corrected (tau_e=20ms, tau_i=10ms)
            #    - Exact connectivity matrix loaded (w_learn)
            #    - Proper supralinear activation function
            # 2. GSM is the authentic stimulus model from the paper
            # 3. Required for proper causal inference testing

            stimulus = self._generate_gsm_stimulus(
                stimulus_contrast,
                stimulus_orientation,  # GSM model parameters
            )

            # Fallback option for debugging (can be enabled if needed)
            # stimulus = self._generate_simple_stimulus(stimulus_contrast)

            # Validate stimulus dimensions
            expected_stimulus_size = self._N_E + self._N_I
            if len(stimulus) != expected_stimulus_size:
                raise ValueError(
                    f"Stimulus dimension mismatch. Expected "
                    f"{expected_stimulus_size}, got {len(stimulus)}. "
                    f"Network: {self._N_E}E + {self._N_I}I = "
                    f"{expected_stimulus_size} total."
                )

        except Exception as e:
            if (
                "dimension mismatch" in str(e).lower()
                or "stimulus" in str(e).lower()
            ):
                raise  # Re-raise stimulus generation errors with clear message
            else:
                raise RuntimeError(
                    f"Failed to generate stimulus for SSN simulation: {e}. "
                    f"Network configuration: {self._N_E}E + "
                    f"{self._N_I}I = {self._N_E + self._N_I} total."
                ) from e

        try:
            # FIX: Use exact same connectivity matrix as original
            # The original code uses pre-computed w_learn matrix
            # Our build_connectivity_matrix() differs by up to 0.339
            if hasattr(self, "_W_exact") and self._W_exact is not None:
                # Use the exact original matrix if loaded
                W = self._W_exact
            else:
                # Fallback to computed matrix (may cause instability)
                W = self.build_connectivity_matrix()

            # Validate connectivity matrix dimensions
            expected_W_shape = (self._N_E + self._N_I, self._N_E + self._N_I)
            if W.shape != expected_W_shape:
                raise ValueError(
                    f"Connectivity matrix dimension mismatch. Expected "
                    f"{expected_W_shape}, got {W.shape}. "
                    f"Network: {self._N_E}E + {self._N_I}I = "
                    f"{self._N_E + self._N_I} total."
                )

        except Exception as e:
            if (
                "dimension mismatch" in str(e).lower()
                or "connectivity" in str(e).lower()
            ):
                raise  # Re-raise connectivity matrix errors with clear message
            else:
                raise RuntimeError(
                    f"Failed to build connectivity matrix: {e}. "
                    f"Network configuration: {self._N_E}E + "
                    f"{self._N_I}I = {self._N_E + self._N_I} total. "
                    f"Ensure model parameters are properly loaded."
                ) from e

        # Main paper Eq. 8: Variables de estado u_α(t=0)
        # Los estados iniciales de las neuronas u(0) se toman de una
        # distribución normal multivariada con μ₀=0 y Σ₀=4I

        # Crear matriz de covarianza Σ₀ = 4I para estados iniciales
        total_neurons = self._N_E + self._N_I
        mu_0 = np.zeros(total_neurons)  # Media μ₀ = 0
        Sigma_0 = 4.0 * np.eye(total_neurons)  # Covarianza Σ₀ = 4I

        # Generar estados iniciales desde distribución normal multivariada
        u_0 = np.random.multivariate_normal(mu_0, Sigma_0)

        # Main paper Eq. 11: ⟨η(t)η(t+s)ᵀ⟩ = Σᶯexp(-s/τᶯ)
        # donde τᶯ = 20ms (Table S1)
        # Main paper p.15: "Σᶯ was the stationary (zero-lag)
        # covariance matrix"
        if self._Sigma_eta is not None:
            # Usa matriz de covarianza Σᶯ entrenada
            # (referida en Supp. Table S1)
            Sigma_eta = self._Sigma_eta * noise_level
        else:
            # Fallback: Supp. p.7 "Σᶯ = σ²I" (aproximación simplificada)
            Sigma_eta = noise_level * np.eye(self._N)

        # NOTA: El código original de Echeveste carga directamente
        # Sigma_eta pre-calculada desde "sigma_eta_learn"
        # que fue optimizada durante Stage 2 del entrenamiento
        # Ver methods.py:388 y full_net.py:51 del código original

        # La matriz Sigma_eta ya está calculada arriba usando
        # la implementación disponible
        # (cargada desde parámetros o construida como identidad escalada)

        # Descomposición Cholesky para generar ruido correlacionado
        # (código original methods.py:398)
        # Permite generar η~N(0,Sigma_eta) a partir de
        # z~N(0,I) mediante η = L @ z
        L = np.linalg.cholesky(Sigma_eta)

        # correlación temporal del ruido η(t)
        # Main paper Eq. 11: ⟨η(t) η(t + s)^T⟩ = Σ_η exp(−s/τ_η)
        # Implementación temporal siguiendo código original methods.py:402-403

        dt = self._time_res / 1000.0  # Convierte ms a segundos
        tau_n_inv = (
            1.0 / self._integrator.f.tau_n
        )  # τ_η^(-1), con τ_η = 0.02s (20ms, Table S1)

        # Coeficientes para actualización temporal
        # (código original methods.py:402-403)
        eps_1 = 1.0 - dt * tau_n_inv  # Aproximación e^(-dt/τ_η) ≈ 1-dt/τ_η
        eps_2 = np.sqrt(
            2.0 * dt * tau_n_inv
        )  # Factor para mantener varianza estacionaria

        # CONDICIÓN INICIAL: η(t=0)~N(0,Σᶯ) según Main paper Eq. 11
        eta_0 = self._random.multivariate_normal(np.zeros(self._N), Sigma_eta)

        # Implementa loop temporal como en network_evolution()
        # del código original, Main paper
        # Eq. 8: τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α

        # Tiempo de simulación para alcanzar steady-state (Main paper Fig. 3)
        simulation_steps = int(simulation_time / self._time_res)

        # Inicialización de variables de estado
        u_old = np.copy(u_0)  # u(t): potenciales membrana en tiempo t
        eta_old = np.copy(eta_0)  # η(t): ruido correlacionado en tiempo t

        # Storage para trayectorias temporales
        u_trajectory = np.zeros((simulation_steps, self._N))

        # Loop temporal principal (siguiendo network_evolution de Echeveste)
        simulation_successful = True
        actual_steps = simulation_steps

        for step in range(simulation_steps):
            # Calcula actividad neuronal usando
            # función supralineal (Main paper, Eq. 9)
            # r_α = k * [u_α]_+^n donde k=0.3, n=2.0, [x]_+ = max(0,x)
            r_e = self._integrator.f.supralinear_activation(
                u_old[:self._N_E]
            )  # Excitatorias
            r_i = self._integrator.f.supralinear_activation(
                u_old[self._N_E:]
            )  # Inhibitorias
            _ = np.concatenate([r_e, r_i])  # noqa: F841

            # Genera ruido blanco Gaussiano independiente para cada neurona
            # Siguiendo methods.py:413 del código original de Echeveste
            temp = L @ self._random.normal(loc=0.0, scale=1.0, size=self._N)

            # Actualiza ruido correlacionado (proceso Ornstein-Uhlenbeck)
            # Siguiendo methods.py:417 del código original
            # η(t+dt) = eps_1 * η(t) + eps_2 * temp
            eta_new = eps_1 * eta_old + eps_2 * temp

            # Actualiza potenciales usando integrador
            # BrainPy (Main paper, Eq. 8)
            # MEJORA: Usamos integrador BrainPy en vez de implementación manual
            # du_α/dt = (-u_α + Σ_β W_αβ r_β + h_α + η_α) / τ_α
            u_e_old, u_i_old = u_old[:self._N_E], u_old[self._N_E:]
            u_e_new, u_i_new = self._integrator(
                u_e_old, u_i_old, step * dt, W, stimulus, eta_old
            )
            u_new = np.concatenate([u_e_new, u_i_new])

            # Guarda estado actual
            u_trajectory[step] = u_new

            # Actualiza variables para siguiente paso temporal
            u_old = np.copy(u_new)
            eta_old = np.copy(eta_new)

            # Verificación de estabilidad numérica
            # (código original: SSN/methods.py:428, 441, 465, 488, 519, 542)
            if np.linalg.norm(u_new) > 1000:
                simulation_successful = False
                actual_steps = step  # Explosion occurred at this step
                print(
                    f"Warning: Activity exploded at step "
                    f"{step}/{simulation_steps}. "
                    f"Using trajectory up to step {step-1}."
                )
                break

        # Handle simulation results based on success/failure
        if not simulation_successful:
            # Truncate trajectory to actual simulated steps
            u_trajectory = u_trajectory[:actual_steps]
            if actual_steps == 0:
                raise RuntimeError(
                    f"Simulation failed immediately. Check network "
                    f"parameters: contrast={stimulus_contrast}, "
                    f"noise_level={noise_level}. Try reducing "
                    f"noise_level or stimulus_contrast."
                )

        # EXTRACCIÓN DE ACTIVIDAD NEURONAL Y POTENCIALES DE MEMBRANA
        # Calcular actividad r_α(t) = k * [u_α(t)]_+^n (Main paper Eq. 9)
        excitatory_activity = self._integrator.f.supralinear_activation(
            u_trajectory[:, :self._N_E]  # Solo neuronas excitatorias
        )
        inhibitory_activity = self._integrator.f.supralinear_activation(
            u_trajectory[:, self._N_E:]  # Solo neuronas inhibitorias
        )

        # Extraer potenciales de membrana u_α(t) (Main paper Eq. 8)
        excitatory_potential = u_trajectory[:, :self._N_E]
        inhibitory_potential = u_trajectory[:, self._N_E:]

        response = {
            # Firing rates r_α(t) (Main paper Eq. 9: r = k * [u]_+^n)
            "excitatory_firing_rate": excitatory_activity,
            "inhibitory_firing_rate": inhibitory_activity,
            # Potenciales de membrana u_α(t) (Main paper Eq. 8)
            "excitatory_potential": excitatory_potential,
            "inhibitory_potential": inhibitory_potential,
        }

        # Note: temporal dimension adjustment for truncated simulations
        # is handled by the custom _make_result method

        extra = {
            # Variable z del modelo GSM (Eq. 2)
            "stimulus_contrast": stimulus_contrast,
            # Orientación G del modelo GSM (Eq. 3)
            "stimulus_orientation": stimulus_orientation,
            # Nivel ruido η (Supp. Sec. 2.3)
            "noise_level": noise_level,
            # Simulation status information
            "simulation_successful": simulation_successful,
            "actual_simulation_steps": actual_steps,
            "requested_simulation_steps": simulation_steps,
            "actual_simulation_time": actual_steps * self._time_res,
            "requested_simulation_time": simulation_time,
        }

        return (
            response,
            extra,
        )  # Retorna actividad y parámetros del experimento

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

    def parametric_connectivity(self, theta_i, theta_j, a_xy, d_xy):
        """
        Calculate parametric connectivity using the formula in Equation 10.

        Implement: W_XY(θi, θj) = a_XY * exp[(cos(2(θi - θj)) - 1) / d_XY²]
        where:
        - θi, θj: preferred orientations of neurons i, j (in radians)
        - a_XY: connectivity amplitude between X→Y populations
        - d_XY: connectivity width (dispersion parameter)

        Mathematical foundation:
        - Main paper, Eq. 10: Parametric connectivity with angular differences
        - Only 8 parameters: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
        - Circular topology: angular differences θi - θj
        determine connection strength
        """
        angular_diff = theta_i - theta_j  # Diferencia θi - θj

        exp_term = np.exp((np.cos(2 * angular_diff) - 1) / (d_xy**2))

        # Conectividad final: amplitud × perfil espacial
        W_xy = a_xy * exp_term

        return W_xy

    def build_connectivity_matrix(self, connectivity_params=None):
        """
        Construct connectivity matrix using parametric formulation (Eq.10).

        Mathematical foundation:
        - Main paper, Eq. 10: W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²]
        - Only 8 parameters: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
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

        # Genera orientaciones preferidas para ring topology
        # (Main paper Fig. 1B)
        # theta[i] = np.pi * i / N (en el caso de 180 grados)
        theta_e = np.linspace(
            np.radians(self._position_range[0]),
            np.radians(self._position_range[1]),
            self._N_E,
        )

        theta_i = np.linspace(
            np.radians(self._position_range[0]),
            np.radians(self._position_range[1]),
            self._N_I,
        )

        # Inicializar matriz W completa (N x N)
        W = np.zeros((self._N, self._N))

        # Construir bloques de conectividad usando parametric_connectivity
        def connectivity_block(theta_pre, theta_post, a, d, sign=1):
            delta_theta = theta_post[:, None] - theta_pre[None, :]
            return sign * a * np.exp((np.cos(2 * delta_theta) - 1) / d**2)

        # Bloques matriciales con signos correctos según código original
        # Original: E→E (+), E→I (-), I→E (+), I→I (-)
        W[0:self._N_E, 0:self._N_E] = connectivity_block(
            theta_e,
            theta_e,
            params["a_EE"],
            params["d_EE"],
            sign=1,
        )
        W[0:self._N_E, self._N_E:self._N] = connectivity_block(
            theta_e,
            theta_i,
            params["a_EI"],
            params["d_EI"],
            sign=-1,
        )
        W[self._N_E:self._N, 0:self._N_E] = connectivity_block(
            theta_i,
            theta_e,
            params["a_IE"],
            params["d_IE"],
            sign=1,
        )
        W[self._N_E:self._N, self._N_E:self._N] = connectivity_block(
            theta_i,
            theta_i,
            params["a_II"],
            params["d_II"],
            sign=-1,
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

    def _generate_gsm_stimulus(self, contrast, orientation=0.0):
        """
        Generate stimulus from Gaussian Scale Mixture (GSM) generative model.

        Mathematical foundation:
        - Main paper, Eq. 1-3: GSM generative model I = z * G where:
          * I: observed image patch
          * z: scale variable (controls contrast)
          * G: Gaussian random field (oriented structure)
        - Main paper, Eq. 4-5: Prior distributions P(z) and P(G)
        - Main paper, Eq. 6-7: Linear projection to neural inputs h = A^T * I
        - Main paper, Figure 1A: Gabor-like receptive fields from
            GSM optimization
        - Echeveste et al. (2020) Supplementary Material: Detailed GSM
          mathematics

        Implementation uses integrated GSM model from generative module.
        """
        try:
            # Initialize GSM if not already created
            if not hasattr(self, "_gsm"):
                self._gsm = self._create_gsm()

            # Generate oriented stimulus using GSM
            # Maps contrast (0-1) to Echeveste's contrast levels (0-4)
            contrast_level = contrast * 4.0

            # Create oriented stimulus centered at specified orientation
            # Convert orientation from degrees to neuron index
            orientation_rad = np.radians(orientation)
            neuron_orientations = np.linspace(0, np.pi, self._N_E)

            # Find the neuron closest to desired orientation
            orientation_diffs = np.abs(neuron_orientations - orientation_rad)
            center_neuron = np.argmin(orientation_diffs)

            # Create Gaussian bump centered at desired orientation
            sigma = 0.15 * self._N_E  # Width parameter (like original GSM)
            h_stimulus = np.zeros(self._N_E)

            for i in range(self._N_E):
                # Circular distance for ring topology
                dist1 = abs(i - center_neuron)
                dist2 = self._N_E - abs(i - center_neuron)
                diff = min(dist1, dist2)
                h_stimulus[i] = (
                    contrast_level * np.exp(-0.5 * (diff / sigma) ** 2)
                )

            # Ensure minimum baseline activity
            h_stimulus += 0.01 * contrast_level

            # Extend to full network size (E + I neurons)
            expected_size = self._N_E + self._N_I
            if len(h_stimulus) == self._N_E:
                # Replicate for I neurons
                h_full = np.concatenate([h_stimulus, h_stimulus])

                if len(h_full) != expected_size:
                    raise ValueError(
                        f"Stimulus dimension mismatch after replication. "
                        f"Expected {expected_size}, got {len(h_full)}. "
                        f"Network: {self._N_E}E + {self._N_I}I = "
                        f"{expected_size} total."
                    )
                return h_full
            else:
                # h_stimulus already has correct size
                return h_stimulus

        except Exception as e:
            if "dimension mismatch" in str(e).lower():
                raise  # Re-raise our custom dimension errors
            else:
                raise RuntimeError(
                    f"Failed to generate GSM stimulus: {e}. "
                    f"Network size: {self._N_E}E + {self._N_I}I = "
                    f"{self._N_E + self._N_I} total. "
                    f"Check that the GSM module is properly configured."
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
        x_reconstructed = np.dot(A, excitatory_activity)  # (256,)

        # Add realistic noise level based on GSM parameters
        if np.std(x_reconstructed) > 0:
            noise_scale = 0.1 * np.std(x_reconstructed)
        else:
            noise_scale = 0.1
        noise = np.random.normal(0, noise_scale, len(x_reconstructed))
        x_observation = x_reconstructed + noise

        # EXACT ECHEVESTE FORMULA - GSM.py line 210-227
        # Compute P(z|x) for each contrast value in contrast_range

        n_contrasts = len(contrast_range)
        D_x = len(x_observation)
        log_p = np.zeros(n_contrasts)

        # Pre-compute matrices for efficiency
        ACA_T = np.dot(A, np.dot(C, A.T))  # A*C*A^T
        mean_x = np.zeros(D_x)  # Mean is always zero in GSM

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
            covariance = z * z * ACA_T + s_x_2 * np.eye(D_x)

            try:
                # Log prior: P(z) ~ Gamma(k, theta)
                if z > 0:
                    from scipy.stats import gamma

                    log_prior = gamma.logpdf(z, k_gamma, scale=theta_gamma)
                else:
                    log_prior = -np.inf  # Zero prior for negative contrasts

                # Log likelihood: P(x|z) ~ N(0, Cov)
                from scipy.stats import multivariate_normal

                log_likelihood = multivariate_normal.logpdf(
                    x_observation, mean_x, covariance
                )

                # Log posterior = log prior + log likelihood
                log_p[i] = log_prior + log_likelihood

            except (np.linalg.LinAlgError, ValueError):
                # Handle numerical issues with singular covariance
                log_p[i] = -np.inf

        # Normalize probabilities (exactly as in original code)
        max_log_p = np.max(log_p)
        p_unnorm = np.exp(log_p - max_log_p)
        norm = np.sum(p_unnorm) * dz

        if norm > 0:
            probabilities = p_unnorm / norm
        else:
            # Fallback: uniform distribution
            probabilities = np.ones(n_contrasts) / n_contrasts

        # Find MAP estimate
        map_idx = np.argmax(probabilities)
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
