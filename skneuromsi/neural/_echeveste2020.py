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
    """

    #: Time constants for excitatory and inhibitory neurons
    #: t_e: 20ms, t_i: 10ms from Echeveste et al. (2020) Supplementary Material
    tau_e: float
    tau_i: float

    #: Time constant for correlated noise η
    #: tau_n: 20ms by Echeveste original implementation
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
        # sino que se calculan dinámicamente usando
        # conectividad paramétrica (Eq. 10)
        W_EE = W[:n_e, :n_e]  # Conexiones E←E
        W_EI = W[:n_e, n_e:]  # Conexiones E←I
        W_IE = W[n_e:, :n_e]  # Conexiones I←E
        W_II = W[n_e:, n_e:]  # Conexiones I←I

        # Implementa dinámicas SSN (Ecuación 8):
        # τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        # Para neuronas excitatorias (α ∈ E):
        du_e_dt = (
            -u_e  # Potencial de membrana
            + h[:n_e]  # Input externo del modelo GSM
            + W_EE @ r_e  # Input excitatorio: Σ_β∈E W_αβ r_β
            - W_EI @ r_i  # Input inhibitorio: -Σ_β∈I W_αβ r_β
            + eta[:n_e]  # Ruido de inferencia: η_α
        ) / self.tau_e  # Divide por constante de tiempo τ_E

        # Para neuronas inhibitorias (α ∈ I):
        du_i_dt = (
            -u_i + h[n_e:] + W_IE @ r_e - W_II @ r_i + eta[n_e:]
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
        {"target": "excitatory", "template": "excitatory"},
        {"target": "inhibitory", "template": "inhibitory"},
    ]
    _output_mode = "excitatory"  # Primary output mode

    # (Supp. Table S1)
    def __init__(
        self,
        *,
        N_E=50,  # Número de neuronas excitatorias
        N_I=50,  # Número de neuronas inhibitorias
        tau_e=20.0,  # ms - Constante de tiempo excitatorias
        tau_i=10.0,  # ms - Constante de tiempo inhibitorias
        tau_n=20.0,  # ms - Timescale ruido correlacionado η
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
        integrator_model = SSNIntegrator(
            tau_e=tau_e / 1000.0,
            tau_i=tau_i / 1000.0,
            tau_n=tau_n
            / 1000.0,  # Timescale ruido correlacionado η (Echeveste)
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
        Train the SSN model following the two-stage optimization.

        Based on Echeveste et al. (2020).
        The GSM (Gaussian Scale Mixture) generative model must be pre-trained.
        The SSN learns to perform sampling-based inference on this GSM.

        Based on Main paper, página 15: "Sampling-based inference optimization"
        - Stage 1: Optimize connectivity parameters (8 parameters from Eq. 10)
        - Stage 2: Optimize other network parameters (stimuli gain, noise, etc)

        Parameters
        ----------
        gsm_model: object
            Pre-trained GSM generative model containing:
            - Gabor filters (receptive fields A)
            - Prior parameters for z and G
            - Trained parameters from natural image statistics
        stage1_params: dict, optional
            Parameters specific to Stage 1 optimization
        stage2_params: dict, optional
            Parameters specific to Stage 2 optimization

        Returns
        -------
        dict
            Training results with optimized parameters and convergence metrics
        """
        print(
            "method not yet implemented, following patterns of the code"
            "published by echeveste, use load_parameters() instead"
        )
        return None

        # Validate GSM model structure
        if not self._validate_gsm_model(gsm_model):
            raise ValueError("Invalid pre-trained GSM model")

        # Stage 1: Optimize connectivity parameters (Eq. 10)
        # Main paper, página 15: First optimize recurrent connectivity
        stage1_results = self._optimize_stage1(gsm_model, stage1_params)
        self._stage1_completed = True

        # Stage 2: Optimize remaining parameters
        # Main paper, página 16: Then optimize stimulus and noise parameters
        stage2_results = self._optimize_stage2(gsm_model, stage2_params)
        self._stage2_completed = True

        # Build final connectivity matrices from optimized parameters
        self._build_connectivity_matrices()
        self._is_trained = True

        return {
            "stage1": stage1_results,
            "stage2": stage2_results,
            "connectivity_params": self._get_connectivity_parameters(),
            "convergence_info": self._get_convergence_info(),
        }

    # TODO revisar mucha IA
    def _validate_gsm_model(self, gsm_model):
        """
        Validate pre-trained GSM generative model structure.

        Based on Main paper, Eq. 1-7: GSM generative model must be
        pre-trained on natural images before SSN inference optimization.
        The GSM provides the generative model that the SSN learns to invert.
        """
        required_attributes = [
            "gabor_filters",
            "scale_prior_params",
            "gaussian_field_params",
        ]
        return all(hasattr(gsm_model, attr) for attr in required_attributes)

    def _optimize_stage1(self, gsm_model, params):
        """
        Stage 1: Optimize recurrent connectivity kernel using Eq. 10.

        In this first stage, the network is trained
        to approximate the GSM posterior by adjusting
        only the 8 kernel parameters
        {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}.
        These parameters define the shape of the recurrent
        connectivity matrix W via a circular Gaussian function
        of orientation difference:

            W_XY(θi, θj) = a_XY * exp((cos(2(θi - θj)) - 1) / d_XY²)

        Main paper, Eq. 10.

        The optimization minimizes the loss defined in Eq. 25
        (moment-matching between network activity and GSM posterior statistics)
        using stochastic simulations of the dynamics (Eq. 8) with noise.
        Gradients are estimated via backpropagation through time,
        and parameters are updated using the ADAM optimizer.

        During training, the burn-in window Tmin is annealed
        to enforce rapid convergence of the network dynamics.

        Parameters
        ----------
        gsm_model: object
            Pre-trained GSM generative model defining
            the target posterior statistics.
        params: dict
            Stage 1 specific parameters (optimizer settings,
            number of trials, constraints, etc.)

        Returns
        -------
        dict
            Optimization results for Stage 1 (final kernel
            parameters, loss trajectory, etc.)
        """
        # TODO: Implementar optimización de 8 parámetros de conectividad
        # Objetivo: SSN debe generar muestras que coincidan
        # con posterior P(z,G|I) del GSM
        # Usa training data generada por GSM:
        # imagen I -> posterior target P(z,G|I)
        pass

    def _optimize_stage2(self, gsm_model, params):
        """
        Stage 2: Fine-tune recurrent and additional network parameters.

        Based on Eq. 25.
        After Stage 1 has set the coarse recurrent structure
        via parametric kernel (Eq. 10), Stage 2 expands the optimization
        to include additional network parameters such as
        feedforward gains, neuronal time constants,
        and noise covariance terms.

        Main paper, Eq. 25: loss = weighted sum of moment-matching penalties
        (mean, variance, covariance, and slowness terms)
        between network activity and the target GSM posterior statistics.

        The optimization in Stage 2 uses a deterministic
        moment-closure method (Assumed Density Filtering, Eqs. 17–20)
        to compute network moments without sampling noise,
        and employs the L-BFGS-B optimizer for stable convergence.

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
        """
        # TODO: Implementar optimización de parámetros de estímulo y ruido
        # Objetivo: optimizar parámetros no relacionados con
        # conectividad para mejor inferencia
        pass

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
        self._W_EI = self._build_parametric_matrix(
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
        self._W_II = self._build_parametric_matrix(
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
        stimulus_contrast=0.5,  # Contraste del estímulo z en GSM
        stimulus_orientation=0.0,  # Orientación del estímulo G en GSM
        noise_level=1.0,  # Nivel de ruido η para inferencia
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
            # Generacion del estimulo
            # Main paper Eq. 1-7: I = z * G
            # donde z es contraste, G es campo orientado
            stimulus = self._generate_gsm_stimulus(
                stimulus_contrast,
                stimulus_orientation,  # Parámetros de GSM model
            )

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
            # Construir matriz de conectividad
            # Main paper Eq. 10:
            # W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²]
            # MEJORA vs Echeveste: Guardamos matrices
            # W_EE, W_EI, W_IE, W_II por separado
            # Ventaja: Memoria eficiente, acceso rápido
            # por bloques, debug más fácil
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
        # Empieza desde estado de reposo para todas las neuronas
        u_0 = np.concatenate(
            [
                np.zeros(self._N_E),  # Potenciales excitatorios iniciales
                np.zeros(self._N_I),  # Potenciales inhibitorios iniciales
            ]
        )

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
        )  # τ_η^(-1), con τ_η = 20ms (Table S1)

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

            # Verificación de estabilidad numérica (como en código original)
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
            u_trajectory_used = u_trajectory[:actual_steps]
            if actual_steps == 0:
                raise RuntimeError(
                    f"Simulation failed immediately. Check network "
                    f"parameters: contrast={stimulus_contrast}, "
                    f"noise_level={noise_level}. Try reducing "
                    f"noise_level or stimulus_contrast."
                )
        else:
            u_trajectory_used = u_trajectory

        # EXTRACCIÓN DE ACTIVIDAD NEURONAL
        # Calcular actividad r_α(t) = k * [u_α(t)]_+^n (Main paper Eq. 9)
        excitatory_activity = self._integrator.f.supralinear_activation(
            u_trajectory_used[:, :self._N_E]  # Solo neuronas excitatorias
        )
        inhibitory_activity = self._integrator.f.supralinear_activation(
            u_trajectory_used[:, self._N_E:]  # Solo neuronas inhibitorias
        )

        response = {
            # Actividad excitatorias (Main paper Fig. 4)
            "excitatory": excitatory_activity,
            # Actividad inhibitorias (Main paper Fig. 4)
            "inhibitory": inhibitory_activity,
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

        # Handle explosion cases by removing problematic last data point
        if isinstance(extra, dict) and not extra.get(
            "simulation_successful", True
        ):
            # For exploded simulations, always remove the last data point
            # This fixes the systematic off-by-one issue
            if hasattr(modes_dict, "values") and modes_dict:
                print("DEBUG: Explosion detected - removing last point")
                truncated_modes_dict = {}
                for key, data in modes_dict.items():
                    if hasattr(data, "__len__") and len(data) > 0:
                        truncated_modes_dict[key] = data[:-1]  # Remove last
                    else:
                        truncated_modes_dict[key] = data
                modes_dict = truncated_modes_dict

                # Fix floating point precision issue in time_range calculation
                # The validation uses: expected = int(time_span/time_res)
                # To ensure int(time_span / time_res) == new_length,
                # we add a small epsilon to compensate
                # for floating point errors
                new_length = len(next(iter(modes_dict.values())))
                epsilon = 1e-10
                new_time_max = new_length * time_res + epsilon
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
            W_full[: self._N_E, : self._N_E] = self._W_EE
            W_full[: self._N_E, self._N_E:] = self._W_EI
            W_full[self._N_E:, : self._N_E] = self._W_IE
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
            delta_theta = theta_pre[:, None] - theta_post[None, :]
            return sign * a * np.exp((np.cos(2 * delta_theta) - 1) / d**2)

        # Bloques matriciales (all positive, signs applied in dynamics)
        W[0:self._N_E, 0:self._N_E] = connectivity_block(
            theta_e,
            theta_e,
            params["a_EE"],
            params["d_EE"],
            sign=1,
        )
        W[0: self._N_E, self._N_E: self._N] = connectivity_block(
            theta_e,
            theta_i,
            params["a_EI"],
            params["d_EI"],
            sign=1,
        )
        W[self._N_E: self._N, 0: self._N_E] = connectivity_block(
            theta_i,
            theta_e,
            params["a_IE"],
            params["d_IE"],
            sign=1,
        )
        W[self._N_E: self._N, self._N_E: self._N] = connectivity_block(
            theta_i,
            theta_i,
            params["a_II"],
            params["d_II"],
            sign=1,
        )

        return W

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

            # Note: orientation parameter reserved for future
            # orientation-specific stimulus generation.
            # Currently GSM generates orientation-distributed
            # stimuli based on contrast level only.
            _ = orientation  # Suppress unused parameter warning

            # Generate stimulus using GSM
            # Maps contrast (0-1) to Echeveste's contrast levels (0-4)
            contrast_level = contrast * 4.0

            # Generate single stimulus sample
            h_samples = self._gsm.get_h_for_contrast_level(
                contrast_level, n_samples=1
            )

            # Extract stimulus vector
            h_stimulus = h_samples[0]

            # Validate stimulus dimensions match network size
            expected_size = self._N_E + self._N_I

            if len(h_stimulus) != expected_size:
                # If GSM generates orientation-only stimulus (N_E dimensions),
                # replicate for both E and I populations
                if len(h_stimulus) == self._N_E:
                    h_full = np.concatenate([h_stimulus, h_stimulus])
                    if len(h_full) != expected_size:
                        raise ValueError(
                            f"Stimulus dimension mismatch after replication. "
                            f"Expected {expected_size}, got {len(h_full)}. "
                            (
                                f"Network: {self._N_E}E + {self._N_I}I = "
                                f"{expected_size} total."
                            )
                        )
                    return h_full
                else:
                    raise ValueError(
                        f"Stimulus dimension mismatch. "
                        f"Expected {expected_size} "
                        f"or {self._N_E} (for replication), "
                        f"got {len(h_stimulus)}. "
                        f"Network configuration: {self._N_E}E + "
                        f"{self._N_I}I = {expected_size} total. "
                        f"Please ensure GSM is configured for the "
                        f"correct network size."
                    )

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
        # altura mínima de un pico para ser considerado
        peak_threshold = kwargs.get("peak_threshold", 0.1)
        # distancia mínima entre picos
        peak_distance = kwargs.get("peak_distance", 10)
        # confianza mínima comparado con la probabilidad máxima
        confidence_threshold = kwargs.get("confidence_threshold", 0.5)

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
            if "excitatory" in kwargs and "inhibitory" in kwargs:
                # Extract final timestep from simulation results
                exc_activity = kwargs["excitatory"]
                inh_activity = kwargs["inhibitory"]

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
        cause_orientations = self._extract_cause_orientations(
            network_activity, causal_results["cause_contrasts"]
        )

        # Return full results dictionary
        return {
            "num_causes": causal_results["num_causes"],
            "cause_positions": cause_orientations,
            "cause_contrasts": causal_results["cause_contrasts"],
            "confidence": causal_results["confidence_scores"],
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
        Extract posterior distribution P(z|x) from network activity.

        This method interprets network activity as samples from the posterior
        distribution over contrast values, similar to the original Echeveste
        implementation.

        Parameters
        ----------
        network_activity : array_like
            Network activity pattern (N,)
        contrast_range : array_like
            Range of contrast values to evaluate

        Returns
        -------
        posterior_dist : dict
            Dictionary with posterior distribution:
            - 'contrast_values': array of contrast values
            - 'probabilities': array of posterior probabilities
            - 'map_estimate': MAP (maximum a posteriori) estimate
        """
        # Extraer actividad excitatoria (neuronas sintonizadas a orientación)
        # Fundamento: paper principal, Sec. 2.1 - solo las neuronas E
        # representan variables latentes del modelo GSM
        excitatory_activity = network_activity[:self._N_E]

        # Inicializar distribución de contraste P(z|x)
        contrast_distribution = np.zeros(len(contrast_range))

        # Crear mapeo de actividad de red a distribución de contraste
        # Basado en conceptos de Echeveste et al. (2020):
        # La actividad total debe reflejar el contraste del estímulo
        for i, contrast in enumerate(contrast_range):
            # Calcular actividad total esperada para este contraste
            # Usa relación supralinear del SSN (paper principal, Eq. 8)
            total_activity = np.sum(excitatory_activity)

            # La actividad debe alcanzar un pico alrededor
            # de ciertos valores de contraste
            # Relación Gaussiana basada en el modelo GSM original
            # Factor de escala 0.1 derivado de
            # parámetros de Echeveste (Tabla S1)
            activity_expected = contrast * self._N_E * 0.1
            activity_diff = abs(total_activity - activity_expected)

            # Convertir diferencia a probabilidad usando kernel Gaussiano
            # Fundamento: Bayesian inference con likelihood Gaussiano
            # Varianza normalizada por número de neuronas (0.5 * N_E)
            contrast_distribution[i] = np.exp(
                -activity_diff / (0.5 * self._N_E)
            )

        # Normalizar para crear distribución de probabilidad apropiada
        # Condición necesaria para inferencia Bayesiana válida
        if np.sum(contrast_distribution) > 0:
            contrast_distribution = contrast_distribution / np.sum(
                contrast_distribution
            )
        else:
            # Distribución uniforme como fallback (prior no informativo)
            contrast_distribution = np.ones(len(contrast_range)) / len(
                contrast_range
            )

        # Encontrar estimador MAP (Maximum A Posteriori)
        # Corresponde al pico principal de la distribución posterior
        map_idx = np.argmax(contrast_distribution)
        map_estimate = contrast_range[map_idx]

        return {
            "contrast_values": contrast_range,
            "probabilities": contrast_distribution,
            "map_estimate": map_estimate,
        }

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
        peaks, properties = find_peaks(
            probabilities,
            # umbral mínimo de altura para considerar un pico
            height=peak_threshold,
            # distancia mínima entre picos para evitar ruido
            distance=peak_distance,
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
        # probable es esa causa comparada con la más probable
        # Confianza = P(causa_i) / max(P(todas_las_causas))
        max_prob = np.max(probabilities)
        confidence_scores = (
            peak_heights / max_prob
            if max_prob > 0
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
        Extract orientation angles of causes from network activity patterns.

        Parameters
        ----------
        network_activity : array_like
            Network activity (N,)
        cause_contrasts : list
            Contrast values of detected causes

        Returns
        -------
        cause_orientations : list
            Orientation angles in degrees for each cause
        """
        # Extraer actividad excitatoria (sintonizada a orientación)
        # Fundamento teórico: Echeveste et al. Sec. 2.1 - solo las neuronas
        # excitatorias representan las variables latentes del modelo GSM
        # Las neuronas E están organizadas según su orientación preferida
        excitatory_activity = network_activity[:self._N_E]

        # Calcular orientaciones preferidas para neuronas excitatorias
        # Fundamento: topología de anillo con neuronas organizadas
        # Echeveste et al. Fig. 1c - "ring topology with neurons arranged
        # according to their preferred orientation"
        orientation_range = np.linspace(
            self._position_range[0], self._position_range[1], self._N_E
        )

        cause_orientations = []

        # Para cada causa detectada, extraer su orientación dominante
        # Fundamento: Echeveste et al. Fig. 5 - cada causa tiene una orient
        # específica determinada por el patrón de actividad de la red
        for contrast in cause_contrasts:
            # Encontrar neuronas con mayor actividad (orientación detectada)
            # Ponderar actividad por intensidad del contraste de la causa
            # Fundamento teórico: la actividad ponderada refleja la
            # contribución relativa de cada orientación a la causa inferida
            weighted_activity = excitatory_activity * contrast

            # Encontrar pico en actividad (orientación dominante)
            # El pico indica la orientación más probable para esta causa
            # según la actividad actual de la red SSN
            peak_neuron = np.argmax(weighted_activity)
            peak_orientation = orientation_range[peak_neuron]

            cause_orientations.append(peak_orientation)

        return cause_orientations
