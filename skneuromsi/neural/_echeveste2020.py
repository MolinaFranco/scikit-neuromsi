#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""
Implementation of the Echeveste et al. (2020) model for cortical-like dynamics
in recurrent circuits optimized for sampling-based probabilistic inference.

This module implements the Stabilized Supralinear Network (SSN) optimized for
sampling-based probabilistic inference under the Gaussian Scale Mixture (GSM)
generative model as described in Echeveste et al. (2020).
"""

import copy
from dataclasses import dataclass
import os

import brainpy as bp

import numpy as np

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
    """

    #: Time constants for excitatory and inhibitory neurons
    #: t_e: 20ms, t_i: 10ms by Supplementary information
    tau_e: float
    tau_i: float

    #: Supralinear exponent for Excitatory neurons, n=2.0 by Supplementary information
    n: float

    #: Scaling factor for firing rates, k=0.3 by Supplementary information
    k: float

    #: Name of the integrator
    name: str = "SSNIntegrator"

    @property
    def __name__(self):
        """Return the name of the Integrator."""
        return self.name

    def supralinear_activation(self, u):
        """
        Supralinear activation function: r = k * [u]_+^n where [x]_+ = max(0, x)

        Mathematical foundation:
        - Echeveste et al. (2020), Equations 8-9: The firing rate of neuron α is
          r_α = k * [u_α]_+^n, where u_α is the membrane potential
        - Supplementary Material, Section 2.1: The supralinear exponent n = 2.0
          produces the required nonlinearity for sampling-based inference
          and k = 0.3 (scaling factor from SSN parameters, Supplementary Table S1)
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
        - Circular topology: angular differences θi - θj determine connection strength

        """

        angular_diff = theta_i - theta_j  # Diferencia θi - θj

        exp_term = np.exp((np.cos(2 * angular_diff) - 1) / (d_xy**2))

        # Conectividad final: amplitud × perfil espacial
        W_xy = a_xy * exp_term

        return W_xy

    def __call__(self, u_e, u_i, t, W, h, eta):
        """
        Compute the SSN dynamics based on Echeveste et al. (2020) Equations 8-9.

        Mathematical foundation:
        - Main paper, Eq. 8: τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        - Main paper, Eq. 9: r_α = k * [u_α]_+^n (supralinear activation)
        - Supplementary Material, Table S1: τ_E = 20ms, τ_I = 10ms
        - Main paper, Section 2.2: E-I network structure with W_EE, W_EI, W_IE, W_II blocks
        - Main paper, Eq. 6-7: GSM generative model provides h (external input)
        - Supplementary Material, Section 2.3: η represents inference noise

        INPUTS:
        - u_e: Excitatory membrane potentials (shape: [N_E]) - variable u_α for α ∈ E
        - u_i: Inhibitory membrane potentials (shape: [N_I]) - variable u_α for α ∈ I
        - t: Current time (scalar)
        - W: Connectivity matrix (shape: [N, N]) - W_αβ from Eq. 8
        - h: External inputs (shape: [N]) - h_α from GSM model (Eq. 6-7)
        - eta: Noise inputs (shape: [N]) - η_α for sampling-based inference

        RETURN:
        - (du_e_dt, du_i_dt): Temporal derivatives from Equation 8
        """

        # Calcula firing rates usando activación supralineal (Eq. 9)
        r_e = self.supralinear_activation(u_e)  # Tasas excitatorias
        r_i = self.supralinear_activation(u_i)  # Tasas inhibitorias

        # Número de neuronas excitatorias (Supp. Table S1: N_E=50)
        n_e = len(u_e)

        # IMPORTANTE: En entrenamiento los pesos W no son matrices almacenadas
        # sino que se calculan dinámicamente usando conectividad paramétrica (Eq. 10)
        W_EE = W[:n_e, :n_e]  # Conexiones E←E
        W_EI = W[:n_e, n_e:]  # Conexiones E←I
        W_IE = W[n_e:, :n_e]  # Conexiones I←E
        W_II = W[n_e:, n_e:]  # Conexiones I←I

        # Implementa dinámicas SSN (Ecuación 8): τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
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

        # IMPLEMENTACIÓN ALTERNATIVA usando conectividad paramétrica (Eq. 10):
        # sera usada en un futuro en la etapa de entrenamiento
        # # Orientaciones preferidas de neuronas (topología circular)
        # # theta_e[i] = np.pi * i / len(u_e)
        # theta_e = np.linspace(0, np.pi, len(u_e))  # Orientaciones excitatorias
        # theta_i = np.linspace(0, np.pi, len(u_i))  # Orientaciones inhibitorias
        #
        # du_e_dt_parametric = np.zeros_like(u_e)
        # for i in range(len(u_e)):  # Para cada neurona excitatoria i
        #     # Input excitatorio: Σ_j W_EE(θi,θj) * r_e[j]
        #     excitatory_input = np.sum([
        #         self.parametric_connectivity(theta_e[i], theta_e[j], a_EE, d_EE) * r_e[j]
        #         for j in range(len(u_e))
        #     ])
        #     # Input inhibitorio: -Σ_j W_EI(θi,θj) * r_i[j] (Dale: W_EI > 0)
        #     inhibitory_input = -np.sum([
        #         self.parametric_connectivity(theta_e[i], theta_i[j], a_EI, d_EI) * r_i[j]
        #         for j in range(len(u_i))
        #     ])
        #     # Dinámicas completas: τ_e * du_e[i]/dt = -u_e[i] + inputs + h[i] + η[i]
        #     du_e_dt_parametric[i] = (
        #         -u_e[i] + excitatory_input + inhibitory_input + h[i] + eta[i]
        #     ) / self.tau_e
        #
        # # FALTA VERSIÓN ALTERNATIVA para du_i_dt usando Eq. 10

        return du_e_dt, du_i_dt  # Retorna derivadas temporales de Eq. 8


class Echeveste2020(SKNMSIMethodABC):
    """
    Cortical-like dynamics model from Echeveste et al. (2020).

    Implementation of "Cortical-like dynamics in recurrent circuits optimized
    for sampling-based probabilistic inference" - Echeveste et al., Nature Neuroscience 2020

    Mathematical foundation:
    - Main paper, Abstract: "optimized recurrent cortical circuits that implement
      sampling-based probabilistic inference"
    - Main paper, Eq. 1-7: Gaussian Scale Mixture (GSM) generative model
    - Main paper, Eq. 8-9: Stabilized Supralinear Network (SSN) dynamics
    - Main paper, Figure 1: Ring topology with circular symmetry for orientation selectivity
    - Supplementary Material, Table S1: Network parameters (N_E=50, N_I=50, etc.)

    Key features:
    - Ring topology with E-I populations (Main paper, Section 2.2)
    - Supralinear activation r = k[u]_+^n (Eq. 9, n=2.0)
    - Parametric connectivity: W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²] (Eq. 10)
    - Solo 8 parámetros optimizados: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
    - GSM generative model for natural image patches (Eq. 1-7)
    - Sampling-based Bayesian inference (Main paper, Section 2.1)
    - Cortical-like gamma oscillations (Main paper, Figure 3)
    """

    # Implementación del modelo de dinámicas corticales de Echeveste et al. (2020)

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
        n=2.0,  # Exponente supralineal de Eq. 9
        k=0.3,  # Factor de escala de Eq. 9
        seed=None,  # Semilla para generador aleatorio
        position_range=(
            0,
            180,
        ),  # Orientaciones ring topology (Main paper, Figure 1B)
        position_res=360 / 100,  # Resolución angular para simetría circular
        time_range=(0, 1000),  # Rango temporal de simulación (ms)
        time_res=0.2,  # Paso temporal dt
        **integrator_kws,  # Keywords adicionales para integrador
    ):
        """
        Initialize the SSN model with parameters from Echeveste et al. (2020).

        Mathematical foundation:
        - Supplementary Material, Table S1: Complete parameter specification
        - Main paper, Section 2.2: Network architecture with ring topology
        - Main paper, Eq. 8-9: SSN dynamics with specified time constants
        - Supplementary Material, Section 2.1: Numerical integration details

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
        self._position_res = position_res  # Resolución angular (3.6°)
        self._time_range = time_range  # Duración simulación (1000ms)
        self._time_res = time_res  # dt = 0.2ms

        # Crea integrador numérico para dinámicas SSN (Supp. Sec. 2.1)
        # Método Euler con dt = 0.2ms asegura estabilidad para dinámicas inhibitorias rápidas
        integrator_kws.setdefault(
            "method", "euler"
        )  # Integración Euler forward
        integrator_kws.setdefault(
            "dt", time_res / 1000.0
        )  # Convierte ms a segundos para BrainPy

        # Inicializa integrador SSN con parámetros optimizados (Supp. Table S1)
        integrator_model = SSNIntegrator(
            tau_e=tau_e / 1000.0,
            tau_i=tau_i / 1000.0,
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
        Train the SSN model following the two-stage optimization from Echeveste et al. (2020).

        The GSM (Gaussian Scale Mixture) generative model must be pre-trained.
        The SSN learns to perform sampling-based inference on this GSM.

        Based on Main paper, página 15: "Sampling-based inference optimization"
        - Stage 1: Optimize connectivity parameters (8 parameters from Eq. 10)
        - Stage 2: Optimize other network parameters (stimuli gain, noise, etc.)

        Parameters
        ----------
        gsm_model : object
            Pre-trained GSM generative model containing:
            - Gabor filters (receptive fields A)
            - Prior parameters for z and G
            - Trained parameters from natural image statistics
        stage1_params : dict, optional
            Parameters specific to Stage 1 optimization
        stage2_params : dict, optional
            Parameters specific to Stage 2 optimization

        Returns
        -------
        dict
            Training results with optimized parameters and convergence metrics
        """

        print(
            "method not yet implemented, following patterns of the code published by echeveste, use load_parameters() instead"
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

        In this first stage, the network is trained to approximate the GSM posterior
        by adjusting only the 8 kernel parameters {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}.
        These parameters define the shape of the recurrent connectivity matrix W via
        a circular Gaussian function of orientation difference:

            W_XY(θi, θj) = a_XY * exp((cos(2(θi - θj)) - 1) / d_XY²)

        Main paper, Eq. 10.

        The optimization minimizes the loss defined in Eq. 25 (moment-matching between
        network activity and GSM posterior statistics) using stochastic simulations of
        the dynamics (Eq. 8) with noise. Gradients are estimated via backpropagation
        through time, and parameters are updated using the ADAM optimizer.

        During training, the burn-in window Tmin is annealed to enforce rapid convergence
        of the network dynamics.

        Parameters
        ----------
        gsm_model : object
            Pre-trained GSM generative model defining the target posterior statistics.
        params : dict
            Stage 1 specific parameters (optimizer settings, number of trials, constraints, etc.)

        Returns
        -------
        dict
            Optimization results for Stage 1 (final kernel parameters, loss trajectory, etc.)
        """
        # TODO: Implementar optimización de 8 parámetros de conectividad
        # Objetivo: SSN debe generar muestras que coincidan con posterior P(z,G|I) del GSM
        # Usa training data generada por GSM: imagen I -> posterior target P(z,G|I)
        pass

    def _optimize_stage2(self, gsm_model, params):
        """
        Stage 2: Fine-tune recurrent and additional network parameters using Eq. 25.

        After Stage 1 has set the coarse recurrent structure via parametric kernel (Eq. 10),
        Stage 2 expands the optimization to include additional network parameters such as
        feedforward gains, neuronal time constants, and noise covariance terms.

        Main paper, Eq. 25: loss = weighted sum of moment-matching penalties
        (mean, variance, covariance, and slowness terms) between network activity
        and the target GSM posterior statistics.

        The optimization in Stage 2 uses a deterministic moment-closure method
        (Assumed Density Filtering, Eqs. 17–20) to compute network moments without
        sampling noise, and employs the L-BFGS-B optimizer for stable convergence.

        Parameters
        ----------
        gsm_model : object
            Pre-trained GSM generative model
        params : dict
            Stage 2 specific parameters

        Returns
        -------
        dict
            Optimization results for Stage 2
        """
        # TODO: Implementar optimización de parámetros de estímulo y ruido
        # Objetivo: optimizar parámetros no relacionados con conectividad para mejor inferencia
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
        # Alternativa de cálculo: orientations = np.pi * np.arange(self._N) / self._N

        # Build matrices using parametric connectivity
        self._W_EE = self._build_parametric_matrix(
            orientations[: self._N_E],
            orientations[: self._N_E],
            self._a_EE,
            self._d_EE,
        )
        self._W_EI = self._build_parametric_matrix(
            orientations[: self._N_E],
            orientations[self._N_E :],
            self._a_EI,
            self._d_EI,
        )
        self._W_IE = self._build_parametric_matrix(
            orientations[self._N_E :],
            orientations[: self._N_E],
            self._a_IE,
            self._d_IE,
        )
        self._W_II = self._build_parametric_matrix(
            orientations[self._N_E :],
            orientations[self._N_E :],
            self._a_II,
            self._d_II,
        )

    def _build_parametric_matrix(self, theta_pre, theta_post, a_xy, d_xy):
        """
        Build connectivity matrix using parametric formula from Eq. 10.

        Parameters
        ----------
        theta_pre : np.ndarray
            Orientations of pre-synaptic neurons
        theta_post : np.ndarray
            Orientations of post-synaptic neurons
        a_xy : float
            Amplitude parameter
        d_xy : float
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
        output_path : str
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

    def load_parameters(self, input_path):
        """
        Load pre-trained parameters from files.

        Can load either from parametric files or full matrix,
        following Echeveste's dual storage approach.

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
            # Try loading parametric parameters first
            params = {}
            for param in [
                "a_ee",
                "a_ei",
                "a_ie",
                "a_ii",
                "d_ee",
                "d_ei",
                "d_ie",
                "d_ii",
            ]:
                params[param] = np.loadtxt(
                    os.path.join(input_path, f"w_{param}_learn")
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

            # Build matrices from parameters
            self._build_connectivity_matrices()
            self._stage1_completed = True
            print("Stage 1 parameters loaded successfully (connectivity)")

        except FileNotFoundError as e:
            print(f"Warning: Could not load Stage 1 parameters: {e}")
            # Try fallback: load full matrix if parametric files not found
            try:
                full_W = np.loadtxt(os.path.join(input_path, "w_learn"))
                print("Loaded full connectivity matrix as fallback")
                # TODO: Extract parameters from full matrix if needed
            except FileNotFoundError:
                print(
                    "Error: No connectivity parameters found (neither parametric nor full matrix)"
                )

        # STAGE 2: Load noise parameters
        try:
            self._Sigma_eta = np.loadtxt(
                os.path.join(input_path, "sigma_eta_learn")
            )
            self._stage2_completed = True
            print("Stage 2 parameters loaded successfully (noise covariance)")
        except FileNotFoundError:
            print(
                "Warning: sigma_eta_learn not found, will use default noise in simulations"
            )
            self._Sigma_eta = None

        # UPDATE TRAINING STATUS
        # Model is considered fully trained if both stages completed
        if self._stage1_completed and self._stage2_completed:
            self._is_trained = True
            print("Model fully trained - both stages completed")
        elif self._stage1_completed:
            print(
                "Model partially trained - only Stage 1 (connectivity) completed"
            )
        elif self._stage2_completed:
            print("Model partially trained - only Stage 2 (noise) completed")
        else:
            print("Model not trained - no parameters loaded successfully")

    def is_trained(self):
        """Check if model has been trained with both stages completed."""
        return (
            self._is_trained
            and self._stage1_completed
            and self._stage2_completed
        )

    # PROPERTY ================================================================

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
        stimulus_contrast=0.5,  # Contraste del estímulo z en GSM (Eq. 1-3)
        stimulus_orientation=0.0,  # Orientación del estímulo G en GSM (Eq. 1-3)
        noise_level=1.0,  # Nivel de ruido η para inferencia (Supp. Sec. 2.3)
        simulation_time=500.0,  # Tiempo de simulación en ms
        **kwargs,  # Parámetros adicionales
    ):
        """
        Run the SSN simulation.
        """
        # Verificar que el modelo esté entrenado antes de la simulación
        if not self.is_trained():
            raise ValueError(
                "Model must be trained before simulation. Use train() or load_parameters() method first."
            )

        # Generacion del estimulo
        # Main paper Eq. 1-7: I = z * G donde z es contraste, G es campo orientado
        stimulus = self._generate_gsm_stimulus(
            stimulus_contrast, stimulus_orientation  # Parámetros de GSM model
        )

        # Construir matriz de conectividad
        # Main paper Eq. 10: W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²]
        # Usa parámetros entrenados (8 parámetros) para construir matriz completa
        W = self.build_connectivity_matrix()

        # Main paper Eq. 8: Variables de estado u_α(t=0)
        # Empieza desde estado de reposo para todas las neuronas
        u_0 = np.concatenate(
            [
                np.zeros(self._N_E),  # Potenciales excitatorios iniciales
                np.zeros(self._N_I),  # Potenciales inhibitorios iniciales
            ]
        )

        # Parametro de ruido
        # Supp. Material: Ruido η correlacionado temporal y espacialmente
        # Usa matriz de covarianza entrenada si está disponible, sino usa default escalado
        if self._Sigma_eta is not None:
            # Usa matriz de covarianza entrenada (optimizada en Stage 2)
            Sigma_eta = self._Sigma_eta * noise_level
        else:
            # Fallback: matriz identidad escalada (ruido no correlacionado)
            Sigma_eta = noise_level * np.eye(self._N)

        # Condición inicial para ruido correlacionado η(t=0)
        eta_0 = self._random.multivariate_normal(np.zeros(self._N), Sigma_eta)

        # =====================================
        # 5. SIMULACIÓN CON BRAINPY
        # =====================================
        # Usar integrador configurado que implementa dinámicas Eq. 8
        # τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α

        # Tiempo de simulación para alcanzar steady-state (Main paper Fig. 3)
        simulation_steps = int(simulation_time / self._time_res)
        time_array = np.linspace(
            0, simulation_time / 1000.0, simulation_steps
        )  # BrainPy usa segundos

        # Ejecutar integración ODE usando BrainPy
        # self._integrator implementa las dinámicas SSN con ruido
        trajectory = self._integrator(
            [u_0, eta_0],  # Condiciones iniciales [u(0), η(0)]
            time_array,  # Vector de tiempo
            W=W,  # Matriz de conectividad
            h=stimulus,  # Input externo del GSM
            Sigma_eta=Sigma_eta,  # Covarianza del ruido
        )

        # =====================================
        # 6. EXTRACCIÓN DE ACTIVIDAD NEURONAL
        # =====================================
        # trajectory contiene [u(t), η(t)] para todos los tiempos
        u_trajectory = trajectory[0]  # Potenciales de membrana u_α(t)

        # Calcular actividad r_α(t) = k * [u_α(t)]_+^n (Main paper Eq. 9)
        excitatory_activity = self.supralinear_activation(
            u_trajectory[:, : self._N_E]  # Solo neuronas excitatorias
        )
        inhibitory_activity = self.supralinear_activation(
            u_trajectory[:, self._N_E :]  # Solo neuronas inhibitorias
        )

        response = {
            "excitatory": excitatory_activity,  # Actividad excitatorias (Main paper Fig. 4)
            "inhibitory": inhibitory_activity,  # Actividad inhibitorias (Main paper Fig. 4)
        }

        extra = {
            "stimulus_contrast": stimulus_contrast,  # Variable z del modelo GSM (Eq. 2)
            "stimulus_orientation": stimulus_orientation,  # Orientación G del modelo GSM (Eq. 3)
            "noise_level": noise_level,  # Nivel ruido η (Supp. Sec. 2.3)
        }

        return (
            response,
            extra,
        )  # Retorna actividad y parámetros del experimento

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
        - Circular topology: angular differences θi - θj determine connection strength

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
        - Supplementary Material: Connectivity Parameter Optimization

        Parameters
        ----------
        connectivity_params : dict, optional
            Dictionary with the 8 connectivity parameters:
            - 'a_EE', 'a_EI', 'a_IE', 'a_II': connectivity amplitudes
            - 'd_EE', 'd_EI', 'd_IE', 'd_II': connectivity dispersions
            If None, uses stored parameters from training.

        Returns
        -------
        W : np.ndarray, shape(N, N)
            Complete connectivity matrix for use in SSNIntegrator
        """

        # Use trained parameters if available, otherwise use provided params
        if connectivity_params is None:
            if not self.is_trained():
                raise ValueError(
                    "Model must be trained or connectivity_params must be provided"
                )
            # Use stored trained parameters
            params = self._get_connectivity_parameters()
        else:
            params = connectivity_params

        # If matrices already built, return assembled matrix
        if (
            self._W_EE is not None
            and self._W_EI is not None
            and self._W_IE is not None
            and self._W_II is not None
        ):
            W_full = np.zeros((self._N, self._N))
            W_full[: self._N_E, : self._N_E] = self._W_EE
            W_full[: self._N_E, self._N_E :] = self._W_EI
            W_full[self._N_E :, : self._N_E] = self._W_IE
            W_full[self._N_E :, self._N_E :] = self._W_II
            return W_full

        # Genera orientaciones preferidas para ring topology (Main paper Fig. 1B)
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
            # return sign * a * np.exp((np.cos(2 * delta_theta) - 1) / d**2) TODELETE
            return sign * self.parametric_connectivity(delta_theta, 0, a, d)

        # Bloques matriciales
        W[0 : self._N_E, 0 : self._N_E] = connectivity_block(
            theta_e,
            theta_e,
            params["a_EE"],
            params["d_EE"],
            sign=1,
        )
        W[0 : self._N_E, self._N_E : self._N] = connectivity_block(
            theta_e,
            theta_i,
            params["a_EI"],
            params["d_EI"],
            sign=-1,
        )
        W[self._N_E : self._N, 0 : self._N_E] = connectivity_block(
            theta_i,
            theta_e,
            params["a_IE"],
            params["d_IE"],
            sign=1,
        )
        W[self._N_E : self._N, self._N_E : self._N] = connectivity_block(
            theta_i,
            theta_i,
            params["a_II"],
            params["d_II"],
            sign=-1,
        )

        return W

    def _generate_gsm_stimulus(self, contrast, orientation):
        """
        Generate stimulus from Gaussian Scale Mixture (GSM) generative model.

        Mathematical foundation:
        - Main paper, Eq. 1-3: GSM generative model I = z * G where:
          * I: observed image patch
          * z: scale variable (controls contrast)
          * G: Gaussian random field (oriented structure)
        - Main paper, Eq. 4-5: Prior distributions P(z) and P(G)
        - Main paper, Eq. 6-7: Linear projection to neural inputs h = A^T * I
        - Main paper, Figure 1A: Gabor-like receptive fields from GSM optimization
        - Supplementary Material, Section 1: Detailed GSM mathematics

        TODO: Full implementation requires:
        - Generate oriented Gabor filters (receptive fields A)
        - Sample from GSM: I = contrast * gaussian_field(orientation)
        - Project to neural space: h = A^T * I
        """
        # Genera estímulo del modelo generativo GSM
        # TODO: Implementar generación completa siguiendo Ec. 1-7 y Supp. Sec. 1
        # - contrast corresponde a variable z (Eq. 2: P(z) prior de escala)
        # - orientation determina estructura de campo G (Eq. 3: campo Gaussiano orientado)
        # - Debe proyectar I = z*G al espacio neural: h = A^T * I (Eq. 6-7)
        return np.zeros(
            self._N
        )  # Placeholder: debe retornar input h de dimensión N

    def calculate_causes(self, **kwargs):
        """
        Extract causal inference from SSN sampling-based dynamics.

        Mathematical foundation:
        - Main paper, Section 2.1: "Networks approximate Bayesian inference by sampling
          from the posterior distribution P(z,G|I)"
        - Main paper, Eq. 10: Posterior sampling through network dynamics
        - Main paper, Figure 4: Population activity reflects posterior statistics
        - Main paper, Figure 5: Causal inference from multi-modal posterior
        - Supplementary Material, Section 3: Inference analysis methods

        TODO: Implement posterior analysis:
        - Extract samples from network steady-state activity (by sampling-based inference)
        - Compute posterior statistics (mean, variance, modes)
        - Detect number of causes from multimodal posterior (number of filters)
        - Estimate cause positions from population activity peaks (orientation of filters)
        """
        # Extrae inferencia causal de dinámicas de sampling de la red SSN

        return {"num_causes": None, "cause_positions": None}  # Placeholder


# =============================================================================
# USAGE EXAMPLE - Two-stage training following Echeveste et al. (2020)
# =============================================================================
#
# # Initialize untrained model
# model = Echeveste2020(N_E=50, N_I=50, tau_e=20.0, tau_i=10.0)
#
# # Load pre-trained GSM generative model (trained on natural images)
# # GSM contains Gabor filters, scale priors, and Gaussian field parameters
# gsm_model = load_pretrained_gsm_model("./gsm_parameters/")
#
# # Execute two-stage training process (SSN learns inference on GSM)
# training_results = model.train(
#     gsm_model=gsm_model,
#     stage1_params={"optimizer": "adam", "learning_rate": 0.01},  # Connectivity optimization
#     stage2_params={"optimizer": "lbfgs", "tolerance": 1e-6}      # Other parameters
# )
#
# # Save trained parameters (8 parametric + full matrices)
# model.save_parameters("./trained_model/")
#
# # Run sampling-based inference simulation
# result = model.run(stimulus_contrast=0.8, stimulus_orientation=45.0)
#
# TODO: Funciones de utilidad a implementar siguiendo el marco matemático:
#
# - create_gabor_stimulus(): Generar estímulos orientados del modelo generativo GSM
#   Base matemática: Main paper Eq. 1-7, Supp. Section 1
#   Implementación: I = z * G donde z~scale_prior, G~gaussian_field(orientation)
#
# - compute_posterior_stats(): Extraer inferencia Bayesiana de muestras de la red
#   Base matemática: Main paper Eq. 10, Figure 4-5
#   Implementación: Analizar actividad steady-state para estimar estadísticas P(z,G|I)
