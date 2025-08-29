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
            tau_e=tau_e / 1000.0,  # Convierte τ_E = 20ms a segundos
            tau_i=tau_i / 1000.0,  # Convierte τ_I = 10ms a segundos
            n=n,  # Exponente supralineal n = 2.0 (Eq. 9)
            k=k,  # Factor de escala k = 0.3 (Eq. 9)
        )

        # Integrador ODE de BrainPy para dinámicas de Ecuación 8
        self._integrator = bp.odeint(f=integrator_model, **integrator_kws)

        self.set_random(
            np.random.default_rng(seed=seed)
        )  # RNG para ruido η (Supp. Sec. 2.3)

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
        # Ejecuta simulación SSN según protocolo de Echeveste et al.

        # Genera estímulo GSM (Main paper Eq. 1-7: I = z * G)
        stimulus = self._generate_gsm_stimulus(
            stimulus_contrast, stimulus_orientation  # Parámetros de GSM model
        )

        # Establece condiciones iniciales para potenciales de membrana (Eq. 8)
        # Empieza desde estado de reposo: u_α(t=0) = 0 para todas las neuronas α
        u_e_0 = np.zeros(self._N_E)  # Potenciales excitatorios iniciales
        u_i_0 = np.zeros(self._N_I)  # Potenciales inhibitorios iniciales

        # TODO: Implementar loop completo de simulación siguiendo dinámicas Eq. 8
        # Debe integrar: τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        # donde W_αβ se calcula dinámicamente de parámetros a_XY, d_XY (Eq. 10)
        # Usando integrador BrainPy con dt = 0.2ms (Supp. Table S1)
        # Hasta alcanzar régimen de sampling steady-state (Main paper, Fig. 3)

        # Respuesta placeholder - debe contener trayectoria SSN real
        # Forma: (time_steps, neurons) donde time_steps = simulation_time/dt
        excitatory_activity = np.zeros(
            (100, self._N_E)
        )  # Actividad población excitatorias r_E(t)
        inhibitory_activity = np.zeros(
            (100, self._N_I)
        )  # Actividad población inhibitorias r_I(t)

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

    def build_connectivity_matrix(self, connectivity_params):
        """
        Construct connectivity matrix using parametric formulation (Eq.10).
        Mathematical foundation:
        - Main paper, Eq. 10: W_XY(θi,θj) = a_XY * exp[(cos(2(θi-θj))-1)/d_XY²]
        - Only 8 parameters: {a_EE, a_EI, a_IE, a_II, d_EE, d_EI, d_IE, d_II}
        - Supplementary Material: Connectivity Parameter Optimization
        Parameters:
        -----------
        connectivity_params : dict
        Dictionary with the 8 connectivity parameters:
        - 'a_EE', 'a_EI', 'a_IE', 'a_II': connectivity amplitudes
        - 'd_EE', 'd_EI', 'd_IE', 'd_II': connectivity dispersions
        Returns:
        --------
        W : np.ndarray, shape(N, N)
        Complete connectivity matrix for use in SSNIntegrator
        """

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
            connectivity_params["a_EE"],
            connectivity_params["d_EE"],
            sign=1,
        )
        W[0 : self._N_E, self._N_E : self._N] = connectivity_block(
            theta_e,
            theta_i,
            connectivity_params["a_EI"],
            connectivity_params["d_EI"],
            sign=-1,
        )
        W[self._N_E : self._N, 0 : self._N_E] = connectivity_block(
            theta_i,
            theta_e,
            connectivity_params["a_IE"],
            connectivity_params["d_IE"],
            sign=1,
        )
        W[self._N_E : self._N, self._N_E : self._N] = connectivity_block(
            theta_i,
            theta_i,
            connectivity_params["a_II"],
            connectivity_params["d_II"],
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


# TODO: Funciones de utilidad a implementar siguiendo el marco matemático:
#
# - load_ssn_parameters(): Cargar parámetros de conectividad (a_XY, d_XY), inputs h, ruido Σ_η
#   Base matemática: Supp. Material, optimización de 8 parámetros + covarianza ruido
#   Archivos: parámetros (a_XY, d_XY para X,Y∈{E,I}), h (inputs GSM), Sigma_eta (ruido)
#
# - create_gabor_stimulus(): Generar estímulos orientados del modelo generativo GSM
#   Base matemática: Main paper Eq. 1-7, Supp. Section 1
#   Implementación: I = z * G donde z~scale_prior, G~gaussian_field(orientation)
#
# - compute_posterior_stats(): Extraer inferencia Bayesiana de muestras de la red
#   Base matemática: Main paper Eq. 10, Figure 4-5
#   Implementación: Analizar actividad steady-state para estimar estadísticas P(z,G|I)
