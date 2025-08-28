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

        # Computa las dinámicas del SSN según Ecuaciones 8-9 de Echeveste et al.
        # Calcula firing rates usando activación supralineal (Eq. 9: r_α = k[u_α]_+^n)
        r_e = self.supralinear_activation(
            u_e
        )  # Tasas excitatorias (Main paper Eq. 9)
        r_i = self.supralinear_activation(
            u_i
        )  # Tasas inhibitorias (Main paper Eq. 9)

        # Extrae bloques de matriz de conectividad (Main paper Sec. 2.2: estructura E-I)
        # W organizada como: [W_EE W_EI; W_IE W_II] siguiendo principio de Dale
        n_e = len(
            u_e
        )  # Número de neuronas excitatorias (Supp. Table S1: N_E=50)
        W_EE = W[
            :n_e, :n_e
        ]  # Conexiones E→E (Main paper Sec. 2.2, ring topology)
        W_EI = W[:n_e, n_e:]  # Conexiones I→E (Dale: W_EI > 0, inhibe)
        W_IE = W[n_e:, :n_e]  # Conexiones E→I (Dale: W_IE > 0, excita)
        W_II = W[n_e:, n_e:]  # Conexiones I→I (Dale: W_II > 0, inhibe)

        # Implementa dinámicas SSN (Ecuación 8): τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        # Para neuronas excitatorias (α ∈ E):
        du_e_dt = (
            -u_e  # Término de leak: -u_α (Main paper Eq. 8)
            + W_EE
            @ r_e  # Input excitatorio: Σ_β∈E W_αβ r_β (Main paper Eq. 8)
            - W_EI @ r_i  # Input inhibitorio: -Σ_β∈I W_αβ r_β (Dale: W_EI > 0)
            + h[:n_e]  # Input externo del modelo GSM: h_α (Eq. 6-7)
            + eta[:n_e]  # Ruido de inferencia: η_α (Supp. Sec. 2.3)
        ) / self.tau_e  # Divide por constante de tiempo τ_E = 20ms (Supp. Table S1)

        # Para neuronas inhibitorias (α ∈ I):
        du_i_dt = (
            -u_i  # Término de leak: -u_α (Main paper Eq. 8)
            + W_IE
            @ r_e  # Input excitatorio: Σ_β∈E W_αβ r_β (Main paper Eq. 8)
            - W_II @ r_i  # Input inhibitorio: -Σ_β∈I W_αβ r_β (Dale: W_II > 0)
            + h[n_e:]  # Input externo del modelo GSM: h_α (Eq. 6-7)
            + eta[n_e:]  # Ruido de inferencia: η_α (Supp. Sec. 2.3)
        ) / self.tau_i  # Divide por constante de tiempo τ_I = 10ms (Supp. Table S1)

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

    def __init__(
        self,
        *,
        N_E=50,  # Número de neuronas excitatorias (Supp. Table S1)
        N_I=50,  # Número de neuronas inhibitorias (Supp. Table S1)
        tau_e=20.0,  # ms - Constante de tiempo excitatorias (Supp. Table S1)
        tau_i=10.0,  # ms - Constante de tiempo inhibitorias (Supp. Table S1)
        n=2.0,  # Exponente supralineal de Eq. 9 (Supp. Table S1)
        k=0.3,  # Factor de escala de Eq. 9 (Supp. Table S1)
        seed=None,  # Semilla para generador aleatorio
        position_range=(
            0,
            180,
        ),  # Orientaciones ring topology (Main paper, Figure 1B)
        position_res=360 / 100,  # Resolución angular para simetría circular
        time_range=(0, 1000),  # Rango temporal de simulación (ms)
        time_res=0.2,  # Paso temporal dt = 0.2ms (Supp. Table S1)
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
        # Inicializa el modelo SSN con parámetros optimizados de Echeveste et al.

        # Estructura básica de la red (Main paper Sec. 2.2: arquitectura E-I)
        self._N_E = N_E  # Número de neuronas excitatorias (Supp. Table S1: 50)
        self._N_I = N_I  # Número de neuronas inhibitorias (Supp. Table S1: 50)
        self._N = N_E + N_I  # Total de neuronas en ring topology

        # Parámetros espaciales y temporales (Main paper Fig. 1B: ring topology)
        self._position_range = position_range  # Rango orientaciones [0°, 180°]
        self._position_res = position_res  # Resolución angular (3.6°)
        self._time_range = time_range  # Duración simulación (1000ms)
        self._time_res = time_res  # dt = 0.2ms (Supp. Table S1)

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
        # Número de neuronas excitatorias (Supp. Table S1)
        return self._N_E

    @property
    def N_I(self):
        """Number of inhibitory neurons."""
        # Número de neuronas inhibitorias (Supp. Table S1)
        return self._N_I

    @property
    def position_range(self):
        """Range of orientations."""
        # Rango de orientaciones para ring topology (Main paper Fig. 1B)
        return self._position_range

    @property
    def position_res(self):
        """Angular resolution."""
        # Resolución angular para simetría circular (Main paper Fig. 1B)
        return self._position_res

    @property
    def time_range(self):
        """Time range for simulation."""
        # Rango temporal para simulación SSN
        return self._time_range

    @property
    def time_res(self):
        """Time resolution."""
        # Resolución temporal dt (Supp. Table S1: 0.2ms)
        return self._time_res

    # Ejecución del modelo
    def set_random(self, rng):
        """Set random number generator."""
        # Establece generador aleatorio para ruido de inferencia η (Supp. Sec. 2.3)
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
        # stimulus_contrast corresponde a variable de escala z (Eq. 2)
        # stimulus_orientation determina estructura del campo Gaussiano G (Eq. 3)
        stimulus = self._generate_gsm_stimulus(
            stimulus_contrast, stimulus_orientation  # Parámetros de GSM model
        )

        # Establece condiciones iniciales para potenciales de membrana (Eq. 8)
        # Empieza desde estado de reposo: u_α(t=0) = 0 para todas las neuronas α
        u_e_0 = np.zeros(self._N_E)  # Potenciales excitatorios iniciales
        u_i_0 = np.zeros(self._N_I)  # Potenciales inhibitorios iniciales

        # TODO: Implementar loop completo de simulación siguiendo dinámicas Eq. 8
        # Debe integrar: τ_α * du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
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
        - Extract samples from network steady-state activity
        - Compute posterior statistics (mean, variance, modes)
        - Detect number of causes from multimodal posterior
        - Estimate cause positions from population activity peaks
        """
        # Extrae inferencia causal de dinámicas de sampling de la red SSN
        # TODO: Implementar análisis de posterior siguiendo Sec. 2.1 y Fig. 4-5
        # - Extraer muestras de actividad steady-state (Main paper Eq. 10)
        # - Computar estadísticas posterior: P(z,G|I) (Main paper Fig. 4)
        # - Detectar número de causas de posterior multimodal (Fig. 5)
        # - Estimar posiciones desde picos de actividad poblacional
        return {"num_causes": None, "cause_positions": None}  # Placeholder


# TODO: Funciones de utilidad a implementar siguiendo el marco matemático:
#
# - load_ssn_parameters(): Cargar conectividad W, inputs h, ruido Σ_η optimizados
#   Base matemática: Supp. Material, procedimiento de optimización de parámetros
#   Archivos: W (conectividad), h (inputs GSM), Sigma_eta (covarianza ruido inferencia)
#
# - create_gabor_stimulus(): Generar estímulos orientados del modelo generativo GSM
#   Base matemática: Main paper Eq. 1-7, Supp. Section 1
#   Implementación: I = z * G donde z~scale_prior, G~gaussian_field(orientation)
#
# - compute_posterior_stats(): Extraer inferencia Bayesiana de muestras de la red
#   Base matemática: Main paper Eq. 10, Figure 4-5
#   Implementación: Analizar actividad steady-state para estimar estadísticas P(z,G|I)
