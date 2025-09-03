#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copia funcional del código original de Echeveste et al. (2020) para scikit-neuromsi.

Implementa simulación de SSN usando parámetros pre-entrenados cargados desde archivos *_learn.
Basado en: ssn_inference_numerical_experiments/SSN/full_net.py

Uso:
    python echeveste_simulation_copy.py
"""

import numpy as np
import datetime
import os
import sys

# Agregar path al código original 
sys.path.append('/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN')

import methods as mt
from parameters import *

def regularize_Sigma(Sigma, out_location, name, epsilon):
    """Regulariza matriz de covarianza para evitar singularidad."""
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    eigenvalues[eigenvalues < epsilon] = epsilon
    Sigma_reg = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return Sigma_reg

def run_echeveste_simulation(parameter_path="parameter_files", output_path="results", 
                           sample_size=20000, lagged_noise=False):
    """
    Ejecuta simulación completa de SSN usando parámetros pre-entrenados.
    
    Esta función replica exactamente el comportamiento de full_net.py del código original.
    
    Parameters
    ----------
    parameter_path : str
        Ruta a directorio con archivos *_learn
    output_path : str  
        Ruta donde guardar resultados
    sample_size : int
        Número de muestras para estadísticas
    lagged_noise : bool
        Si usar ruido con lag temporal
        
    Returns
    -------
    dict
        Resultados de simulación con momentos finales
    """
    
    # Configuración inicial
    epsilon = 1.0E-12
    
    if lagged_noise:
        print("########################")
        print("#  USING LAGGED NOISE  #") 
        print("########################")
        output_path = parameter_path + "/w_lagged_noise"
        lag_time = tau_e/5.0  # 4ms
        buffer_size = int(lag_time/dt)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("Working with " + str(N_pat) + " patterns")
    
    # ======================================
    # Carga de parámetros pre-entrenados
    # ======================================
    
    print("Loading pre-trained parameters...")
    
    # Matriz de covarianza del ruido (100x100)
    Sigma_eta = np.loadtxt(os.path.join(parameter_path, "sigma_eta_learn"))
    
    # Inputs para cada patrón de contraste (5 patrones × 100 neuronas)
    h = np.empty([N_pat, N])
    for alpha in range(N_pat):
        h[alpha] = np.loadtxt(os.path.join(parameter_path, f"h_true_{alpha}_learn"))
    
    # Matriz de conectividad completa (100x100)
    W = np.loadtxt(os.path.join(parameter_path, "w_learn"))
    
    print(f"Loaded connectivity matrix W: {W.shape}")
    print(f"Loaded noise covariance Sigma_eta: {Sigma_eta.shape}")
    print(f"Loaded {N_pat} input patterns h")
    
    # ======================================
    # Condiciones iniciales
    # ======================================
    
    mu_0 = np.empty([N_pat, N])
    Sigma_0 = np.empty([N_pat, N, N])
    
    for alpha in range(N_pat):
        mu_0[alpha] = np.loadtxt(os.path.join(parameter_path, f"mu_learn_{alpha}"))
        Sigma_0[alpha] = np.loadtxt(os.path.join(parameter_path, f"sigma_learn_{alpha}"))
        Sigma_0[alpha] = regularize_Sigma(Sigma_0[alpha], output_path, f"Sigma_0{alpha}", epsilon)
    
    # Arrays para resultados finales
    mu_final = np.empty([N_pat, N])
    Sigma_final = np.empty([N_pat, N, N])
    nu_final = np.empty([N_pat, N])
    Lambda_final = np.empty([N_pat, N, N])
    
    # Parámetros de muestreo
    t_bet_samp = 10 * tau_e  # Tiempo entre muestras
    steps_bet_samp = int(t_bet_samp/dt)
    
    print("Start time:", datetime.datetime.now())
    
    # ======================================
    # Simulación para cada patrón
    # ======================================
    
    if lagged_noise:
        L = np.linalg.cholesky(Sigma_eta)
    
    results = {}
    
    for alpha in range(N_pat):
        print(f"Processing pattern {alpha}...")
        
        # Condiciones iniciales aleatorias
        u0 = np.random.multivariate_normal(mean=mu_0[alpha], cov=Sigma_0[alpha])
        eta0 = np.random.multivariate_normal(mean=np.zeros(N), cov=Sigma_eta)
        
        if lagged_noise:
            print("Buffering noise")
            buffered_noise = mt.buffer_init(buffer_size, L)
            
            print(f"Evolving pattern {alpha}")
            u, eta, buffered_noise = mt.network_evolution_w_lagged_noise(
                W, h[alpha], u0, Sigma_eta, L, buffered_noise, eta=eta0
            )
            
            print("Sampling from the network")
            u_samples, eta_samples, _, _, buffered_noise = mt.network_sample_w_lagged_noise(
                W, h[alpha], u, eta, sample_size, steps_bet_samp, Sigma_eta, L, buffered_noise
            )
            
            np.save(os.path.join(output_path, f"eta_samples_{alpha}"), eta_samples)
        
        else:
            print(f"Evolving pattern {alpha}")
            u, eta = mt.network_evolution(W, h[alpha], u0, Sigma_eta, eta=eta0)
            
            print("Sampling moments")
            u_samples, _, _, _ = mt.network_sample(W, h[alpha], u, eta, sample_size, steps_bet_samp, Sigma_eta)
        
        # Calcular momentos finales
        mu_final[alpha] = np.mean(u_samples, axis=0)
        Sigma_final[alpha] = np.cov(u_samples, rowvar=False)
        
        # Calcular actividad y sus momentos
        r_samples = mt.get_r(u_samples)
        nu_final[alpha] = np.mean(r_samples, axis=0)
        Lambda_final[alpha] = np.cov(r_samples, rowvar=False)
        
        # Guardar resultados individuales
        np.savetxt(os.path.join(output_path, f"mu_evolved_net_{alpha}"), mu_final[alpha])
        np.savetxt(os.path.join(output_path, f"sigma_evolved_net_{alpha}"), Sigma_final[alpha])
        np.savetxt(os.path.join(output_path, f"std_evolved_net_{alpha}"), np.sqrt(np.diag(Sigma_final[alpha])))
        
        np.savetxt(os.path.join(output_path, f"nu_evolved_net_{alpha}"), nu_final[alpha])
        np.savetxt(os.path.join(output_path, f"lambda_evolved_net_{alpha}"), Lambda_final[alpha])
        np.savetxt(os.path.join(output_path, f"std_r_evolved_net_{alpha}"), np.sqrt(np.diag(Lambda_final[alpha])))
        
        # Almacenar en diccionario de resultados
        results[f"pattern_{alpha}"] = {
            "mu_final": mu_final[alpha],
            "Sigma_final": Sigma_final[alpha],
            "nu_final": nu_final[alpha], 
            "Lambda_final": Lambda_final[alpha],
            "u_samples": u_samples,
            "r_samples": r_samples
        }
    
    # Guardar resultados consolidados
    np.savetxt(os.path.join(output_path, "all_mus_evolved_net"), mu_final)
    np.savetxt(os.path.join(output_path, "all_nus_evolved_net"), nu_final)
    
    print("End time:", datetime.datetime.now())
    
    return {
        "mu_final": mu_final,
        "Sigma_final": Sigma_final,
        "nu_final": nu_final,
        "Lambda_final": Lambda_final,
        "results_by_pattern": results,
        "parameters_used": {
            "W": W,
            "h": h,
            "Sigma_eta": Sigma_eta,
            "sample_size": sample_size
        }
    }

def verify_parameter_files(parameter_path="parameter_files"):
    """
    Verifica que todos los archivos de parámetros necesarios existan.
    
    Returns
    -------
    bool
        True si todos los archivos existen
    """
    required_files = [
        "w_learn",
        "sigma_eta_learn",
    ]
    
    # Archivos por patrón
    for alpha in range(N_pat):
        required_files.extend([
            f"h_true_{alpha}_learn",
            f"mu_learn_{alpha}",
            f"sigma_learn_{alpha}"
        ])
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(parameter_path, filename)
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        print("Missing parameter files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    print("All parameter files found!")
    return True

def load_connectivity_parameters(parameter_path="parameter_files"):
    """
    Carga los 8 parámetros de conectividad parametrica del código original.
    
    Returns
    -------
    dict
        Diccionario con los 8 parámetros a_XY y d_XY
    """
    params = {}
    
    # Mapeo de nombres de archivos a parámetros
    param_mapping = {
        "w_ee_height_learn": "a_EE",
        "w_ee_width_learn": "d_EE", 
        "w_ei_height_learn": "a_EI",
        "w_ei_width_learn": "d_EI",
        "w_ie_height_learn": "a_IE", 
        "w_ie_width_learn": "d_IE",
        "w_ii_height_learn": "a_II",
        "w_ii_width_learn": "d_II"
    }
    
    for filename, param_name in param_mapping.items():
        filepath = os.path.join(parameter_path, filename)
        if os.path.exists(filepath):
            params[param_name] = np.loadtxt(filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    return params

if __name__ == "__main__":
    # Verificar archivos de parámetros
    parameter_dir = "/home/molina/FAMAF/5to-Famaf/TESIS/ssn_inference_numerical_experiments/SSN/parameter_files"
    
    if not verify_parameter_files(parameter_dir):
        print("Cannot run simulation - missing parameter files")
        sys.exit(1)
    
    # Cargar parámetros de conectividad parametrica
    connectivity_params = load_connectivity_parameters(parameter_dir)
    print("Loaded connectivity parameters:")
    for k, v in connectivity_params.items():
        print(f"  {k}: {v}")
    
    # Ejecutar simulación completa
    results = run_echeveste_simulation(
        parameter_path=parameter_dir,
        output_path="results_copy_test",
        sample_size=5000  # Reduced for testing
    )
    
    print(f"Simulation completed. Results saved to results_copy_test/")
    print(f"Final moments computed for {len(results['results_by_pattern'])} patterns")