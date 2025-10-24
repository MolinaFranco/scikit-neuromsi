#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""
Visualization module for Echeveste et al. (2020) SSN model dynamics.

This module provides functions to visualize neural states during simulation,
including membrane potentials, firing rates, oscillations, transients, and
other important dynamics metrics.

Mathematical foundation:
    - Echeveste et al. (2020), Nature Neuroscience: "Cortical-like dynamics in
      recurrent circuits optimized for sampling-based probabilistic inference"
    - Main paper figures 2-7 provide the visualization templates
    - Supplementary Material contains detailed mathematical analyses

References:
    - echepaper.pdf: Main paper with visualization examples
    - Supplementary_information_echerpaper(1).pdf: Mathematical details
    - Original code: SSN repository (network_evolution, plotting functions)
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def _extract_mode_data(result, mode_name):
    """
    Extract and reshape data for a specific mode from NDResult.

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    mode_name : str
        Name of the mode (e.g., 'excitatory_firing_rate',
        'excitatory_potential', etc.)

    Returns
    -------
    data : numpy.ndarray
        Reshaped data with shape (time_steps, n_neurons)
    """
    df = result.get_modes()
    mode_series = df[mode_name]
    # Reshape to (time_steps, n_neurons)
    n_times = len(result.times_)
    data = mode_series.values.reshape(n_times, -1)
    return data


def plot_membrane_potentials(
    result,
    neuron_indices=None,
    population="both",
    figsize=(12, 6),
    title=None,
):
    """
    Plot membrane potentials u(t) over time for selected neurons.

    Visualizes the temporal evolution of membrane potentials, which represent
    the latent variable values in the sampling-based inference framework.

    Mathematical foundation:
        - Main paper Eq. 8: τ_α du_α/dt = -u_α + Σ_β W_αβ r_β + h_α + η_α
        - u_α(t) represents the membrane potential of neuron α
        - Related to Main paper Fig. 2, 4, 5

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run() containing simulation data
    neuron_indices : list of int, optional
        Indices of specific neurons to plot. If None, plots first 5 neurons
        from each population
    population : {"excitatory", "inhibitory", "both"}
        Which population(s) to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Custom title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes or array of Axes
        The axes object(s)

    Examples
    --------
    >>> ssn = Echeveste2020(N_E=50, N_I=50)
    >>> ssn.train()
    >>> result = ssn.run(stimulus_contrast=0.019, simulation_time=500.0)
    >>> fig, ax = plot_membrane_potentials(result, neuron_indices=[0,1,2,3,4])
    """
    # Extraer potenciales de membrana u_α(t) del resultado
    excitatory = _extract_mode_data(
        result, "excitatory_potential"
    )  # (time_steps, N_E)
    inhibitory = _extract_mode_data(
        result, "inhibitory_potential"
    )  # (time_steps, N_I)

    # Obtener eje temporal desde el resultado
    time_axis = result.times_  # en milisegundos

    # Determinar índices de neuronas a graficar
    if neuron_indices is None:
        # Por defecto: primeras 5 neuronas de cada población
        n_exc = min(5, excitatory.shape[1])
        n_inh = min(5, inhibitory.shape[1])
        exc_indices = list(range(n_exc))
        inh_indices = list(range(n_inh))
    else:
        exc_indices = neuron_indices
        inh_indices = neuron_indices

    # Crear figura según población solicitada
    if population == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Graficar neuronas excitatorias
        for idx in exc_indices:
            if idx < excitatory.shape[1]:
                ax1.plot(
                    time_axis,
                    excitatory[:, idx],
                    alpha=0.7,
                    label=f"E{idx}",
                )
        ax1.set_ylabel("Membrane potential u (a.u.)")
        ax1.set_title("Excitatory neurons - Membrane potentials")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Graficar neuronas inhibitorias
        for idx in inh_indices:
            if idx < inhibitory.shape[1]:
                ax2.plot(
                    time_axis,
                    inhibitory[:, idx],
                    alpha=0.7,
                    label=f"I{idx}",
                    color=f"C{idx}"
                )
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Membrane potential u (a.u.)")
        ax2.set_title("Inhibitory neurons - Membrane potentials")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.00)

        plt.tight_layout()
        return fig, (ax1, ax2)

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if population == "excitatory_firing_rate":
            data = excitatory
            pop_name = "Excitatory"
            indices = exc_indices
        else:  # inhibitory
            data = inhibitory
            pop_name = "Inhibitory"
            indices = inh_indices

        for idx in indices:
            if idx < data.shape[1]:
                ax.plot(
                    time_axis,
                    data[:, idx],
                    alpha=0.7,
                    label=f"{pop_name[0]}{idx}"
                )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Membrane potential u (a.u.)")
        ax.set_title(title or f"{pop_name} neurons - Membrane potentials")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax


def plot_firing_rates(
    result,
    neuron_indices=None,
    population="both",
    figsize=(12, 6),
    title=None,
):
    """
    Plot firing rates r(t) over time for selected neurons.

    Visualizes the temporal evolution of firing rates computed from membrane
    potentials via the supralinear activation function.

    Mathematical foundation:
        - Main paper Eq. 9: r_α = k * [u_α]_+^n where k=0.3, n=2.0
        - r_α(t) represents the firing rate of neuron α
        - Related to Main paper Fig. 2, 4

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run() containing simulation data
    neuron_indices : list of int, optional
        Indices of specific neurons to plot. If None, plots first 5 neurons
        from each population
    population : {"excitatory", "inhibitory", "both"}
        Which population(s) to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
    title : str, optional
        Custom title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes or array of Axes
        The axes object(s)

    Examples
    --------
    >>> ssn = Echeveste2020(N_E=50, N_I=50)
    >>> ssn.train()
    >>> result = ssn.run(stimulus_contrast=0.019, simulation_time=500.0)
    >>> fig, ax = plot_firing_rates(result, neuron_indices=[0,1,2,3,4])
    """
    # Extraer firing rates r_α(t) del resultado
    excitatory = _extract_mode_data(
        result, "excitatory_firing_rate"
    )  # (time_steps, N_E)
    inhibitory = _extract_mode_data(
        result, "inhibitory_firing_rate"
    )  # (time_steps, N_I)

    # Obtener eje temporal desde el resultado
    time_axis = result.times_  # en milisegundos

    # Determinar índices de neuronas a graficar
    if neuron_indices is None:
        # Por defecto: primeras 5 neuronas de cada población
        n_exc = min(5, excitatory.shape[1])
        n_inh = min(5, inhibitory.shape[1])
        exc_indices = list(range(n_exc))
        inh_indices = list(range(n_inh))
    else:
        exc_indices = neuron_indices
        inh_indices = neuron_indices

    # Crear figura según población solicitada
    if population == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Graficar neuronas excitatorias
        for idx in exc_indices:
            if idx < excitatory.shape[1]:
                ax1.plot(
                    time_axis,
                    excitatory[:, idx],
                    alpha=0.7,
                    label=f"E{idx}",
                )
        ax1.set_ylabel("Firing rate (Hz)")
        ax1.set_title("Excitatory neurons - Firing rates")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Graficar neuronas inhibitorias
        for idx in inh_indices:
            if idx < inhibitory.shape[1]:
                ax2.plot(
                    time_axis,
                    inhibitory[:, idx],
                    alpha=0.7,
                    label=f"I{idx}",
                    color=f"C{idx}"
                )
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Firing rate (Hz)")
        ax2.set_title("Inhibitory neurons - Firing rates")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=14, y=1.00)

        plt.tight_layout()
        return fig, (ax1, ax2)

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if population == "excitatory_firing_rate":
            data = excitatory
            pop_name = "Excitatory"
            indices = exc_indices
        else:  # inhibitory
            data = inhibitory
            pop_name = "Inhibitory"
            indices = inh_indices

        for idx in indices:
            if idx < data.shape[1]:
                ax.plot(
                    time_axis,
                    data[:, idx],
                    alpha=0.7,
                    label=f"{pop_name[0]}{idx}"
                )

        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Firing rate (Hz)")
        ax.set_title(title or f"{pop_name} neurons - Firing rates")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax


def plot_population_activity(
    result, figsize=(12, 8), n_neurons=None, title=None
):
    """
    Plot population activity as raster-like plots for E and I populations.

    Visualizes the activity of all neurons over time, similar to Main paper
    Fig. 2a showing population firing patterns.

    Mathematical foundation:
        - Main paper Fig. 2a: "Population activity" showing E and I responses
        - Firing rates r_α(t) = k[u_α(t)]_+^n derived from membrane potentials
        - Color intensity represents firing rate magnitude

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    figsize : tuple, optional
        Figure size (width, height)
    n_neurons : int, optional
        Number of neurons to display. If None, shows all neurons
    title : str, optional
        Custom title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : array of matplotlib.axes.Axes
    """
    excitatory = _extract_mode_data(
        result, "excitatory_firing_rate"
    )  # (time_steps, N_E)
    inhibitory = _extract_mode_data(
        result, "inhibitory_firing_rate"
    )  # (time_steps, N_I)
    time_axis = result.times_

    # Limitar número de neuronas si se especifica
    if n_neurons is not None:
        excitatory = excitatory[:, :n_neurons]
        inhibitory = inhibitory[:, :n_neurons]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Graficar actividad excitatorias como heatmap
    im1 = ax1.imshow(
        excitatory.T,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, excitatory.shape[1]],
        origin="lower",
    )
    ax1.set_ylabel("Excitatory neuron #")
    ax1.set_title("Excitatory population activity")
    plt.colorbar(im1, ax=ax1, label="Firing rate (Hz)")

    # Graficar actividad inhibitorias como heatmap
    im2 = ax2.imshow(
        inhibitory.T,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, inhibitory.shape[1]],
        origin="lower",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Inhibitory neuron #")
    ax2.set_title("Inhibitory population activity")
    plt.colorbar(im2, ax=ax2, label="Firing rate (Hz)")

    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_mean_firing_rates(
    result, window_size=50, figsize=(12, 5), title=None
):
    """
    Plot mean firing rates and standard deviations over time.

    Shows population-averaged activity with variability, similar to Main
    paper Fig. 2b showing mean and std of neural responses.

    Mathematical foundation:
        - Main paper Fig. 2b: Mean ± std of firing rates across population
        - Population average: ⟨r_E(t)⟩ and ⟨r_I(t)⟩
        - Variability modulated by stimulus (Main paper Fig. 5d)

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    window_size : int, optional
        Window size for smoothing (in time steps)
    figsize : tuple, optional
        Figure size
    title : str, optional
        Custom title

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    excitatory = _extract_mode_data(result, "excitatory_firing_rate")
    inhibitory = _extract_mode_data(result, "inhibitory_firing_rate")
    time_axis = result.times_

    # Calcular media y desviación estándar poblacional
    exc_mean = np.mean(excitatory, axis=1)
    exc_std = np.std(excitatory, axis=1)
    inh_mean = np.mean(inhibitory, axis=1)
    inh_std = np.std(inhibitory, axis=1)

    # Suavizado con media móvil (opcional)
    if window_size > 1:
        from scipy.ndimage import uniform_filter1d

        exc_mean = uniform_filter1d(exc_mean, size=window_size, mode="nearest")
        exc_std = uniform_filter1d(exc_std, size=window_size, mode="nearest")
        inh_mean = uniform_filter1d(inh_mean, size=window_size, mode="nearest")
        inh_std = uniform_filter1d(inh_std, size=window_size, mode="nearest")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Graficar media ± std para excitatorias
    ax.plot(time_axis, exc_mean, label="Excitatory mean", color="C0", lw=2)
    ax.fill_between(
        time_axis,
        exc_mean - exc_std,
        exc_mean + exc_std,
        alpha=0.3,
        color="C0",
        label="Excitatory ± std",
    )

    # Graficar media ± std para inhibitorias
    ax.plot(time_axis, inh_mean, label="Inhibitory mean", color="C1", lw=2)
    ax.fill_between(
        time_axis,
        inh_mean - inh_std,
        inh_mean + inh_std,
        alpha=0.3,
        color="C1",
        label="Inhibitory ± std",
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Population firing rate (Hz)")
    ax.set_title(title or "Mean population firing rates")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_autocorrelation(
    result, max_lag=200, population="excitatory", figsize=(12, 5), title=None
):
    """
    Plot temporal autocorrelation of neural activity.

    Analyzes temporal correlations in firing rates, revealing oscillatory
    dynamics as shown in Main paper Fig. 4a.

    Mathematical foundation:
        - Main paper Fig. 4a: Autocorrelation functions with oscillations
        - Main paper Eq. 11: Noise correlations
          ⟨η(t)η(t+s)^T⟩ = Σ_η exp(-s/τ_η)
        - Oscillations from E-I interactions (20-80 Hz gamma band)

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    max_lag : int, optional
        Maximum lag to compute (in time steps)
    population : {"excitatory", "inhibitory"}
        Which population to analyze
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Custom title for the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if population == "excitatory_firing_rate":
        data = _extract_mode_data(result, "excitatory_firing_rate")
        pop_name = "Excitatory"
    else:
        data = _extract_mode_data(result, "inhibitory_firing_rate")
        pop_name = "Inhibitory"

    time_axis = result.times_
    dt = time_axis[1] - time_axis[0]  # resolución temporal en ms

    # Calcular autocorrelación promedio sobre todas las neuronas
    n_neurons = data.shape[1]
    autocorr_sum = np.zeros(max_lag)

    for neuron_idx in range(n_neurons):
        activity = data[:, neuron_idx]
        # Normalizar actividad (mean-centering)
        activity_centered = activity - np.mean(activity)

        # Calcular autocorrelación usando correlate
        autocorr = np.correlate(
            activity_centered, activity_centered, mode="full"
        )
        # Tomar solo la mitad positiva y normalizar
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]  # Normalizar por lag=0

        # Acumular
        if len(autocorr) >= max_lag:
            autocorr_sum += autocorr[:max_lag]

    # Promediar sobre neuronas
    autocorr_mean = autocorr_sum / n_neurons

    # Crear eje de lags en ms
    lag_axis = np.arange(max_lag) * dt

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(lag_axis, autocorr_mean, lw=2, color="C0")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time lag (ms)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title or f"{pop_name} population autocorrelation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_power_spectrum(
    result,
    population="excitatory_firing_rate",
    freq_range=(0, 100),
    figsize=(12, 5),
    title=None,
):
    """
    Plot power spectrum to analyze oscillatory dynamics (gamma oscillations).

    Reveals oscillatory frequencies in neural activity, particularly gamma
    oscillations (20-80 Hz) as shown in Main paper Fig. 5c and Fig. 6.

    Mathematical foundation:
        - Main paper Fig. 5c: Power spectral density showing gamma peak
        - Main paper Fig. 6: Oscillation analysis across principal components
        - Supplementary Sec. S2.2: Mathematical analysis of oscillations
        - Gamma oscillations emerge from E-I recurrent dynamics

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    population : {"excitatory", "inhibitory"}
        Which population to analyze
    freq_range : tuple, optional
        (min_freq, max_freq) in Hz to display
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if population == "excitatory_firing_rate":
        data = _extract_mode_data(result, "excitatory_firing_rate")
        pop_name = "Excitatory"
    else:
        data = _extract_mode_data(result, "inhibitory_firing_rate")
        pop_name = "Inhibitory"

    time_axis = result.times_
    dt = (time_axis[1] - time_axis[0]) / 1000.0  # convertir ms a segundos
    fs = 1.0 / dt  # frecuencia de muestreo en Hz

    # Calcular espectro de potencia promedio sobre todas las neuronas
    n_neurons = data.shape[1]
    psd_sum = None

    for neuron_idx in range(n_neurons):
        activity = data[:, neuron_idx]

        # Calcular PSD usando método de Welch
        freqs, psd = signal.welch(
            activity, fs=fs, nperseg=min(256, len(activity) // 4)
        )

        if psd_sum is None:
            psd_sum = psd
        else:
            psd_sum += psd

    # Promediar sobre neuronas
    psd_mean = psd_sum / n_neurons

    # Filtrar rango de frecuencias de interés
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_plot = freqs[freq_mask]
    psd_plot = psd_mean[freq_mask]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.semilogy(freqs_plot, psd_plot, lw=2, color="C0")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.set_title(title or f"{pop_name} population power spectrum")
    ax.grid(True, alpha=0.3, which="both")

    # Marcar banda gamma (20-80 Hz) si está en el rango
    if freq_range[0] <= 20 and freq_range[1] >= 80:
        ax.axvspan(
            20, 80, alpha=0.1, color="red", label="Gamma band (20-80 Hz)"
        )
        ax.legend()

    plt.tight_layout()
    return fig, ax


def plot_transient_analysis(
    result,
    baseline_window=(0, 100),
    transient_window=(100, 300),
    figsize=(12, 5),
    title=None,
):
    """
    Analyze transient overshoots in neural responses.

    Visualizes transient dynamics when stimulus is presented, showing
    characteristic overshoots as analyzed in Main paper Fig. 7.

    Mathematical foundation:
        - Main paper Fig. 7: Transient overshoots analysis
        - Supplementary Sec. S2.3: Mathematical transients analysis
        - Transients arise from fast E-I dynamics before settling
        - Overshoot magnitude correlates with stimulus features

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    baseline_window : tuple, optional
        (start_time, end_time) in ms for baseline period
    transient_window : tuple, optional
        (start_time, end_time) in ms for transient analysis
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    excitatory = _extract_mode_data(result, "excitatory_firing_rate")
    time_axis = result.times_

    # Encontrar índices de tiempo para las ventanas
    baseline_mask = (time_axis >= baseline_window[0]) & (
        time_axis <= baseline_window[1]
    )
    transient_mask = (time_axis >= transient_window[0]) & (
        time_axis <= transient_window[1]
    )

    # Calcular media poblacional
    exc_mean = np.mean(excitatory, axis=1)

    # Calcular baseline y máximo transitorio
    baseline = np.mean(exc_mean[baseline_mask])
    transient_max = np.max(exc_mean[transient_mask])
    overshoot = transient_max - baseline

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Graficar actividad media
    ax.plot(time_axis, exc_mean, lw=2, color="C0", label="Mean firing rate")

    # Marcar ventanas de análisis
    ax.axvspan(
        baseline_window[0],
        baseline_window[1],
        alpha=0.2,
        color="green",
        label="Baseline",
    )
    ax.axvspan(
        transient_window[0],
        transient_window[1],
        alpha=0.2,
        color="red",
        label="Transient",
    )

    # Marcar niveles
    ax.axhline(baseline, color="green", linestyle="--", alpha=0.5)
    ax.axhline(transient_max, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title(
        title
        or f"Transient overshoot analysis (overshoot = {overshoot:.2f} Hz)"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_fano_factor(
    result,
    window_size=50,
    figsize=(12, 5),
    title=None,
):
    """
    Plot Fano factor (variance-to-mean ratio) over time.

    Analyzes trial-to-trial variability as shown in Main paper
    Fig. 5d, revealing stimulus modulation of neural variability.

    Mathematical foundation:
        - Main paper Fig. 5d: Fano factor vs time
        - Fano factor: FF = Var(r) / Mean(r)
        - Variability modulated by stimulus intensity
          (sampling-based inference)
        - Lower variability for higher contrast stimuli

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    window_size : int, optional
        Window size for computing variance and mean (in time steps)
    figsize : tuple
    title : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    excitatory = _extract_mode_data(result, "excitatory_firing_rate")
    time_axis = result.times_

    # Calcular media y varianza poblacional en ventanas temporales
    n_steps = len(time_axis)
    n_windows = n_steps - window_size + 1

    fano_factors = np.zeros(n_windows)
    window_times = np.zeros(n_windows)

    for i in range(n_windows):
        window_data = excitatory[i:i + window_size, :]
        mean_activity = np.mean(window_data)
        var_activity = np.var(window_data)

        # Fano factor: variance / mean
        if mean_activity > 0:
            fano_factors[i] = var_activity / mean_activity
        else:
            fano_factors[i] = np.nan

        window_times[i] = time_axis[i + window_size // 2]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(window_times, fano_factors, lw=2, color="C0")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Fano Factor (Var/Mean)")
    ax.set_title(title or "Fano factor over time (excitatory population)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_neural_dynamics_summary(result, figsize=(16, 12)):
    """
    Create a comprehensive summary figure with multiple dynamics plots.

    Combines several key analyses in a single figure, similar to the
    multi-panel figures in the Main paper (e.g., Fig. 5).

    Mathematical foundation:
        - Combines multiple analyses from Main paper Figures 2-7
        - Provides overview of: activity, oscillations, variability, transients

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    figsize : tuple, optional
        Figure size

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : array of matplotlib.axes.Axes
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    excitatory = _extract_mode_data(result, "excitatory_firing_rate")
    inhibitory = _extract_mode_data(result, "inhibitory_firing_rate")
    time_axis = result.times_

    # Panel A: Actividad poblacional media
    ax1 = fig.add_subplot(gs[0, 0])
    exc_mean = np.mean(excitatory, axis=1)
    inh_mean = np.mean(inhibitory, axis=1)
    ax1.plot(time_axis, exc_mean, label="Excitatory", color="C0", lw=2)
    ax1.plot(time_axis, inh_mean, label="Inhibitory", color="C1", lw=2)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Mean firing rate (Hz)")
    ax1.set_title("A. Population mean activity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel B: Raster de actividad excitatorias (primeras 30 neuronas)
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(30, excitatory.shape[1])
    im = ax2.imshow(
        excitatory[:, :n_show].T,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        extent=[time_axis[0], time_axis[-1], 0, n_show],
        origin="lower",
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Neuron #")
    ax2.set_title("B. Excitatory activity (first 30 neurons)")
    plt.colorbar(im, ax=ax2, label="Firing rate (Hz)")

    # Panel C: Autocorrelación
    ax3 = fig.add_subplot(gs[1, 0])
    max_lag = min(200, len(time_axis) // 2)
    dt = time_axis[1] - time_axis[0]

    # Calcular autocorrelación promedio
    autocorr_sum = np.zeros(max_lag)
    for neuron_idx in range(min(10, excitatory.shape[1])):
        activity = excitatory[:, neuron_idx]
        activity_centered = activity - np.mean(activity)
        autocorr = np.correlate(
            activity_centered, activity_centered, mode="full"
        )
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = autocorr / autocorr[0]
        if len(autocorr) >= max_lag:
            autocorr_sum += autocorr[:max_lag]

    autocorr_mean = autocorr_sum / min(10, excitatory.shape[1])
    lag_axis = np.arange(max_lag) * dt

    ax3.plot(lag_axis, autocorr_mean, lw=2, color="C0")
    ax3.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax3.set_xlabel("Time lag (ms)")
    ax3.set_ylabel("Autocorrelation")
    ax3.set_title("C. Temporal autocorrelation")
    ax3.grid(True, alpha=0.3)

    # Panel D: Espectro de potencia
    ax4 = fig.add_subplot(gs[1, 1])
    dt_sec = dt / 1000.0
    fs = 1.0 / dt_sec

    # Calcular PSD promedio
    psd_sum = None
    for neuron_idx in range(min(10, excitatory.shape[1])):
        activity = excitatory[:, neuron_idx]
        freqs, psd = signal.welch(
            activity, fs=fs, nperseg=min(256, len(activity) // 4)
        )
        if psd_sum is None:
            psd_sum = psd
        else:
            psd_sum += psd

    psd_mean = psd_sum / min(10, excitatory.shape[1])
    freq_mask = (freqs >= 0) & (freqs <= 100)

    ax4.semilogy(freqs[freq_mask], psd_mean[freq_mask], lw=2, color="C0")
    ax4.axvspan(20, 80, alpha=0.1, color="red", label="Gamma (20-80 Hz)")
    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Power Spectral Density")
    ax4.set_title("D. Power spectrum")
    ax4.legend()
    ax4.grid(True, alpha=0.3, which="both")

    # Panel E: Varianza poblacional
    ax5 = fig.add_subplot(gs[2, 0])
    exc_std = np.std(excitatory, axis=1)
    inh_std = np.std(inhibitory, axis=1)
    ax5.plot(time_axis, exc_std, label="Excitatory", color="C0", lw=2)
    ax5.plot(time_axis, inh_std, label="Inhibitory", color="C1", lw=2)
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Population std")
    ax5.set_title("E. Population variability")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Panel F: Fano factor
    ax6 = fig.add_subplot(gs[2, 1])
    window_size = min(50, len(time_axis) // 10)
    n_windows = len(time_axis) - window_size + 1
    fano_factors = np.zeros(n_windows)
    window_times = np.zeros(n_windows)

    for i in range(n_windows):
        window_data = excitatory[i:i + window_size, :]
        mean_activity = np.mean(window_data)
        var_activity = np.var(window_data)
        if mean_activity > 0:
            fano_factors[i] = var_activity / mean_activity
        else:
            fano_factors[i] = np.nan
        window_times[i] = time_axis[i + window_size // 2]

    ax6.plot(window_times, fano_factors, lw=2, color="C0")
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Fano Factor")
    ax6.set_title("F. Fano factor (Var/Mean)")
    ax6.grid(True, alpha=0.3)

    # Título general
    fig.suptitle(
        "Neural Dynamics Summary - Echeveste et al. (2020) SSN Model",
        fontsize=16,
        y=0.995,
    )

    plt.tight_layout()
    return fig, fig.axes


def plot_heatmap_activity(result, population='excitatory', figsize=(16, 10),
                          orientation_centered=True):
    """
    Create heatmap visualization showing ALL neurons' activity over time.

    This function generates a comprehensive heatmap visualization that displays
    the activity of all neurons (50 E or 50 I) as a color-coded matrix where:
    - Rows represent individual neurons (IDs or orientations)
    - Columns represent time points
    - Color intensity represents activity level

    Mathematical foundation:
        - Main paper Eq. 8: Membrane potential dynamics u_α(t)
        - Main paper Eq. 9: Firing rates r_α(t) = k * [u_α]_+^n
        - Visualizes complete population activity patterns
        - Main paper Fig. 2A: Example heatmap with centered orientations

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    population : str, optional
        'excitatory' or 'inhibitory', default 'excitatory'
    figsize : tuple, optional
        Figure size, default (16, 10)
    orientation_centered : bool, optional
        If True, show orientations as -90° to 90° (like Fig. 2A) (default)
        If False, show neuron IDs 0-49

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the heatmap panels
    axes : tuple
        Tuple of axes (ax1, ax2, ax3, ax4) for customization

    Examples
    --------
    >>> ssn = Echeveste2020(N_E=50, N_I=50)
    >>> result = ssn.run(stimulus_contrast=0.5, simulation_time=200.0)
    >>> # Default: Centered orientations -90° to 90° (like Fig. 2A)
    >>> fig, axes = plot_heatmap_activity(result, population='excitatory')
    >>> # Alternative: Show neuron IDs 0-49
    >>> fig, axes = plot_heatmap_activity(result,
    ...                                   orientation_centered=False)

    Notes
    -----
    The figure contains 4 panels:
    1. Heatmap of membrane potentials u(t) for all neurons
       Units: arbitrary (a.u.) - this is a rate model, not conductance
    2. Heatmap of firing rates r(t) for all neurons
       Units: Hz (spikes/second)
    3. Bar plot: average firing rate per neuron (spatial tuning)
    4. Time series: population mean ± std over time

    Orientation conventions:
    - orientation_centered=False: Y-axis = Neuron ID (0 to N-1)
    - orientation_centered=True: Y-axis = Orientation (-90° to 90°)
      Assumes ring topology covering 0° to 180°, recentered at 0°
    """
    from matplotlib.gridspec import GridSpec

    df = result.get_modes()
    n_times = len(result.times_)

    if population == 'excitatory':
        u = df['excitatory_potential'].values.reshape(n_times, -1)
        r = df['excitatory_firing_rate'].values.reshape(n_times, -1)
        pop_name = 'Excitatory'
        color_u = 'RdBu_r'
        color_r = 'viridis'
    else:
        u = df['inhibitory_potential'].values.reshape(n_times, -1)
        r = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
        pop_name = 'Inhibitory'
        color_u = 'RdBu_r'
        color_r = 'plasma'

    # Calcular extent del eje Y según convención
    n_neurons = u.shape[1]
    if orientation_centered:
        # Convención centrada: -90° a 90° (como Fig. 2A)
        # Asume ring de 0° a 180°, lo recentramos
        # NOTA: Con endpoint=False, el último punto está en 176.4°
        # Al centrar: -90° a 86.4° (no exactamente 90°)
        # Pero usamos extent -90 a 90 para visualización clara
        y_min = -90.0
        y_max = 90.0
        y_label = 'Orientation (degrees)'
    else:
        # Convención por defecto: Neuron ID
        y_min = 0
        y_max = n_neurons
        y_label = 'Neuron ID'

    fig = plt.figure(figsize=figsize)
    # Aumentar hspace para evitar solapamiento de títulos/labels
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8],
                  hspace=0.5, wspace=0.3)

    # Panel 1: Heatmap de potenciales de membrana
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(u.T, aspect='auto', cmap=color_u,
                     extent=[result.times_[0], result.times_[-1],
                             y_min, y_max],
                     origin='lower', interpolation='nearest')
    ax1.set_ylabel(y_label)
    ax1.set_title(f'{pop_name} Population - Membrane Potentials u(t)\n'
                  f'Heatmap: {n_neurons} neurons × time')
    plt.colorbar(im1, ax=ax1, label='u (a.u.)')

    # Configurar ticks del eje Y según convención
    if orientation_centered:
        # Mostrar límites explícitos: -90°, 0°, 90°
        ax1.set_yticks([-90, -50, 0, 50, 90])
        ax1.set_yticklabels(['-90°', '-50°', '0°', '50°', '90°'])

    # Quitar xlabel del primer panel para evitar solapamiento
    # (el eje X es el mismo en ambos heatmaps)

    # Panel 2: Heatmap de firing rates
    ax2 = fig.add_subplot(gs[1, :])
    im2 = ax2.imshow(r.T, aspect='auto', cmap=color_r,
                     extent=[result.times_[0], result.times_[-1],
                             y_min, y_max],
                     origin='lower', interpolation='nearest')
    ax2.set_ylabel(y_label)
    ax2.set_title(f'{pop_name} Population - Firing Rates r(t)\n'
                  f'Heatmap: {n_neurons} neurons × time')
    plt.colorbar(im2, ax=ax2, label='r (Hz)')
    ax2.set_xlabel('Time (ms)')

    # Configurar ticks del eje Y según convención
    if orientation_centered:
        # Mostrar límites explícitos: -90°, 0°, 90°
        ax2.set_yticks([-90, -50, 0, 50, 90])
        ax2.set_yticklabels(['-90°', '-50°', '0°', '50°', '90°'])

    # Panel 3: Promedio temporal por neurona (tuning curve espacial)
    ax3 = fig.add_subplot(gs[2, 0])
    r_mean_time = np.mean(r, axis=0)
    if orientation_centered:
        # Usar orientaciones centradas para el eje X
        x_positions = np.linspace(-90, 90, len(r_mean_time))
        ax3.bar(x_positions, r_mean_time, width=180.0/len(r_mean_time),
                alpha=0.7, edgecolor='black', lw=0.5)
        ax3.set_xlabel('Orientation (degrees)')
    else:
        # Usar neuron IDs
        neuron_ids = np.arange(len(r_mean_time))
        ax3.bar(neuron_ids, r_mean_time, alpha=0.7,
                edgecolor='black', lw=0.5)
        ax3.set_xlabel('Neuron ID')
    ax3.set_ylabel('Mean firing rate (Hz)')
    ax3.set_title('Spatial tuning: Average activity per neuron')
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Promedio poblacional en el tiempo
    ax4 = fig.add_subplot(gs[2, 1])
    r_mean_pop = np.mean(r, axis=1)
    ax4.plot(result.times_, r_mean_pop, lw=2, color='darkblue')
    ax4.fill_between(result.times_,
                     r_mean_pop - np.std(r, axis=1),
                     r_mean_pop + np.std(r, axis=1),
                     alpha=0.3)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Population mean ± std (Hz)')
    ax4.set_title('Population dynamics over time')
    ax4.grid(True, alpha=0.3)

    fig.suptitle(f'{pop_name} Population - Complete Activity Profile',
                 fontsize=16, y=0.995)

    plt.tight_layout()
    return fig, (ax1, ax2, ax3, ax4)


def plot_spatial_sampling(result, n_samples=10, figsize=(16, 10),
                          orientation_centered=True):
    """
    Plot spatially sampled neurons across the ring topology.

    Instead of plotting only the first few neurons (which are spatially
    close), this function samples neurons uniformly distributed across the
    entire orientation range, revealing the diversity of
    responses across the population.

    Mathematical foundation:
        - Main paper Eq. 10: Parametric connectivity W_XY(θi, θj)
        - Ring topology: neurons arranged by preferred orientation
        - Samples reveal orientation selectivity patterns

    Parameters
    ----------
    result : NDResult
        Result object from Echeveste2020.run()
    n_samples : int, optional
        Number of neurons to sample (default 10)
    figsize : tuple, optional
        Figure size, default (16, 10)
    orientation_centered : bool, optional
        If True, show orientations as -90° to 90° (default)
        If False, show orientations as 0° to 180°

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : numpy.ndarray
        2x2 array of axes for customization

    Examples
    --------
    >>> ssn = Echeveste2020(N_E=50, N_I=50)
    >>> result = ssn.run(stimulus_contrast=0.5, simulation_time=200.0)
    >>> # Default: Centered orientations -90° to 90°
    >>> fig, axes = plot_spatial_sampling(result, n_samples=10)
    >>> # Alternative: Internal orientations 0° to 180°
    >>> fig, axes = plot_spatial_sampling(result, n_samples=10,
    ...                                   orientation_centered=False)

    Notes
    -----
    Creates 4 panels:
    - Top left: Excitatory membrane potentials
    - Top right: Excitatory firing rates
    - Bottom left: Inhibitory membrane potentials
    - Bottom right: Inhibitory firing rates

    Each panel shows n_samples neurons uniformly distributed across
    the ring topology. By default (orientation_centered=True), shows
    neurons at -90°, -70°, -50°, ..., 70° (centered convention).
    With orientation_centered=False, shows 0°, 20°, 40°, ..., 160°.
    """
    df = result.get_modes()
    n_times = len(result.times_)

    r_e = df['excitatory_firing_rate'].values.reshape(n_times, -1)
    r_i = df['inhibitory_firing_rate'].values.reshape(n_times, -1)
    u_e = df['excitatory_potential'].values.reshape(n_times, -1)
    u_i = df['inhibitory_potential'].values.reshape(n_times, -1)

    # Seleccionar neuronas espaciadas uniformemente
    n_e = r_e.shape[1]
    n_i = r_i.shape[1]
    sample_indices_e = np.linspace(0, n_e-1, n_samples, dtype=int)
    sample_indices_i = np.linspace(0, n_i-1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Potenciales de membrana - Excitatorias
    ax1 = axes[0, 0]
    for idx in sample_indices_e:
        # Calcular orientación (0-180° internamente)
        orientation = idx * 180.0 / n_e
        # Convertir a centrada si es necesario
        if orientation_centered:
            orientation_display = orientation - 90.0
        else:
            orientation_display = orientation
        ax1.plot(result.times_, u_e[:, idx], alpha=0.7,
                 label=f'E{idx} (θ={orientation_display:.1f}°)')
    ax1.set_ylabel('Membrane potential u (a.u.)')
    ax1.set_title(f'Excitatory neurons - Sampled spatially '
                  f'({n_samples}/{n_e} neurons)')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Firing rates - Excitatorias
    ax2 = axes[0, 1]
    for idx in sample_indices_e:
        orientation = idx * 180.0 / n_e
        if orientation_centered:
            orientation_display = orientation - 90.0
        else:
            orientation_display = orientation
        ax2.plot(result.times_, r_e[:, idx], alpha=0.7,
                 label=f'E{idx} (θ={orientation_display:.1f}°)')
    ax2.set_ylabel('Firing rate (Hz)')
    ax2.set_title('Excitatory neurons - Firing rates')
    ax2.legend(loc='best', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Potenciales de membrana - Inhibitorias
    ax3 = axes[1, 0]
    for idx in sample_indices_i:
        orientation = idx * 180.0 / n_i
        if orientation_centered:
            orientation_display = orientation - 90.0
        else:
            orientation_display = orientation
        ax3.plot(result.times_, u_i[:, idx], alpha=0.7,
                 label=f'I{idx} (θ={orientation_display:.1f}°)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Membrane potential u (a.u.)')
    ax3.set_title(f'Inhibitory neurons - Sampled spatially '
                  f'({n_samples}/{n_i} neurons)')
    ax3.legend(loc='best', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Firing rates - Inhibitorias
    ax4 = axes[1, 1]
    for idx in sample_indices_i:
        orientation = idx * 180.0 / n_i
        if orientation_centered:
            orientation_display = orientation - 90.0
        else:
            orientation_display = orientation
        ax4.plot(result.times_, r_i[:, idx], alpha=0.7,
                 label=f'I{idx} (θ={orientation_display:.1f}°)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Firing rate (Hz)')
    ax4.set_title('Inhibitory neurons - Firing rates')
    ax4.legend(loc='best', fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Spatial Sampling: Neurons distributed across '
                 'ring topology',
                 fontsize=14, y=0.995)
    plt.tight_layout()

    return fig, axes
