"""
JAX-based training utilities for Echeveste2020 SSN model.

Este módulo implementa el método de Assumed Density Filtering (ADF)
y las funciones de costo para entrenar la Stabilized Supralinear Network
siguiendo Echeveste et al. (2020).

------------------------
PAPER PRINCIPAL: Echeveste et al. (2020) Nature Neuroscience

1. Modelo SSN (Ecuación 8 del paper):
   τ_α · du_α/dt = -u_α + Σ_β W_αβ r_β(u_β) + h_α + η_α
   donde:
   - u_α: potencial de membrana de neurona α
   - r_α = k[u_α]₊² : activación supralineal (Eq. 9)
   - W_αβ: matriz de conectividad
   - h_α: input externo (del GSM)
   - η_α: ruido con covarianza Σ_η (Eq. 11)

2. ADF (Assumed Density Filtering) - Ecuaciones 17-20:
   En lugar de simular con ruido, evolucionamos los MOMENTOS:
   - μ: vector de medias E[u_α]
   - Σ: matriz de covarianza Cov[u_α, u_β]

   Esto permite entrenar determinísticamente sin sampleo.

3. Función de Costo (Ecuación 25):
   L = λ_μ·||μ_SSN - μ_GSM||² + λ_σ·||var_SSN - var_GSM||²
       + λ_Σ·||Σ_SSN - Σ_GSM||²_F + λ_slow·slowness

   El SSN aprende a replicar las estadísticas del posterior GSM.

CÓDIGO ORIGINAL: ssn_inference_optimizer/objective.ml
- Lines 254-268: Nonlinear moments (nu, gamma)
- Lines 314-332: ADF evolution equations
- Lines 377-408: Cost function computation
- Lines 153-180: Parameter packing/unpacking

Referencias:
-----------
Echeveste, R., Aitchison, L., Hennequin, G., & Lengyel, M. (2020).
Cortical-like dynamics in recurrent circuits optimized for
sampling-based probabilistic inference. Nature Neuroscience, 23(9), 1138-1149.

Código original OCaml:
ssn_inference_optimizer/objective.ml
ssn_inference_optimizer/train.ml
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats as jax_stats


# =============================================================================
# Nonlinear moment functions
# =============================================================================

@jax.jit
def compute_nonlinear_moments(mu, sigma, k=0.3):
    """
    Compute nonlinear moments for supralinear activation.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 9: r_α = k · [u_α]₊^n donde n=2 (supralinear cuadrática)

    Problema: Para ADF necesitamos E[r_α] y ∂E[r_α]/∂μ_α, pero r es NO LINEAL.

    Solución (Supplementary Material, Eqs. S15-S17):
    Si u_α ~ N(μ_α, σ_α²), entonces usando integración gaussiana:

    1. E[[u]₊] = μ·Ψ(μ/σ) + σ·φ(μ/σ)  [rectificación lineal]
       donde φ = PDF gaussiana, Ψ = CDF gaussiana

    2. Para n=2 (cuadrática):
       E[[u]₊²] = μ·E[[u]₊] + σ²·Ψ(μ/σ)

    3. Por lo tanto:
       ν = E[r] = k · E[[u]₊²] = k·(μ·nu1 + σ²·Ψ)
       donde nu1 = E[[u]₊]

    4. Para ADF también necesitamos γ = ∂E[r]/∂μ:
       γ = ∂ν/∂μ = k · ∂E[[u]₊²]/∂μ = 2k · nu1

    CÓDIGO ORIGINAL: objective.ml lines 254-268
    - Line 261: nu1 = mu * psi + sigma * phi
    - Line 266: nu_fun = k * (mu * nu1 + sigma² * psi)
    - Line 267: gamma_fun = 2k * nu1

    Parameters
    ----------
    mu : jax.Array
        Mean vector of neuron inputs E[u_α], shape (n,)
    sigma : jax.Array
        Covariance matrix Cov[u_α, u_β], shape (n, n)
    k : float
        Scaling constant (Suppl. Table S1: k=0.3)

    Returns
    -------
    nu : jax.Array
        ν = E[r_α] = primer momento no lineal, shape (n,)
    gamma : jax.Array
        γ = ∂E[r_α]/∂μ_α = derivada del momento, shape (n,)

    Notes
    -----
    Estos momentos son CRÍTICOS para ADF porque permiten
    evolucionar las estadísticas de la activación NO LINEAL
    usando solo los momentos gaussianos (μ, Σ).

    Sin esto, tendríamos que hacer sampling estocástico
    (mucho más lento y ruidoso para optimización).
    """
    # PASO 1: Extraer varianzas individuales σ_α²
    # Para cada neurona α, necesitamos su varianza individual (diagonal de Σ)
    sigma2 = jnp.diag(sigma)  # σ_α² para cada α
    sigma_std = jnp.sqrt(sigma2)  # σ_α = sqrt(σ_α²)

    # PASO 2: Calcular ratio μ/σ (argumento de las funciones gaussianas)
    # Manejo especial para evitar división por cero cuando σ ≈ 0
    # Si σ muy pequeño → neurona casi determinística → usar ratio = 0
    ratio = jnp.where(
        sigma_std > 1e-10,  # Si σ > threshold
        mu / sigma_std,      # ratio normal
        0.0                  # Si σ ≈ 0, usar 0 (límite cuando σ→0)
    )

    # PASO 3: Evaluar funciones gaussianas estándar
    # φ(x) = (1/√(2π)) · exp(-x²/2) [PDF gaussiana estándar]
    # Ψ(x) = ∫_{-∞}^x φ(t)dt [CDF gaussiana estándar]
    phi = jax_stats.norm.pdf(ratio)  # φ(μ_α/σ_α) para cada α
    psi = jax_stats.norm.cdf(ratio)  # Ψ(μ_α/σ_α) para cada α

    # PASO 4: Calcular E[[u]₊] = primer momento de ReLU
    # Fórmula (Suppl. Eq. S15): E[[u]₊] = μ·Ψ(μ/σ) + σ·φ(μ/σ)
    # Interpretación:
    # - μ·Ψ: contribución de la parte positiva desplazada por la media
    # - σ·φ: contribución gaussiana en el borde (threshold en 0)
    # objective.ml line 261
    nu1 = mu * psi + sigma_std * phi

    # PASO 5: Calcular ν = E[r] = E[k·[u]₊²] (n=2 hard-coded)
    # Fórmula (Suppl. Eq. S17): E[[u]₊²] = μ·E[[u]₊] + σ²·Ψ(μ/σ)
    # Luego multiplicamos por k (factor de escala del SSN)
    # objective.ml line 266
    nu = k * (mu * nu1 + sigma2 * psi)

    # PASO 6: Calcular γ = ∂E[r]/∂μ (derivada del momento respecto a la media)
    # Para n=2: γ = ∂(k·E[[u]₊²])/∂μ = 2k·E[[u]₊] = 2k·nu1
    # Este término es CRUCIAL para la evolución de Σ en ADF
    # (aparece en la matriz Jacobiana J)
    # objective.ml line 267
    gamma = 2.0 * k * nu1

    return nu, gamma  # Retornar ambos momentos no lineales


# =============================================================================
# ADF moment evolution
# =============================================================================

@jax.jit
def compute_jacobian_matrix(w, gamma, inv_taus):
    """
    Compute Jacobian matrix J for ADF evolution.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 8: τ·du/dt = -u + W·r(u) + h + η

    Para ADF, necesitamos la LINEALIZACIÓN de la dinámica alrededor de μ.

    Derivando respecto a u:
    τ·dμ/dt = -μ + W·E[r(u)] + h

    El Jacobiano controla cómo pequeñas perturbaciones δu afectan la dinámica:
    τ·d(δu)/dt = -δu + W·(∂E[r]/∂u)·δu

    Dividiendo por τ:
    d(δu)/dt = (1/τ)·(-δu + W·diag(γ)·δu)
             = (1/τ)·(W·diag(γ) - I)·δu

    Por lo tanto:
    J = diag(1/τ)·(W·diag(γ) - I)

    donde γ = ∂E[r]/∂μ es la derivada del momento no lineal (computed before).

    Esta matriz J es CRÍTICA porque:
    1. Controla la evolución de Σ (Eq. 18: dΣ/dt ∝ J·Σ + Σ·J^T)
    2. Determina la estabilidad de la red (eigenvalues de J)
    3. Aparece en la propagación de incertidumbre

    CÓDIGO ORIGINAL: objective.ml lines 306-309
    - Line 308: w_eff = W * diag(γ) - I
    - Line 309: J = diag(1/τ) * w_eff

    Parameters
    ----------
    w : jax.Array
        Connectivity matrix W_αβ, shape (n, n)
    gamma : jax.Array
        Nonlinear moment derivatives γ_α = ∂E[r_α]/∂μ_α, shape (n,)
    inv_taus : jax.Array
        Inverse time constants 1/τ_α, shape (n,)

    Returns
    -------
    j_mat : jax.Array
        Jacobian matrix J, shape (n, n)

    Notes
    -----
    La matriz efectiva W_eff = W·diag(γ) - I representa:
    - W·diag(γ): conectividad ponderada por ganancia no lineal
    - -I: término de leak (decay pasivo)

    Luego diag(1/τ) escala todo por las constantes de tiempo.
    """
    # PASO 1: Construir matriz efectiva W_eff = W·diag(γ) - I
    # Broadcasting: w * gamma[None, :] multiplica cada columna j de w por γ_j
    # Esto equivale a W·diag(γ) en notación matricial
    # objective.ml line 308
    w_eff = w * gamma[None, :] - jnp.eye(w.shape[0])

    # PASO 2: Escalar por constantes de tiempo J = diag(1/τ)·W_eff
    # Broadcasting: inv_taus[:, None] multiplica cada fila i por 1/τ_i
    # Esto equivale a multiplicación diag(1/τ)·W_eff
    # objective.ml line 309
    j_mat = inv_taus[:, None] * w_eff

    return j_mat


@jax.jit
def dmu_dt(w, mu, nu, h, inv_taus):
    """
    Compute mean evolution dμ/dt.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 8: τ_α·du_α/dt = -u_α + Σ_β W_αβ·r_β(u_β) + h_α + η_α

    Tomando valor esperado E[·] en ambos lados:
    τ_α·dE[u_α]/dt = -E[u_α] + Σ_β W_αβ·E[r_β(u_β)] + E[h_α] + E[η_α]

    Como:
    - μ_α = E[u_α] (media del potencial)
    - ν_β = E[r_β] (media de activación, ver compute_nonlinear_moments)
    - E[h_α] = h_α (input externo es determinístico)
    - E[η_α] = 0 (ruido tiene media cero)

    Obtenemos la ECUACIÓN 17 del paper:
    τ_α·dμ_α/dt = -μ_α + Σ_β W_αβ·ν_β + h_α

    Reescribiendo en forma vectorial:
    τ·dμ/dt = -μ + W·ν + h

    Dividiendo por τ:
    dμ/dt = (1/τ)·(-μ + W·ν + h)

    Esta ecuación describe cómo evoluciona la MEDIA de los potenciales
    bajo la aproximación ADF (sin ruido, solo momentos determinísticos).

    CÓDIGO ORIGINAL: objective.ml lines 313-316
    - Line 315: z = (h - mu) + W·ν
    - Line 316: dμ/dt = (1/τ)·z

    Parameters
    ----------
    w : jax.Array
        Connectivity matrix W_αβ, shape (n, n)
    mu : jax.Array
        Current mean vector μ_α = E[u_α], shape (n,)
    nu : jax.Array
        Nonlinear moment ν_α = E[r_α], shape (n,)
    h : jax.Array
        External input h_α (from GSM), shape (n,)
    inv_taus : jax.Array
        Inverse time constants 1/τ_α, shape (n,)

    Returns
    -------
    dmu : jax.Array
        Time derivative dμ/dt, shape (n,)

    Notes
    -----
    Esta función implementa la evolución DETERMINÍSTICA de la media.
    La varianza Σ evoluciona por separado (ver new_sigma).

    El término W·ν es la suma de inputs recurrentes:
    (W·ν)_α = Σ_β W_αβ·ν_β = input recurrente a neurona α
    """
    # PASO 1: Calcular término combinado z = (h - μ) + W·ν
    # Interpretación:
    # - (h - μ): diferencia entre input externo y estado actual (driving force)
    # - W·ν: input recurrente promedio desde otras neuronas
    # objective.ml line 315
    z = (h - mu) + jnp.dot(w, nu)

    # PASO 2: Escalar por constante de tiempo 1/τ
    # Esto da la tasa de cambio real considerando la dinámica temporal
    # objective.ml line 316
    dmu = inv_taus * z

    return dmu


@jax.jit
def new_sigma_star(j_mat, sigma_eta, sigma_star, dt,
                   tau_eta, inv_taus):
    """
    Update Σ* (auxiliary covariance for noise autocorrelation).

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 11: El ruido η_α(t) tiene correlación temporal:
    E[η_α(t)·η_β(t')] = (Σ_η)_αβ · exp(-|t-t'|/τ_η)

    Problema: En ADF necesitamos rastrear cómo el ruido CORRELACIONADO
    temporalmente afecta la covarianza de los potenciales.

    Solución (Paper Eq. 19 y Supplementary Material):
    Se introduce una COVARIANZA AUXILIAR Σ* que rastrea la
    "memoria" del ruido en la dinámica.

    ECUACIÓN 19 del paper:
    dΣ*/dt = -(1/τ_η)·Σ* + J·Σ* + diag(1/τ)·Σ_η

    Esta ecuación tiene tres términos:
    1. -(1/τ_η)·Σ*: decay de la correlación temporal del ruido
    2. J·Σ*: propagación de la correlación a través de la dinámica
    3. diag(1/τ)·Σ_η: inyección de nueva correlación desde el ruido

    Integrando con método de Euler (discretización temporal):
    Σ*(t+dt) ≈ Σ*(t) + dt·dΣ*/dt

    El código original usa una discretización mejorada (ver objetivo.ml):
    Σ*(t+dt) = ε₁·(Σ* + dt·J·Σ*) + ε₂·diag(1/τ)·Σ_η

    donde:
    - ε₁ = 1 - dt/τ_η (factor de decay del ruido)
    - ε₂ = dt·(1 + dt/τ_η) (factor de inyección)

    CÓDIGO ORIGINAL: objective.ml lines 318-322
    - Lines 127-128: definición de ε₁, ε₂
    - Line 320: tmp1 = Σ* + dt·J·Σ*
    - Line 321: tmp2 = diag(1/τ)·Σ_η
    - Line 322: Σ*_new = ε₁·tmp1 + ε₂·tmp2

    Parameters
    ----------
    j_mat : jax.Array
        Jacobian matrix J, shape (n, n)
    sigma_eta : jax.Array
        Noise covariance matrix Σ_η, shape (n, n)
    sigma_star : jax.Array
        Current auxiliary covariance Σ*, shape (n, n)
    dt : float
        Time step (seconds)
    tau_eta : float
        Noise autocorrelation time constant τ_η (seconds)
    inv_taus : jax.Array
        Inverse neuronal time constants 1/τ_α, shape (n,)

    Returns
    -------
    sigma_star_new : jax.Array
        Updated auxiliary covariance Σ*(t+dt), shape (n, n)

    Notes
    -----
    Σ* es CRUCIAL para modelar ruido con correlación temporal.
    Si τ_η → 0 (ruido blanco), entonces Σ* → 0 y se simplifica.

    La interacción entre τ_η (timescale del ruido) y τ (timescale
    de las neuronas) determina cuánta memoria retiene la red.
    """
    # PASO 1: Calcular coeficientes temporales
    # ε₁ controla cuánto "olvida" la red (decay)
    # ε₂ controla cuánto "recuerda" (inyección de ruido)
    # objective.ml lines 127-128
    eps1 = 1.0 - dt / tau_eta  # ε₁ = 1 - dt/τ_η
    eps2 = dt * (1.0 + dt / tau_eta)  # ε₂ = dt·(1 + dt/τ_η)

    # PASO 2: Propagar Σ* a través de la dinámica
    # tmp1 = Σ* + dt·J·Σ* (Euler step con Jacobiano)
    # J·Σ* describe cómo la dinámica de la red propaga la correlación
    # objective.ml line 320
    j_mat_sigma_star = jnp.dot(j_mat, sigma_star)
    tmp1 = sigma_star + dt * j_mat_sigma_star

    # PASO 3: Preparar inyección de ruido
    # tmp2 = diag(1/τ)·Σ_η
    # El factor 1/τ escala el ruido por las constantes de tiempo neuronales
    # (neuronas rápidas tienen menos tiempo para acumular ruido)
    # objective.ml line 321
    tmp2 = inv_taus[:, None] * sigma_eta

    # PASO 4: Combinar decay y inyección
    # ε₁·tmp1: estado propagado con decay
    # ε₂·tmp2: nueva correlación inyectada desde Σ_η
    # objective.ml line 322
    sigma_star_new = eps1 * tmp1 + eps2 * tmp2

    return sigma_star_new


@jax.jit
def new_sigma(j_mat, sigma_star, sigma, dt, inv_taus):
    """
    Update Σ (covariance matrix).

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 8: τ·du/dt = -u + W·r(u) + h + η

    Queremos evolucionar Σ_αβ = Cov[u_α, u_β] = E[(u_α - μ_α)·(u_β - μ_β)]

    Derivando Σ_αβ respecto al tiempo y usando la dinámica de u:
    dΣ/dt = E[(du/dt)·(u - μ)^T] + E[(u - μ)·(du/dt)^T]

    Sustituyendo la ecuación de u y LINEALIZANDO alrededor de μ:
    du/dt ≈ (1/τ)·(-u + W·E[r] + J·(u - μ) + h + η)

    donde J = diag(1/τ)·(W·diag(γ) - I) es el Jacobiano.

    Después de algebra (ver Supplementary Material, derivación completa):

    ECUACIÓN 18 del paper:
    dΣ/dt = J·Σ + Σ·J^T + 2·diag(1/τ)·Σ* + (términos de orden dt²)

    Integrando con método de Euler mejorado (objetivo.ml):
    Σ(t+dt) = P·Σ·P^T + B + B^T

    donde:
    - P = I + dt·J (propagador lineal de primer orden)
    - B = dt·diag(1/τ)·Σ* + dt²·J·Σ* (término de ruido + corrección)

    Interpretación física:
    1. P·Σ·P^T: Propagación de la covarianza existente a través de la dinámica
    2. B + B^T: Inyección simétrica de nueva varianza desde el ruido

    CÓDIGO ORIGINAL: objective.ml lines 325-330
    - Line 327-328: B = dt·diag(1/τ)·Σ* + dt²·J·Σ*
    - Line 329: P = I + dt·J
    - Line 330: Σ_new = P·Σ·P^T + B + B^T

    Parameters
    ----------
    j_mat : jax.Array
        Jacobian matrix J, shape (n, n)
    sigma_star : jax.Array
        Auxiliary covariance Σ* (noise memory), shape (n, n)
    sigma : jax.Array
        Current covariance matrix Σ, shape (n, n)
    dt : float
        Time step (seconds)
    inv_taus : jax.Array
        Inverse time constants 1/τ_α, shape (n,)

    Returns
    -------
    sigma_new : jax.Array
        Updated covariance matrix Σ(t+dt), shape (n, n)

    Notes
    -----
    Esta es la ecuación MÁS CRÍTICA de ADF porque:
    1. Captura cómo la variabilidad se propaga en el tiempo
    2. Incluye efectos del ruido correlacionado (vía Σ*)
    3. La simetría (B + B^T) garantiza que Σ sea simétrica
    4. La forma P·Σ·P^T preserva positividad (si dt pequeño)

    Si J es estable (eigenvalues < 0), entonces Σ converge a
    un estado estacionario.
    Si J tiene eigenvalues > 0, la red es inestable y Σ explota.
    """
    # PRE-CÓMPUTO: Calcular J·Σ* una sola vez (usado en B)
    # Este producto matricial describe cómo la dinámica transforma
    # la memoria del ruido
    j_mat_sigma_star = jnp.dot(j_mat, sigma_star)

    # PASO 1: Construir matriz B (inyección de varianza desde ruido)
    # B tiene dos términos:
    # 1. dt·diag(1/τ)·Σ*: contribución directa del ruido (orden dt)
    # 2. dt²·J·Σ*: corrección de segundo orden por dinámica (orden dt²)
    # objective.ml lines 327-328
    b = dt * (inv_taus[:, None] * sigma_star)  # Término orden dt
    b = b + (dt * dt) * j_mat_sigma_star  # Término orden dt²

    # PASO 2: Construir propagador P = I + dt·J
    # P describe cómo la linealización de la dinámica propaga perturbaciones
    # Para dt pequeño, P ≈ exp(dt·J) (exponencial matricial aproximada)
    # objective.ml line 329
    prop = jnp.eye(j_mat.shape[0]) + dt * j_mat

    # PASO 3: Propagar Σ y agregar ruido
    # P·Σ·P^T: transforma Σ según la dinámica linealizada
    # B + B^T: agrega varianza del ruido simétricamente
    # La simetría es crítica: Cov[X,Y] = Cov[Y,X]
    # objective.ml line 330
    sigma_new = jnp.dot(prop, jnp.dot(sigma, prop.T))  # Propagación
    sigma_new = sigma_new + b + b.T  # Inyección de ruido (simétrica)

    return sigma_new


@jax.jit
def evolve_moments_single_step(
    mu, sigma, sigma_star, w, h, sigma_eta,
    dt, tau_eta, inv_taus, k=0.3
):
    """
    Evolve moments (μ, Σ, Σ*) for a single time step using ADF.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Esta función INTEGRA todas las ecuaciones de ADF para un paso temporal.

    Paper Eqs. 17-20 (sistema acoplado de ecuaciones diferenciales):

    1. Eq. 17: dμ/dt = (1/τ)·(-μ + W·ν + h)
       → Evolución de la MEDIA de los potenciales

    2. Eq. 18: dΣ/dt = J·Σ + Σ·J^T + 2·diag(1/τ)·Σ*
       → Evolución de la COVARIANZA de los potenciales

    3. Eq. 19: dΣ*/dt = -(1/τ_η)·Σ* + J·Σ* + diag(1/τ)·Σ_η
       → Evolución de la MEMORIA del ruido

    donde:
    - ν, γ son momentos no lineales de r = k·[u]₊² (Eq. 9)
    - J = diag(1/τ)·(W·diag(γ) - I) es el Jacobiano

    PIPELINE COMPUTACIONAL:
    -----------------------
    Para cada paso temporal t → t+dt:

    1. Calcular momentos no lineales (ν, γ) dado μ, Σ actuales
       → Necesitamos ν para dμ/dt
       → Necesitamos γ para construir J

    2. Construir Jacobiano J dado γ
       → Necesitamos J para dΣ/dt y dΣ*/dt

    3. Integrar μ usando método de Euler: μ(t+dt) = μ(t) + dt·dμ/dt
       → Actualiza la media

    4. Integrar Σ usando esquema mejorado (ver new_sigma)
       → Actualiza la covarianza

    5. Integrar Σ* usando esquema mejorado (ver new_sigma_star)
       → Actualiza la memoria del ruido

    ORDEN DE EJECUCIÓN:
    Este orden es CRÍTICO porque:
    - ν, γ dependen de μ, Σ del paso ANTERIOR
    - J depende de γ del paso ANTERIOR
    - μ_new, Σ_new, Σ*_new se calculan en PARALELO usando valores antiguos

    Esto corresponde a un método de Euler EXPLÍCITO (forward Euler).

    CÓDIGO ORIGINAL: objective.ml lines 349-372
    - Lines 355-357: cálculo de ν, γ
    - Lines 358-359: cálculo de J
    - Line 361: actualización de μ
    - Line 362: actualización de Σ
    - Line 363: actualización de Σ*

    Parameters
    ----------
    mu : jax.Array
        Current mean vector μ(t), shape (n,)
    sigma : jax.Array
        Current covariance matrix Σ(t), shape (n, n)
    sigma_star : jax.Array
        Current auxiliary covariance Σ*(t), shape (n, n)
    w : jax.Array
        Connectivity matrix W (fixed), shape (n, n)
    h : jax.Array
        External input h (from GSM), shape (n,)
    sigma_eta : jax.Array
        Noise covariance Σ_η (fixed), shape (n, n)
    dt : float
        Time step Δt (seconds)
    tau_eta : float
        Noise autocorrelation time τ_η (seconds)
    inv_taus : jax.Array
        Inverse time constants 1/τ_α, shape (n,)
    k : float
        Supralinear scaling constant (default 0.3)

    Returns
    -------
    mu_new : jax.Array
        Updated mean μ(t+dt), shape (n,)
    sigma_new : jax.Array
        Updated covariance Σ(t+dt), shape (n, n)
    sigma_star_new : jax.Array
        Updated auxiliary covariance Σ*(t+dt), shape (n, n)

    Notes
    -----
    Esta función es el CORAZÓN de ADF:
    - Se llama repetidamente en un loop temporal
    - Cada llamada avanza el sistema un paso dt
    - La estabilidad numérica requiere dt suficientemente pequeño
      (típicamente dt << min(τ_e, τ_i, τ_η))

    El método es determinístico (no hay sampling) pero captura
    efectos del ruido estocástico mediante Σ y Σ*.
    """
    # PASO 1: Calcular momentos no lineales (ν, γ) para el estado actual
    # Estos dependen solo de (μ, Σ) del tiempo t
    # ν = E[r] = E[k·[u]₊²] (valor esperado de la activación)
    # γ = ∂E[r]/∂μ (sensibilidad de ν a cambios en μ)
    nu, gamma = compute_nonlinear_moments(mu, sigma, k)

    # PASO 2: Construir matriz Jacobiana J
    # J = diag(1/τ)·(W·diag(γ) - I)
    # Esta matriz controla la linealización de la dinámica
    j_mat = compute_jacobian_matrix(w, gamma, inv_taus)

    # PASO 3: Evolucionar media μ (Eq. 17)
    # μ(t+dt) = μ(t) + dt·dμ/dt
    # donde dμ/dt = (1/τ)·(-μ + W·ν + h)
    mu_new = mu + dt * dmu_dt(w, mu, nu, h, inv_taus)

    # PASO 4: Evolucionar covarianza Σ (Eq. 18)
    # Usa esquema mejorado: Σ(t+dt) = P·Σ·P^T + B + B^T
    # Incluye propagación de Σ y efectos del ruido vía Σ*
    sigma_new = new_sigma(
        j_mat, sigma_star, sigma, dt, inv_taus
    )

    # PASO 5: Evolucionar covarianza auxiliar Σ* (Eq. 19)
    # Σ*(t+dt) = ε₁·(Σ* + dt·J·Σ*) + ε₂·diag(1/τ)·Σ_η
    # Rastrea la memoria de la correlación temporal del ruido
    sigma_star_new = new_sigma_star(
        j_mat, sigma_eta, sigma_star, dt,
        tau_eta, inv_taus
    )

    return mu_new, sigma_new, sigma_star_new


def evolve_moments(
    w, h, sigma_eta, inv_taus, dt, tau_eta,
    t_max, k=0.3, mu_init=None, sigma_init=None
):
    """
    Evolve moments from t=0 to t=t_max using ADF.

    Evoluciona los momentos (μ, Σ) desde condiciones iniciales
    hasta alcanzar el tiempo final t_max.

    Based on objective.ml lines 348-372.

    Parameters
    ----------
    w : jax.Array or np.ndarray
        Connectivity matrix, shape (n, n)
    h : jax.Array or np.ndarray
        External input, shape (n,)
    sigma_eta : jax.Array or np.ndarray
        Noise covariance, shape (n, n)
    inv_taus : jax.Array or np.ndarray
        Inverse time constants, shape (n,)
    dt : float
        Time step
    tau_eta : float
        Noise autocorrelation time constant
    t_max : float
        Maximum integration time
    k : float
        Supralinear scaling constant
    mu_init : jax.Array or None
        Initial mean (if None, uses zeros)
    sigma_init : jax.Array or None
        Initial covariance (if None, uses 4*I)

    Returns
    -------
    mu_final : jax.Array
        Final mean vector, shape (n,)
    sigma_final : jax.Array
        Final covariance matrix, shape (n, n)
    sigma_star_final : jax.Array
        Final auxiliary covariance, shape (n, n)

    Notes
    -----
    Implementa el loop temporal de objective.ml lines 351-372.
    """
    # Convertir a JAX arrays si es necesario
    w = jnp.asarray(w)
    h = jnp.asarray(h)
    sigma_eta = jnp.asarray(sigma_eta)
    inv_taus = jnp.asarray(inv_taus)

    n = w.shape[0]
    n_time_bins = int(t_max / dt)

    # Condiciones iniciales (objective.ml lines 367-371)
    if mu_init is None:
        mu = jnp.zeros(n)
    else:
        mu = jnp.asarray(mu_init)

    if sigma_init is None:
        sigma = 4.0 * jnp.eye(n)
    else:
        sigma = jnp.asarray(sigma_init)

    # Σ* inicial (objective.ml line 371)
    sigma_star = 4.0 * jnp.eye(n)

    # Loop temporal (objective.ml lines 351-364)
    for t in range(n_time_bins):
        mu, sigma, sigma_star = evolve_moments_single_step(
            mu, sigma, sigma_star, w, h, sigma_eta,
            dt, tau_eta, inv_taus, k
        )

    return mu, sigma, sigma_star


# =============================================================================
# Cost functions
# =============================================================================

@jax.jit
def compute_cost_components(
    mu, sigma, target_mu, target_sigma,
    lambda_mean=1.0, lambda_var=1.0, lambda_cov=1.0
):
    """
    Compute cost function components (mean, variance, covariance).

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Paper Eq. 25: Función de costo para entrenar el SSN

    L = λ_μ·||μ_SSN - μ_GSM||² + λ_σ²·||var_SSN - var_GSM||²
        + λ_Σ·||Σ_SSN - Σ_GSM||²_F + λ_slow·slowness

    Objetivo: El SSN debe REPLICAR las estadísticas del posterior del GSM

    ¿Por qué estos tres términos?

    1. MEAN MATCHING (λ_μ·||μ_SSN - μ_GSM||²):
       - Fuerza que la actividad PROMEDIO de la red coincida con el posterior
       - μ_GSM = E[y|x,z] es la estimación MAP del GSM
       - El SSN debe representar esta estimación en su actividad media
       - Norma L2 (suma de cuadrados) penaliza desviaciones grandes

    2. VARIANCE MATCHING (λ_σ²·||var_SSN - var_GSM||²):
       - Fuerza que la VARIABILIDAD INDIVIDUAL de cada neurona coincida
       - var_α = (Σ)_αα es la varianza de la neurona α (diagonal de Σ)
       - Importante para capturar la INCERTIDUMBRE del posterior
       - Si GSM está seguro (var pequeña) → SSN debe tener poca variabilidad
       - Si GSM está inseguro (var grande) → SSN debe tener alta variabilidad

    3. COVARIANCE MATCHING (λ_Σ·||Σ_SSN - Σ_GSM||²_F):
       - Fuerza que las CORRELACIONES entre neuronas coincidan
       - ||·||_F es la norma de Frobenius: ||A||²_F = Σ_ij A²_ij
       - Las correlaciones capturan la ESTRUCTURA del posterior
       - Ejemplo: si orientaciones cercanas están correlacionadas en GSM,
         neuronas cercanas en SSN también deben estarlo
       - Este término es CRÍTICO para sampling-based inference

    ¿Por qué ponderaciones λ separadas?
    - Diferentes términos tienen diferentes escalas (μ ~ O(1), Σ ~ O(0.1))
    - λ's permiten balancear la importancia relativa
    - Valores típicos (Suppl. Table S1): λ_μ = 1.0, λ_σ² = 1.0, λ_Σ = 0.1

    IMPLEMENTACIÓN:
    Solo usamos neuronas EXCITATORIAS (primeras m) porque:
    - Las neuronas inhibitorias son "hidden" (no observables)
    - El GSM solo tiene targets para orientaciones (E neurons)
    - Las I neurons se ajustan implícitamente vía conectividad

    CÓDIGO ORIGINAL: objective.ml lines 395-398
    - Line 395: cost_mean = λ_μ·Σ_α(μ_target - μ)²
    - Line 396: cost_var = λ_σ²·Σ_α(var_target - var)²
    - Lines 397-398: cost_cov = λ_Σ·Σ_αβ(Σ_target - Σ)²_αβ

    Parameters
    ----------
    mu : jax.Array
        Network mean (excitatory neurons only), shape (m,)
    sigma : jax.Array
        Network covariance (excitatory neurons only), shape (m, m)
    target_mu : jax.Array
        Target mean from GSM posterior E[y|x,z], shape (m,)
    target_sigma : jax.Array
        Target covariance from GSM posterior Cov[y|x,z], shape (m, m)
    lambda_mean : float
        Weight λ_μ for mean matching term
    lambda_var : float
        Weight λ_σ² for variance matching term
    lambda_cov : float
        Weight λ_Σ for covariance matching term

    Returns
    -------
    cost_mean : float
        Mean matching cost λ_μ·||μ - μ_target||²
    cost_var : float
        Variance matching cost λ_σ²·||var - var_target||²
    cost_cov : float
        Covariance matching cost λ_Σ·||Σ - Σ_target||²_F

    Notes
    -----
    Minimizar estos tres términos simultáneamente hace que
    el SSN aprenda a:
    1. Estimar correctamente (mean matching)
    2. Representar incertidumbre (variance matching)
    3. Capturar estructura (covariance matching)

    Esto permite sampling-based probabilistic inference:
    la red genera samples ~ P(y|x,z) mediante ruido interno.
    """
    # PASO 1: Mean matching - diferencia en medias
    # ||μ_target - μ||² = Σ_α (μ_target,α - μ_α)²
    # Penaliza que la actividad promedio no coincida con el posterior
    # objective.ml line 395
    cost_mean = lambda_mean * jnp.sum((target_mu - mu) ** 2)

    # PASO 2: Variance matching - diferencia en varianzas individuales
    # Extraemos solo la diagonal de Σ (varianzas individuales)
    # ||var_target - var||² = Σ_α (Σ_target,αα - Σ_αα)²
    # Penaliza que la variabilidad individual no coincida
    # objective.ml line 396
    cost_var = lambda_var * jnp.sum(
        (jnp.diag(target_sigma) - jnp.diag(sigma)) ** 2
    )

    # PASO 3: Covariance matching - diferencia en toda la matriz
    # Norma de Frobenius: ||A||²_F = Σ_αβ A²_αβ
    # jnp.sum(A**2) calcula Σ_ij A²_ij (equivalente a ||A||²_F)
    # Penaliza que las correlaciones no coincidan
    # objective.ml lines 397-398
    cost_cov = lambda_cov * jnp.sum((target_sigma - sigma) ** 2)

    return cost_mean, cost_var, cost_cov


# =============================================================================
# Temporal weighting utilities
# =============================================================================

def create_temporal_weighting(n_time_bins, min_time_bins, zero_up_to=None):
    """
    Create temporal weighting vector for cost evaluation.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    El temporal weighting permite controlar qué timesteps contribuyen
    al costo durante el entrenamiento. Esto es CRUCIAL para:

    1. TRANSIENT FILTERING: Ignorar timesteps tempranos donde la red
       todavía está estabilizándose.

    2. TEMPORAL ANNEALING: Durante entrenamiento con ADAM, permite
       expandir progresivamente la ventana temporal evaluada.

    CÓDIGO ORIGINAL: objective.ml lines 374-375, train.ml lines 240-243
    - objective.ml:374: step_fun define pesos 0/1
    - train.ml:240-243: update_temporal_weighting actualiza ventana

    PAPER: Echeveste et al. (2020) Methods, página 18:
    "the beginning of the averaging time window, T_min in Eqs. 26-28,
    was systematically changed ('annealed') from T_min = 0 ms to
    T_max - 50 ms"

    Esta estrategia permite que:
    - Inicialmente: La red aprende el comportamiento asintótico
      (últimos 50ms) que es más estable y fácil de optimizar
    - Gradualmente: Se incluyen timesteps más tempranos, forzando
      a la red a refinar su dinámica transitoria

    Parameters
    ----------
    n_time_bins : int
        Total number of time bins
    min_time_bins : int
        Minimum number of bins at the end that always have weight 1.0
    zero_up_to : int or None
        Number of initial bins with weight 0.0
        If None, uses (n_time_bins - min_time_bins), meaning only
        the last min_time_bins have weight 1.0

    Returns
    -------
    temporal_weighting : np.ndarray, shape (n_time_bins,)
        Weight vector: 0.0 before zero_up_to, 1.0 after

    Notes
    -----
    Ref: objective.ml line 374-375
    let step_fun i = if i<(n_time_bins - min_time_bins) then 0. else 1.
    let temporal_weighting = Vec.init n_time_bins (fun i -> step_fun i)
    """
    import numpy as np

    if zero_up_to is None:
        # Por defecto: solo últimos min_time_bins tienen peso 1.0
        zero_up_to = n_time_bins - min_time_bins

    # Crear vector de pesos
    # Ref: objective.ml line 374
    temporal_weighting = np.where(
        np.arange(n_time_bins) < zero_up_to,
        0.0,
        1.0
    )

    return temporal_weighting


@jax.jit
def compute_slowness_cost(w, mu, sigma, inv_taus, dt, t_max_slow,
                          k, lambda_slow):
    """
    Compute slowness penalty cost.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    El slowness penalty penaliza cambios rápidos en la covarianza Σ
    durante la dinámica transitoria del sistema. Esto ayuda a prevenir
    oscilaciones no deseadas y fuerza una dinámica más suave.

    Paper Echeveste et al. (2020): No mencionado explícitamente en paper
    principal, pero usado en código original para estabilizar training.

    ECUACIÓN DE SLOWNESS:
    ---------------------
    La covarianza Σ evoluciona (aproximadamente) según:
        dΣ/dt ≈ Σ·J^T + J·Σ

    donde J es el Jacobiano del sistema linealizado.

    El costo de slowness mide cuánto cambia Σ durante la dinámica:

    L_slow = λ_slow · Σ_{t=0}^{T_slow} ||Σ_norm(t)||²

    donde Σ_norm(t) = diag(Σ(t)) / diag(Σ_final)

    ALGORITMO (objective.ml:437-452):
    ----------------------------------
    1. Calcular γ = ∂E[r]/∂μ para estado steady-state (μ, Σ)
    2. Construir Jacobiano J = diag(1/τ) · (W·diag(γ) - I)
    3. Inicializar S = Σ (covarianza final/steady-state)
    4. Iterar hacia atrás en el tiempo:
        a. Normalizar diagonal: S_norm_ii = S_ii / Σ_ii
        b. Acumular costo: cost += ||S_norm||²
        c. Evolucionar S: S ← S + dt·(S @ J^T)
    5. Retornar λ_slow · cost

    INTERPRETACIÓN:
    ---------------
    - Si la red tiene oscilaciones grandes durante transitorios,
      Σ(t) variará mucho y el costo será alto
    - Penalizando esto, forzamos dinámica más estable y predecible
    - Útil principalmente en Stage 2 para refinar parámetros de ruido

    CÓDIGO ORIGINAL: objective.ml lines 437-452
    - Line 440-442: Calcular gamma vector
    - Line 443: Construir Jacobiano j_mat
    - Line 444-452: Iterar evolución de covarianza

    Parameters
    ----------
    w : jax.Array, shape (N, N)
        Connectivity matrix W
    mu : jax.Array, shape (N,)
        Mean vector at steady-state (from ADF evolution)
    sigma : jax.Array, shape (N, N)
        Covariance matrix at steady-state (from ADF evolution)
    inv_taus : jax.Array, shape (N,)
        Inverse time constants 1/τ_α
    dt : float
        Time step for slowness integration (seconds)
    t_max_slow : float
        Maximum time for slowness integration (seconds)
        Typically same as t_max for main evolution
    k : float
        Supralinear scaling constant (default 0.3)
    lambda_slow : float
        Slowness penalty weight (default 0, can use 0.1)
        Already normalized by system size

    Returns
    -------
    cost_slow : float
        Slowness penalty cost

    Notes
    -----
    - Esta función se llama SOLO en Stage 2 (L-BFGS con ADF)
    - En Stage 1 (ADAM con samples), lambda_slow = 0
    - Default lambda_slow = 0, pero puede ser ~0.1 para estabilidad
    - El costo se normaliza por (2 * N * n_time_bins_slow * n_targets)
      antes de multiplicar por lambda_slow (ver objective.ml:113)

    References
    ----------
    .. [1] ssn_inference_optimizer/objective.ml lines 437-452
    .. [2] ssn_inference_optimizer/objective.ml line 113 (normalization)
    """
    # PASO 1: Calcular momentos no lineales γ = ∂E[r]/∂μ
    # Necesitamos γ para construir el Jacobiano
    # Ref: objective.ml:440-442
    _, gamma = compute_nonlinear_moments(mu, sigma, k)

    # PASO 2: Construir Jacobiano J = diag(1/τ) · (W·diag(γ) - I)
    # Este Jacobiano controla cómo evoluciona la covarianza
    # Ref: objective.ml:443
    j_mat = compute_jacobian_matrix(w, gamma, inv_taus)

    # PASO 3: Normalizar lambda_slow por tamaño del sistema
    # Ref: objective.ml:113
    # lambda_slow /= (2 * N * n_time_bins_slow * n_targets)
    N = len(mu)
    n_time_bins_slow = int(t_max_slow / dt)
    n_targets = 1  # Típicamente 1 target
    lambda_slow_norm = lambda_slow / (2.0 * N * n_time_bins_slow * n_targets)

    # PASO 4: Extraer diagonal de Σ para normalización
    # Usaremos esto para normalizar S en cada paso
    sigma_diag = jnp.diag(sigma)  # Shape: (N,)

    # PASO 5: Iterar evolución de covarianza hacia atrás
    # Ref: objective.ml:444-452
    def iterate_step(carry, t):
        """
        Single iteration step for slowness cost.

        carry: (S_current, cost_accumulated)
        t: timestep index
        """
        s_current, cost_accu = carry

        # a. Normalizar diagonal de S por diagonal de Σ
        # s_norm_ii = S_ii / Σ_ii
        # Ref: objective.ml:447
        s_diag = jnp.diag(s_current)  # Shape: (N,)
        s_norm_diag = s_diag / (sigma_diag + 1e-10)  # Evitar div/0

        # b. Acumular costo: ||s_norm_diag||²
        # Ref: objective.ml:448
        cost_step = jnp.sum(s_norm_diag ** 2)
        cost_accu = cost_accu + cost_step

        # c. Evolucionar S: dS/dt = S @ J^T
        # Discretización: S_new = S + dt·(S @ J^T)
        # Ref: objective.ml:449
        ds_dt = jnp.dot(s_current, j_mat.T)
        s_new = s_current + dt * ds_dt

        return (s_new, cost_accu), None

    # PASO 6: Inicializar con Σ (steady-state) e iterar
    # Ref: objective.ml:452
    s_init = sigma
    cost_init = 0.0

    # Iterar n_time_bins_slow pasos
    # Ref: objective.ml:444-451 (función recursiva iterate)
    (s_final, total_cost), _ = jax.lax.scan(
        iterate_step,
        (s_init, cost_init),
        jnp.arange(n_time_bins_slow)
    )

    # PASO 7: Aplicar lambda_slow normalizado
    # Ref: objective.ml:445
    return lambda_slow_norm * total_cost


def update_temporal_weighting(temporal_weighting, n_time_bins, min_time_bins,
                              iteration, max_iterations=200,
                              no_progression=False):
    """
    Update temporal weighting for annealing during ADAM training.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Durante el entrenamiento con ADAM (sample-based), se actualiza
    progresivamente la ventana temporal para implementar annealing.

    ALGORITMO (train.ml lines 240-243):
    ------------------------------------
    zero_up_to = min(n_time_bins - min_time_bins,
                     round(n_time_bins * iteration / max_iterations))

    for i in 1..n_time_bins:
        temporal_weighting[i] = 0.0 if i < zero_up_to else 1.0

    PROGRESIÓN:
    - iteration=0: zero_up_to ≈ n_time_bins - min_time_bins
      → Solo últimos min_time_bins evaluados
    - iteration=100: zero_up_to ≈ n_time_bins/2
      → Segunda mitad evaluada
    - iteration=200: zero_up_to = 0
      → Todos los bins evaluados

    PAPER: Echeveste et al. (2020) Methods, página 18:
    "the beginning of the averaging time window, T_min in Eqs. 26-28,
    was systematically changed ('annealed') from T_min = 0 ms to
    T_max - 50 ms during the initial 200 iterations"

    Parameters
    ----------
    temporal_weighting : np.ndarray, shape (n_time_bins,)
        Current temporal weighting vector (modified in-place)
    n_time_bins : int
        Total number of time bins
    min_time_bins : int
        Minimum number of bins at the end that always have weight 1.0
    iteration : int
        Current training iteration
    max_iterations : int
        Maximum iterations for annealing (default 200, from OCaml code)
    no_progression : bool
        If True, skip annealing and use final weighting immediately
        Ref: train.ml line 242: check "-no_progression"

    Returns
    -------
    None
        Modifies temporal_weighting in-place

    Notes
    -----
    Ref: train.ml lines 240-243
    let update_temporal_weighting iter =
      let zero_up_to = min (X.n_time_bins - X.min_time_bins)
        (if Cmdargs.check "-no_progression" then max_int
         else round (float n_time_bins *. float iter /. 200.)) in
      for i=1 to X.n_time_bins do
        X.Samples.temporal_weighting.{i} <- if i<zero_up_to then 0.
                                            else 1. done
    """
    if no_progression:
        # Sin progresión: ir directo al final (todos bins = 1.0)
        zero_up_to = 0
    else:
        # Con progresión: expandir gradualmente la ventana
        # Ref: train.ml line 241-242
        progression = int(
            round(n_time_bins * iteration / max_iterations)
        )
        zero_up_to = min(n_time_bins - min_time_bins, progression)

    # Actualizar vector in-place
    # Ref: train.ml line 243
    for i in range(n_time_bins):
        temporal_weighting[i] = 0.0 if i < zero_up_to else 1.0


def compute_evolution_costs(
    w, h, sigma_eta, inv_taus, dt, tau_eta, t_max, t_subsamp,
    target_mu, target_sigma, k=0.3,
    lambda_mean=1.0, lambda_var=1.0, lambda_cov=1.0,
    lambda_slow=0.0,
    min_time=0.05, temporal_weighting=None
):
    """
    Compute cost over entire time evolution using ADF.

    JUSTIFICACIÓN MATEMÁTICA
    -------------------------
    Esta es la FUNCIÓN DE COSTO COMPLETA que se optimiza durante
    el entrenamiento.

    Concepto: No solo queremos que el SSN coincida con el GSM en el estado
    FINAL, sino durante todo el período de "lectura" (min_time hasta t_max).

    ¿Por qué integrar en el tiempo?

    1. ESTABILIDAD: Asegura que la red no solo llega al target, sino que
       se mantiene estable cerca de él durante un período de tiempo.

    2. SAMPLING: Para probabilistic inference, la red debe generar samples
       durante un período de tiempo. Queremos que TODO ese período refleje
       el posterior correcto, no solo el estado final.

    3. DINÁMICA REALISTA: Las redes neuronales biológicas tienen dinámica
       transitoria. Queremos capturar el comportamiento realista después
       de que la red se "asienta" (después de min_time).

    TEMPORAL WEIGHTING:
    -------------------
    El parámetro temporal_weighting permite ponderar diferentes timesteps
    en el cálculo del costo. Esto es CRUCIAL para:

    1. TRANSIENT FILTERING: Ignorar timesteps tempranos donde la red
       todavía está estabilizándose (wt=0 para t < t_max - min_time)

    2. TEMPORAL ANNEALING: Durante el entrenamiento con ADAM, se puede
       aumentar progresivamente la ventana temporal evaluada, permitiendo
       que la red primero aprenda el comportamiento asintótico y luego
       refine la dinámica transitoria.

    CÓDIGO ORIGINAL: objective.ml lines 374-400
    - Line 374: step_fun define pesos 0/1 según min_time
    - Line 375: temporal_weighting vector con pesos por timestep
    - Line 394: wt = temporal_weighting[t] multiplica cada costo
    - train.ml lines 240-243: update_temporal_weighting para annealing

    ALGORITMO:
    -----------
    1. Inicializar μ = 0, Σ = 4I, Σ* = 4I (condiciones iniciales)
       → Valores de objective.ml lines 404-407
       → 4I da varianza inicial moderada (ni muy pequeña ni muy grande)

    2. Loop temporal desde t=0 hasta t=t_max:
       a. Evolucionar (μ, Σ, Σ*) un paso dt usando ADF (Eqs. 17-19)
       b. Si t es múltiplo de t_subsamp:
          - Extraer solo neuronas excitatorias (primeras m)
          - Calcular costo comparando con targets del GSM
          - Multiplicar por temporal_weighting[t]
          - Acumular costo

    3. Retornar costo total acumulado

    ¿Por qué min_time?
    - La red necesita tiempo para "asentarse" (transiente inicial)
    - Solo evaluamos costo en los últimos min_time segundos
    - Típicamente min_time = 50ms (suficiente para estabilización)

    ¿Por qué t_subsamp?
    - No necesitamos evaluar costo en CADA paso temporal
    - Subsampling reduce costo computacional
    - Típicamente t_subsamp = 5ms (cada 5ms evaluamos costo)

    GRADIENTES:
    JAX calculará automáticamente ∂L/∂W usando diferenciación automática
    a través de TODO este loop temporal (backpropagation through time).

    CÓDIGO ORIGINAL: objective.ml lines 377-408
    - Lines 379-391: loop temporal con evolución de momentos
    - Lines 393-400: acumulación de costo con temporal weighting
    - Lines 404-407: condiciones iniciales

    Parameters
    ----------
    w : jax.Array or np.ndarray
        Connectivity matrix W, shape (n, n) where n = N_E + N_I
    h : jax.Array or np.ndarray
        External input h (from GSM), shape (n,)
    sigma_eta : jax.Array or np.ndarray
        Noise covariance Σ_η, shape (n, n)
    inv_taus : jax.Array or np.ndarray
        Inverse time constants 1/τ_α, shape (n,)
    dt : float
        Time step Δt (seconds, típicamente 0.0005s = 0.5ms)
    tau_eta : float
        Noise autocorrelation time τ_η (seconds, típicamente 0.01s = 10ms)
    t_max : float
        Maximum integration time (seconds, típicamente 0.1s = 100ms)
    t_subsamp : float
        Cost evaluation interval (seconds, típicamente 0.005s = 5ms)
    target_mu : np.ndarray
        Target mean μ_GSM from posterior, shape (m,) where m = N_E
    target_sigma : np.ndarray
        Target covariance Σ_GSM from posterior, shape (m, m)
    k : float
        Supralinear scaling constant (default 0.3)
    lambda_mean : float
        Weight λ_μ for mean matching (default 1.0)
    lambda_var : float
        Weight λ_σ² for variance matching (default 1.0)
    lambda_cov : float
        Weight λ_Σ for covariance matching (default 1.0)
    lambda_slow : float
        Weight λ_slow for slowness penalty (default 0.0)
        Ref: objective.ml:437-452, train.ml:39
        If > 0, penalizes rapid changes in covariance during dynamics
        Typical value: 0.1 for Stage 2 stability
    min_time : float
        Evaluation window at end (seconds, default 0.05s = 50ms)
    temporal_weighting : jax.Array or None
        Temporal weighting vector, shape (n_time_bins,)
        If None, uses step function: 0 before min_time, 1 after
        Ref: objective.ml lines 374-375

    Returns
    -------
    total_cost : float
        Total accumulated cost L = Σ_t [wt * (mean_cost + var_cost +
        cov_cost)]
    mu_final : jax.Array
        Final mean state μ(t_max), shape (n,)
    sigma_final : jax.Array
        Final covariance state Σ(t_max), shape (n, n)

    Notes
    -----
    Esta función es DIFERENCIABLE por JAX:
    - grad_w = jax.grad(lambda w: compute_evolution_costs(w, ...)[0])
    - Calcula ∂L/∂W automáticamente vía backprop through time
    - Esto es CRÍTICO para L-BFGS-B optimization

    El costo total puede ser muy grande (suma sobre muchos timesteps),
    pero L-BFGS-B se encarga de escalar los gradientes apropiadamente.

    Solo usamos neuronas E para el costo porque:
    - GSM solo provee targets para orientaciones (E neurons)
    - I neurons se optimizan implícitamente vía balance E/I
    """
    # PASO 0: Convertir todos los inputs a JAX arrays
    # JAX necesita arrays propios para diferenciación automática
    # numpy arrays no son diferenciables
    w = jnp.asarray(w)
    h = jnp.asarray(h)
    sigma_eta = jnp.asarray(sigma_eta)
    inv_taus = jnp.asarray(inv_taus)
    target_mu = jnp.asarray(target_mu)
    target_sigma = jnp.asarray(target_sigma)

    # PASO 1: Calcular parámetros temporales
    n = w.shape[0]  # n = N_E + N_I (total neuronas)
    m = n // 2  # m = N_E (solo excitatorias, asumiendo N_E = N_I)
    n_time_bins = int(t_max / dt)  # Número total de pasos temporales
    subsamp_bins = int(t_subsamp / dt)  # Intervalo entre evaluaciones
    min_time_bins = int(min_time / dt)  # Bins en ventana evaluación

    # PASO 1b: Crear temporal_weighting si no se provee
    # Ref: objective.ml line 374-375
    # step_fun i = if i<(n_time_bins - min_time_bins) then 0. else 1.
    if temporal_weighting is None:
        # Peso 0.0 para timesteps antes de min_time
        # Peso 1.0 para timesteps después de min_time
        temporal_weighting = jnp.where(
            jnp.arange(n_time_bins) < (n_time_bins - min_time_bins),
            0.0,
            1.0
        )
    else:
        temporal_weighting = jnp.asarray(temporal_weighting)

    # PASO 2: Condiciones iniciales de los momentos
    # Valores estándar del código original (objective.ml lines 404-407)
    # μ(0) = 0: red comienza en reposo (sin actividad)
    # Σ(0) = 4I: varianza inicial moderada (ni determinística ni muy ruidosa)
    # Σ*(0) = 4I: memoria inicial del ruido (mismo valor que Σ)
    mu = jnp.zeros(n)  # Medias iniciales en cero
    sigma = 4.0 * jnp.eye(n)  # Covarianza inicial proporcional a identidad
    sigma_star = 4.0 * jnp.eye(n)  # Aux. covarianza inicial

    # PASO 3: Inicializar acumuladores de costo
    # Separamos los tres componentes para debugging/analysis
    # (aunque solo retornamos el total)
    accu_mean = 0.0  # Acumulador de mean matching cost
    accu_var = 0.0  # Acumulador de variance matching cost
    accu_cov = 0.0  # Acumulador de covariance matching cost

    # PASO 4: Loop temporal principal
    # Evoluciona momentos desde t=0 hasta t=t_max
    # Acumula costo solo en ventana de evaluación (últimos min_time segundos)
    # objective.ml lines 379-401
    for t in range(n_time_bins):
        # SUB-PASO 4a: Evolucionar momentos un paso temporal Δt
        # Usa método de Euler explícito para integrar Eqs. 17-19
        # Actualiza (μ, Σ, Σ*) in-place para siguiente iteración
        mu, sigma, sigma_star = evolve_moments_single_step(
            mu, sigma, sigma_star, w, h, sigma_eta,
            dt, tau_eta, inv_taus, k
        )

        # SUB-PASO 4b: Evaluar y acumular costo (condicionalmente)
        # Condición: t % subsamp_bins == 0
        #   → Solo cada t_subsamp segundos (subsampling)
        #   → Reduce costo computacional sin perder información
        # objective.ml lines 393-400
        if t % subsamp_bins == 0:
            # Obtener peso temporal para este timestep
            # wt = 0.0 si estamos en transiente inicial
            # wt = 1.0 si estamos en ventana de evaluación
            # Ref: objective.ml line 394
            wt = temporal_weighting[t]

            # Extraer solo neuronas EXCITATORIAS (primeras m)
            # Las I neurons (últimas n-m) no se usan para costo
            # porque GSM solo provee targets para orientaciones (E)
            mu_exc = mu[:m]  # Media de E neurons
            sigma_exc = sigma[:m, :m]  # Covarianza de E neurons

            # Calcular los tres componentes de costo
            # Compara μ_exc, Σ_exc con μ_target, Σ_target del GSM
            c_mean, c_var, c_cov = compute_cost_components(
                mu_exc, sigma_exc, target_mu, target_sigma,
                lambda_mean, lambda_var, lambda_cov
            )

            # Acumular costos para este timestep MULTIPLICADOS POR wt
            # El costo total será la suma ponderada sobre timesteps
            # Ref: objective.ml line 399
            accu_mean += wt * c_mean
            accu_var += wt * c_var
            accu_cov += wt * c_cov

    # PASO 5: Calcular costo total (matching terms)
    # L_match = Σ_t [L_mean(t) + L_var(t) + L_cov(t)]
    total_cost = accu_mean + accu_var + accu_cov

    # PASO 5b: Agregar slowness penalty si lambda_slow > 0
    # Ref: objective.ml:464-466, line 473
    # El slowness cost penaliza cambios rápidos en Σ durante dinámica
    # Solo se usa en Stage 2 (L-BFGS) con lambda_slow típicamente 0.1
    if lambda_slow > 0:
        cost_slow = compute_slowness_cost(
            w, mu, sigma, inv_taus, dt, t_max, k, lambda_slow
        )
        total_cost = total_cost + cost_slow

    # PASO 6: Retornar costo y estados finales
    # - total_cost: usado por optimizador (JAX array, no convertir a float)
    # - mu, sigma: útiles para debugging/visualización
    # IMPORTANTE: No usar float() aquí porque rompe autodiff de JAX
    return total_cost, mu, sigma


# =============================================================================
# Optimization objective function
# =============================================================================

def create_objective_function(
    gsm_model, N_E, N_I, tau_e, tau_i, tau_eta, k,
    dt, t_max, t_subsamp, min_time=0.05,
    lambda_mean=1.0, lambda_var=1.0, lambda_cov=1.0
):
    """
    Create objective function for L-BFGS-B optimization.

    Crea la función objetivo que será minimizada durante el
    entrenamiento usando L-BFGS-B.

    Parameters
    ----------
    gsm_model : object
        Pre-trained GSM model with target statistics
    N_E : int
        Number of excitatory neurons
    N_I : int
        Number of inhibitory neurons
    tau_e : float
        Excitatory time constant (seconds)
    tau_i : float
        Inhibitory time constant (seconds)
    tau_eta : float
        Noise autocorrelation time (seconds)
    k : float
        Supralinear scaling constant
    dt : float
        Time step (seconds)
    t_max : float
        Maximum integration time (seconds)
    t_subsamp : float
        Cost subsampling interval (seconds)
    min_time : float
        Minimum time before cost evaluation
    lambda_mean : float
        Mean matching weight
    lambda_var : float
        Variance matching weight
    lambda_cov : float
        Covariance matching weight

    Returns
    -------
    objective_fn : callable
        Objective function f(x) -> cost
        Maps 15-parameter vector to scalar cost
    gradient_fn : callable
        Gradient function f'(x) -> grad
        Returns gradient with respect to all 15 parameters
    pack_fn : callable
        Function to pack physical parameters into optimization vector
        Signature: pack_fn(params_dict) -> x
    unpack_fn : callable
        Function to unpack optimization vector to physical parameters
        Signature: unpack_fn(x) -> params_dict

    Notes
    -----
    La función objetivo empaqueta/desempaqueta parámetros
    y calcula el costo usando ADF.
    """
    n = N_E + N_I

    # Inverse time constants
    inv_taus = jnp.concatenate([
        jnp.full(N_E, 1.0 / tau_e),
        jnp.full(N_I, 1.0 / tau_i)
    ])

    # Calcular input_baseline_lb dinámicamente desde el GSM
    # Ref: ssn_inference_optimizer/train.ml lines 47-49
    # Este valor garantiza que beta_h + filter_response >= 0.1
    # para todos los datos de entrenamiento
    input_baseline_lb = gsm_model.compute_input_baseline_lb(n_samples=100)
    print(f"Calculated input_baseline_lb: {input_baseline_lb:.6f}")

    # Placeholder: necesitamos el GSM model para obtener h y targets
    # Por ahora, usamos valores dummy
    h_vec = jnp.ones(n)  # Será reemplazado con h del GSM
    target_mu = jnp.zeros(N_E)  # Será reemplazado con μ del GSM posterior
    target_sigma = jnp.eye(N_E)  # Será reemplazado con Σ del GSM posterior

    def pack_parameters(params):
        """
        Pack physical parameters into optimization vector.

        This is the inverse operation of unpack_parameters. It packs
        the 15 physical parameters into a vector for optimization,
        applying the appropriate transformations.

        Uses the input_baseline_lb computed dynamically from the GSM
        model to ensure proper parameter constraints.

        Parameters
        ----------
        params : dict
            Dictionary with physical parameters:
            - 'input_baseline': β_h (offset/baseline)
            - 'input_scaling': α_h (multiplicador/scaling)
            - 'input_nl_pow': γ_h (exponente/power)
            - 'a_EE', 'a_EI', 'a_IE', 'a_II': weight heights
            - 'd_EE', 'd_EI', 'd_IE', 'd_II': weight widths
            - 'sigma_eta_width': d_σ
            - 'sigma_eta_std_e': σ_E
            - 'sigma_eta_std_i': σ_I
            - 'sigma_eta_rho': ρ (in [0, 1])

        Returns
        -------
        x : jax.numpy.ndarray, shape (15,)
            Packed optimization vector

        References
        ----------
        .. [1] ssn_inference_optimizer/objective.ml lines 185-214

        Notes
        -----
        The input_baseline_lb used in the transformation is computed
        from the GSM data to guarantee positive arguments in the
        nonlinear transformation.
        """
        x = jnp.zeros(15)

        # Input transformation parameters (indices 0-2)
        # Ref: objective.ml lines 189-191
        # x[0] → input_baseline (β_h): offset que garantiza positividad
        # x[1] → input_scaling (α_h): multiplicador de escala
        # x[2] → input_nl_pow (γ_h): exponente de la no linealidad
        x = x.at[0].set(jnp.sqrt(params['input_baseline'] -
                                 input_baseline_lb))
        x = x.at[1].set(jnp.sqrt(params['input_scaling']))
        x = x.at[2].set(jnp.sqrt(params['input_nl_pow']))

        # Weight matrix heights (indices 3-6)
        # Inverse of: a = 0.01 + x²
        # Check that heights are > 0.01
        for i, key in enumerate(['a_EE', 'a_EI', 'a_IE', 'a_II']):
            h = params[key]
            if h <= 0.011:
                raise ValueError(
                    f"Weight height {key}={h} must be > 0.011"
                )
            x = x.at[3 + i].set(jnp.sqrt(h - 0.01))

        # Weight matrix widths (indices 7-10)
        for i, key in enumerate(['d_EE', 'd_EI', 'd_IE', 'd_II']):
            x = x.at[7 + i].set(params[key])

        # Noise covariance parameters (indices 11-14)
        x = x.at[11].set(params['sigma_eta_width'])
        x = x.at[12].set(params['sigma_eta_std_e'])
        x = x.at[13].set(params['sigma_eta_std_i'])

        # Inverse of: ρ = 0.5 * (1 + tanh(x[14]))
        # This gives: x[14] = arctanh(2*ρ - 1)
        # Using: arctanh(z) = 0.5 * (log(1+z) - log(1-z))
        rho = params['sigma_eta_rho']
        z = 2.0 * rho - 1.0
        rho_packed = 0.5 * (jnp.log(1.0 + z) - jnp.log(1.0 - z))
        x = x.at[14].set(rho_packed)

        return x

    def unpack_parameters(x):
        """
        Unpack optimization vector to physical parameters.

        Unpacks the 15-parameter vector into physical parameters for the
        SSN model following the parametrization in Echeveste et al. 2020.

        Uses the input_baseline_lb computed dynamically from the GSM
        model to ensure proper parameter constraints.

        Parameters
        ----------
        x : array_like, shape (15,)
            Optimization parameter vector. The parameters are packed in
            the following order (see objective.ml lines 153-180):
            - x[0]: input_baseline (β_h) - transformed as β_h_lb + x[0]²
            - x[1]: input_scaling (α_h) - transformed as x[1]²
            - x[2]: input_nl_pow (γ_h) - transformed as x[2]²
            - x[3:7]: Weight heights a_EE, a_EI, a_IE, a_II
            - x[7:11]: Weight widths d_EE, d_EI, d_IE, d_II
            - x[11]: Noise width (d_σ)
            - x[12]: Noise E std (σ_E)
            - x[13]: Noise I std (σ_I)
            - x[14]: Noise correlation (ρ) - transformed via tanh

        Returns
        -------
        params : dict
            Dictionary with unpacked parameters:
            - 'input_baseline': β_h (offset/baseline)
            - 'input_scaling': α_h (multiplicador/scaling)
            - 'input_nl_pow': γ_h (exponente/power)
            - 'a_EE', 'a_EI', 'a_IE', 'a_II': weight heights
            - 'd_EE', 'd_EI', 'd_IE', 'd_II': weight widths
            - 'sigma_eta_width': d_σ
            - 'sigma_eta_std_e': σ_E
            - 'sigma_eta_std_i': σ_I
            - 'sigma_eta_rho': ρ

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Eq. 10, 12-14
        .. [2] ssn_inference_optimizer/objective.ml lines 153-180

        Notes
        -----
        The input_baseline_lb used in the transformation is computed
        from the GSM data to guarantee positive arguments in the
        nonlinear transformation.
        """
        params = {}

        # Input transformation parameters (indices 0-2)
        # Ref: objective.ml lines 157-159
        params['input_baseline'] = input_baseline_lb + x[0]**2
        params['input_scaling'] = x[1]**2
        params['input_nl_pow'] = x[2]**2

        # Weight matrix heights (indices 3-6)
        # Ref: objective.ml lines 161-164
        # Minimum value of 0.01 ensures numerical stability
        params['a_EE'] = 0.01 + x[3]**2
        params['a_EI'] = 0.01 + x[4]**2
        params['a_IE'] = 0.01 + x[5]**2
        params['a_II'] = 0.01 + x[6]**2

        # Weight matrix widths (indices 7-10)
        # Ref: objective.ml lines 165-168
        params['d_EE'] = x[7]
        params['d_EI'] = x[8]
        params['d_IE'] = x[9]
        params['d_II'] = x[10]

        # Noise covariance parameters (indices 11-14)
        # Ref: objective.ml lines 174-178
        params['sigma_eta_width'] = x[11]
        params['sigma_eta_std_e'] = x[12]
        params['sigma_eta_std_i'] = x[13]
        # rho is constrained to [0, 1] via tanh transformation
        params['sigma_eta_rho'] = 0.5 * (1.0 + jnp.tanh(x[14]))

        return params

    def build_w_from_params(params):
        """Construir matriz W desde parámetros."""
        # Basado en objective.ml lines 277-287
        W = jnp.zeros((n, n))

        # Orientaciones en el ring
        theta = jnp.linspace(0, jnp.pi, N_E, endpoint=False)

        # Función auxiliar para construir bloques
        # Ref: objective.ml lines 285-287
        # Formula: W_XY(θi, θj) = s * a * exp[(cos(θi - θj) - 1) / d²]
        # IMPORTANTE: SIN factor 2 en el coseno (ver Equation 10 del paper)
        def connectivity_block(theta_pre, theta_post, a, d, sign):
            delta = theta_pre[:, None] - theta_post[None, :]
            return sign * a * jnp.exp(
                (jnp.cos(delta) - 1) / (d**2)
            )

        # Construir bloques
        W = W.at[:N_E, :N_E].set(
            connectivity_block(theta, theta, params['a_EE'],
                               params['d_EE'], 1.0)
        )
        W = W.at[:N_E, N_E:].set(
            connectivity_block(theta, theta, params['a_EI'],
                               params['d_EI'], -1.0)
        )
        W = W.at[N_E:, :N_E].set(
            connectivity_block(theta, theta, params['a_IE'],
                               params['d_IE'], 1.0)
        )
        W = W.at[N_E:, N_E:].set(
            connectivity_block(theta, theta, params['a_II'],
                               params['d_II'], -1.0)
        )

        return W

    def build_sigma_eta(params):
        """
        Build noise covariance matrix Σ_η from parameters.

        Constructs the noise covariance matrix following the
        parametrization in Echeveste et al. 2020, Equations 12-13.
        The covariance is spatially structured with a squared
        exponential kernel.

        Parameters
        ----------
        params : dict
            Parameter dictionary containing:
            - 'sigma_eta_width': spatial width d_σ
            - 'sigma_eta_std_e': E population std σ_E
            - 'sigma_eta_std_i': I population std σ_I
            - 'sigma_eta_rho': E-I correlation ρ

        Returns
        -------
        Sigma_eta : jax.numpy.ndarray, shape (2*N_E, 2*N_E)
            Noise covariance matrix.

        References
        ----------
        .. [1] Echeveste et al. (2020) Eq. 12-13
        .. [2] ssn_inference_optimizer/objective.ml lines 289-304
        """
        # Extract parameters
        width = params['sigma_eta_width']
        std_e = params['sigma_eta_std_e']
        std_i = params['sigma_eta_std_i']
        rho = params['sigma_eta_rho']

        var_e = std_e ** 2
        var_i = std_i ** 2

        theta = jnp.linspace(0, jnp.pi, N_E, endpoint=False)
        # Δθ
        delta = theta[:, None] - theta[None, :]

        # Spatial kernel: exp((cos(Δθ) - 1) / d_σ²)
        # Ref: objective.ml line 303
        spatial_kernel = jnp.exp((jnp.cos(delta) - 1) / (width**2))

        # Build blocks of the covariance matrix
        # Ref: Echeveste et al. 2020, Eq. 12-13
        Sigma_ee = var_e * spatial_kernel
        Sigma_ii = var_i * spatial_kernel
        Sigma_ei = rho * jnp.sqrt(var_e * var_i) * spatial_kernel

        # Assemble full matrix
        Sigma_eta = jnp.block([
            [Sigma_ee, Sigma_ei],
            [Sigma_ei.T, Sigma_ii]
        ])

        # Add small diagonal term for numerical stability
        # Ref: objective.ml line 305
        Sigma_eta = Sigma_eta + 0.01 * jnp.eye(n)

        return Sigma_eta

    def transform_h_input(h_vec, params):
        """
        Transform input h using learned nonlinear transformation.

        Applies the nonlinear transformation:
        h = α_h · (β_h + h_vec)^γ_h

        where α_h is scaling, β_h is baseline, γ_h is power.

        Parameters
        ----------
        h_vec : array_like
            Input vector from GSM model
        params : dict
            Parameter dictionary containing:
            - 'input_baseline': β_h (offset/baseline)
            - 'input_scaling': α_h (multiplicador/scaling)
            - 'input_nl_pow': γ_h (exponente/power)

        Returns
        -------
        h : jax.numpy.ndarray
            Transformed input vector

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Eq. 14
        .. [2] ssn_inference_optimizer/objective.ml lines 410-413
            Original formula:
            prms.input_scaling * exp(prms.input_nl_pow * log(
                h_vec + prms.input_baseline))
            = input_scaling · (h_vec + input_baseline)^input_nl_pow
            = α_h · (β_h + h_vec)^γ_h
        """
        # Extraer parámetros con nombres correctos según nomenclatura
        # α_h = input_scaling (multiplicador)
        # β_h = input_baseline (offset)
        # γ_h = input_nl_pow (exponente)
        alpha_h = params['input_scaling']
        beta_h = params['input_baseline']
        gamma_h = params['input_nl_pow']

        # h = α_h · exp(γ_h · log(β_h + h_vec))
        # This is equivalent to: h = α_h · (β_h + h_vec)^γ_h
        # Ref: objective.ml line 413
        return alpha_h * jnp.exp(gamma_h * jnp.log(beta_h + h_vec))

    def objective(x):
        """
        Optimization objective function.

        Computes the total cost for the SSN training following
        Echeveste et al. 2020. The cost includes matching terms
        for mean, variance, and covariance, plus regularization.

        Parameters
        ----------
        x : array_like, shape (15,)
            Optimization parameter vector

        Returns
        -------
        cost : float
            Total objective value

        References
        ----------
        .. [1] Echeveste et al. (2020) Nature Neuroscience, Eq. 25
        .. [2] ssn_inference_optimizer/objective.ml lines 469-474
        """
        # Unpack all 15 parameters
        params = unpack_parameters(x)

        # Build connectivity matrix W
        w = build_w_from_params(params)

        # Build noise covariance matrix Σ_η
        sigma_eta = build_sigma_eta(params)

        # Transform input h
        h = transform_h_input(h_vec, params)

        # Compute evolution costs
        cost, _, _ = compute_evolution_costs(
            w, h, sigma_eta, inv_taus,
            dt, tau_eta, t_max, t_subsamp,
            target_mu, target_sigma, k,
            lambda_mean, lambda_var, lambda_cov,
            min_time
        )

        return cost

    # Create gradient function using JAX automatic differentiation
    gradient_fn = jax.grad(objective)

    # Return tuple: (objective, gradient, pack, unpack)
    # This allows users to easily initialize optimization and inspect params
    return objective, gradient_fn, pack_parameters, unpack_parameters


# =============================================================================
# Sample-based stochastic optimization (for ADAM stage)
# =============================================================================


def simulate_ssn_with_noise(w, h_vec, sigma_eta, inv_taus, dt,
                            tau_eta, t_max, k, key, u_init=None):
    """
    Simulate SSN dynamics with process noise for one trial.

    This function implements stochastic simulation of the SSN dynamics
    (Eq. 8) by integrating with Euler-Maruyama method and adding
    correlated noise at each timestep.

    Based on Echeveste et al. (2020) Methods section:
    "we employed a stochastic gradient method using N_trial = 50 trials
    for each training stimulus to estimate the corresponding moments of
    network responses"

    Parameters
    ----------
    w : jax.Array, shape (N, N)
        Connectivity matrix
    h_vec : jax.Array, shape (N,)
        External input vector
    sigma_eta : jax.Array, shape (N, N)
        Noise covariance matrix
    inv_taus : jax.Array, shape (N,)
        Inverse time constants (1/τ_α)
    dt : float
        Integration timestep (seconds)
    tau_eta : float
        Noise correlation time constant (seconds)
    t_max : float
        Total simulation time (seconds)
    k : float
        Supralinear activation scaling factor
    key : jax.random.PRNGKey
        Random key for noise generation
    u_init : jax.Array, shape (N,), optional
        Initial membrane potentials. If None, sampled from N(0, I)

    Returns
    -------
    u_trajectory : jax.Array, shape (n_steps, N)
        Trajectory of membrane potentials over time
    r_trajectory : jax.Array, shape (n_steps, N)
        Trajectory of firing rates over time

    References
    ----------
    .. [1] Echeveste et al. (2020) Nature Neuroscience, Methods page 18
    .. [2] ssn_inference_optimizer/objective.ml lines 485-515
    """
    N = len(h_vec)
    n_steps = int(t_max / dt)

    # Inicialización: condiciones iniciales
    # Paper: "Initial conditions were drawn from a Gaussian distribution
    # N(μ0, Σ0)"
    if u_init is None:
        key, subkey = jax.random.split(key)
        # Por simplicidad usamos N(0, I). En el paper usan valores específicos
        # (Supplementary Table S1)
        u_init = jax.random.normal(subkey, shape=(N,))

    # Factorización de Cholesky de Σ_η para generar ruido correlacionado
    # η_t ~ N(0, Σ_η) se genera como: η_t = L @ ξ_t donde ξ_t ~ N(0, I)
    L_eta = jnp.linalg.cholesky(sigma_eta)

    # Ruido correlacionado temporalmente (proceso de Ornstein-Uhlenbeck)
    # dη/dt = -η/τ_η + ξ(t) donde ξ(t) es ruido blanco
    # Discretización: η_{t+1} = α·η_t + β·ξ_t
    # donde α = exp(-dt/τ_η), β = sqrt(Σ_η * (1 - α²))
    alpha_eta = jnp.exp(-dt / tau_eta)
    beta_eta_scale = jnp.sqrt(1 - alpha_eta**2)

    def step(carry, t):
        """Single integration step with noise."""
        u, eta = carry
        key_t = jax.random.fold_in(key, t)

        # Activación supralineal: r = k * [u]_+^2
        r = k * jnp.maximum(0, u)**2

        # Dinámica determinística: du/dt = -u + W@r + h
        # Ref: Echeveste Eq. 8 (without noise term)
        f = inv_taus * (-u + w @ r + h_vec)

        # Actualizar ruido correlacionado (Ornstein-Uhlenbeck)
        # Paper: "the process noise were re-sampled for each trial and
        # iteration"
        xi = jax.random.normal(key_t, shape=(N,))
        noise_white = L_eta @ xi
        eta_new = alpha_eta * eta + beta_eta_scale * noise_white

        # Integración Euler-Maruyama:
        # u_{t+1} = u_t + dt * f(u_t) + eta_t
        u_new = u + dt * f + eta_new

        # Soft threshold para evitar explosión numérica
        # Ref: objective.ml:538-539
        # Paper: Limita u al rango ~[-100, 100] para estabilidad
        soft_gain = 100.0
        u_new = soft_gain * jnp.tanh(u_new / soft_gain)

        return (u_new, eta_new), (u_new, r)

    # Inicializar ruido
    key, subkey = jax.random.split(key)
    eta_init = L_eta @ jax.random.normal(subkey, shape=(N,))

    # Integrar dinámica
    _, (u_traj, r_traj) = jax.lax.scan(
        step,
        (u_init, eta_init),
        jnp.arange(n_steps)
    )

    return u_traj, r_traj


def compute_costs_with_samples(w, h_vec, sigma_eta, inv_taus, dt,
                               tau_eta, t_max, t_subsamp, target_mu,
                               target_sigma, k, lambda_mean, lambda_var,
                               lambda_cov, min_time, n_trials, key,
                               temporal_weighting=None):
    """
    Compute costs using sample-based stochastic optimization.

    This function implements the first stage of training described in
    Echeveste et al. (2020), where N_trial = 50 trials are simulated
    with process noise to estimate network response moments.

    CRITICAL DIFFERENCES FROM ADF:
    -------------------------------
    1. Simula N_trial trayectorias estocásticas completas
    2. Computa momentos empíricos across trials EN CADA TIMESTEP
    3. Acumula costos timestep-por-timestep con temporal_weighting
    4. Momentos de u (membrane potentials), NO de r (firing rates)

    ALGORITMO (siguiendo objective.ml:530-556):
    --------------------------------------------
    1. Inicializar u0 ~ N(0, 4I) para n_trials (objetivo.ml:554)
    2. Para t = 0 hasta n_time_bins:
       a. Actualizar u para todos los trials en paralelo
       b. Si t es múltiplo de subsamp_bins:
          - Calcular mu = mean(u) across trials (objective.ml:541)
          - Calcular sigma = cov(u) across trials (objective.ml:542-543)
          - Obtener wt = temporal_weighting[t] (objective.ml:545)
          - Acumular costos: accu += wt * costs (objective.ml:549)
    3. Retornar costos acumulados

    TEMPORAL WEIGHTING & ANNEALING:
    --------------------------------
    Durante el entrenamiento con ADAM (sample-based), se utiliza
    temporal weighting que se actualiza progresivamente (annealing):

    - Inicialmente: Solo se evalúan costos en últimos 50ms
    - Gradualmente: Se expande la ventana hacia atrás
    - Finalmente (iter=200): Se evalúan todos los timesteps desde t=0

    Esto facilita el entrenamiento porque:
    1. Comportamiento asintótico es más estable y fácil de aprender
    2. Dinámica transitoria es más compleja y se refina después

    CÓDIGO ORIGINAL: objective.ml lines 530-556
    - Line 541-543: Momentos across trials en cada timestep
    - Line 544-549: Aplicar temporal_weighting y acumular costos
    - Line 554: u0 inicializado con gaussian_noise(2.0)

    Based on Echeveste et al. (2020) Methods section, page 18:
    "During the first stage, we employed a stochastic gradient method
    using N_trial = 50 trials for each training stimulus to estimate the
    corresponding moments of network responses"

    "the beginning of the averaging time window, T_min in Eqs. 26-28,
    was systematically changed ('annealed') from T_min = 0 ms to
    T_max - 50 ms during the initial 200 iterations"

    Parameters
    ----------
    w : jax.Array, shape (N, N)
        Connectivity matrix
    h_vec : jax.Array, shape (N,)
        External input vector
    sigma_eta : jax.Array, shape (N, N)
        Noise covariance matrix
    inv_taus : jax.Array, shape (N,)
        Inverse time constants
    dt : float
        Integration timestep (seconds)
    tau_eta : float
        Noise correlation time constant (seconds)
    t_max : float
        Total simulation time (seconds)
    t_subsamp : float
        Subsampling interval for moment computation (seconds)
    target_mu : jax.Array, shape (N_E,)
        Target mean from GSM posterior
    target_sigma : jax.Array, shape (N_E, N_E)
        Target covariance from GSM posterior
    k : float
        Supralinear activation scaling factor
    lambda_mean : float
        Weight for mean matching term
    lambda_var : float
        Weight for variance matching term
    lambda_cov : float
        Weight for covariance matching term
    min_time : float
        Start of evaluation window (T_min, seconds) - NOT USED HERE
        (temporal_weighting handles this instead)
    n_trials : int
        Number of stochastic trials (typically 50)
    key : jax.random.PRNGKey
        Random key for reproducibility
    temporal_weighting : np.ndarray or None
        Temporal weighting vector, shape (n_time_bins,)
        If None, uses all 1.0 (evaluate all timesteps equally)
        During ADAM training, this vector is updated progressively
        to implement temporal annealing
        Ref: objective.ml line 528, 545, train.ml lines 240-243, 286

    Returns
    -------
    cost : float
        Total cost averaged over time with temporal weighting
    mu_empirical : jax.Array, shape (N_E,)
        Empirical mean of membrane potentials (last timestep)
    sigma_empirical : jax.Array, shape (N_E, N_E)
        Empirical covariance of membrane potentials (last timestep)

    References
    ----------
    .. [1] Echeveste et al. (2020) Nature Neuroscience, Methods page 18
    .. [2] ssn_inference_optimizer/objective.ml lines 530-556
    .. [3] ssn_inference_optimizer/train.ml lines 240-243 (annealing)
    """
    import numpy as np

    N = len(h_vec)
    N_E = len(target_mu)  # Número de neuronas excitatorias
    n_time_bins = int(t_max / dt)
    subsamp_bins = int(t_subsamp / dt)

    # Default temporal weighting (todos los timesteps con peso 1.0)
    # Ref: objective.ml line 528
    if temporal_weighting is None:
        temporal_weighting = np.ones(n_time_bins)

    # IMPORTANT: Convert temporal_weighting to JAX array for indexing
    # inside scan. This avoids TracerArrayConversionError when indexing
    # with JAX tracer
    temporal_weighting = jnp.array(temporal_weighting)

    # Normalizar lambdas por tamaño del sistema
    # Ref: objective.ml:110-113
    # "give the multipliers their natural scaling with system size"
    n_subsamp_bins = n_time_bins // subsamp_bins
    n_targets = 1  # Típicamente 1 target

    lambda_mean_norm = lambda_mean / (2.0 * N_E * n_targets * n_subsamp_bins)
    lambda_var_norm = lambda_var / (2.0 * N_E * n_targets * n_subsamp_bins)
    lambda_cov_norm = lambda_cov / (
        2.0 * N_E * N_E * n_targets * n_subsamp_bins
    )

    # Inicialización de Cholesky para ruido
    L_eta = jnp.linalg.cholesky(sigma_eta)

    # Parámetros de ruido OU
    alpha_eta = jnp.exp(-dt / tau_eta)
    beta_eta_scale = jnp.sqrt(1 - alpha_eta**2)

    # Inicializar condiciones iniciales para todos los trials
    # Ref: objective.ml:554 - gaussian_noise 2.0
    key, subkey = jax.random.split(key)
    u_current = 2.0 * jax.random.normal(
        subkey, shape=(N, n_trials)
    )  # Shape: (N, n_trials)

    # Inicializar ruido para cada trial
    key, subkey = jax.random.split(key)
    eta_current = L_eta @ jax.random.normal(
        subkey, shape=(N, n_trials)
    )  # Shape: (N, n_trials)

    # Acumuladores de costo
    # Ref: objective.ml:555
    accu_mean = 0.0
    accu_var = 0.0
    accu_cov = 0.0

    # Variables para retornar (último timestep)
    mu_empirical = jnp.zeros(N_E)
    sigma_empirical = jnp.zeros((N_E, N_E))

    def timestep_update(carry, t):
        """Single timestep update for all trials in parallel."""
        u, eta, accu_m, accu_v, accu_c = carry

        # Generar key para este timestep
        key_t = jax.random.fold_in(key, t)

        # Activación supralineal: r = k * [u]_+^2
        # Shape: (N, n_trials)
        r = k * jnp.maximum(0, u)**2

        # Dinámica determinística: du/dt = -u + W@r + h
        # Ref: objective.ml:534-536
        # Shape: (N, n_trials)
        f = inv_taus[:, None] * (-u + (w @ r) + h_vec[:, None])

        # Actualizar ruido OU para cada trial
        # Ref: objective.ml:524 (proceso OU discretizado)
        xi = jax.random.normal(key_t, shape=(N, n_trials))
        noise_white = L_eta @ xi
        eta_new = alpha_eta * eta + beta_eta_scale * noise_white

        # Integración Euler-Maruyama
        u_new = u + dt * f + eta_new

        # Soft threshold para estabilidad
        # Ref: objective.ml:538-539
        soft_gain = 100.0
        u_new = soft_gain * jnp.tanh(u_new / soft_gain)

        # Calcular momentos across trials si es timestep subsampleado
        # Ref: objective.ml:544-549
        def compute_costs_at_timestep():
            # Momentos across trials (solo neuronas excitatorias)
            # Ref: objective.ml:541 - mu across trials
            u_exc = u_new[:N_E, :]  # Shape: (N_E, n_trials)
            mu = u_exc.mean(axis=1)  # Shape: (N_E,)

            # Ref: objective.ml:542-543 - sigma across trials
            u_centered = u_exc - mu[:, None]  # Shape: (N_E, n_trials)
            sigma = (u_centered @ u_centered.T) / n_trials  # (N_E, N_E)

            # Temporal weighting
            # Ref: objective.ml:545
            wt = temporal_weighting[t]

            # Costos
            # Ref: objective.ml:546-548
            target_var = jnp.diag(target_sigma)
            empirical_var = jnp.diag(sigma)

            c_mean = lambda_mean_norm * jnp.sum((mu - target_mu)**2)
            c_var = lambda_var_norm * jnp.sum((empirical_var - target_var)**2)
            c_cov = lambda_cov_norm * jnp.sum((sigma - target_sigma)**2)

            # Acumular con temporal_weighting
            # Ref: objective.ml:549
            return (
                accu_m + wt * c_mean,
                accu_v + wt * c_var,
                accu_c + wt * c_cov,
                mu,
                sigma
            )

        def no_costs():
            return accu_m, accu_v, accu_c, mu_empirical, sigma_empirical

        # Solo calcular costos en timesteps subsampleados
        is_subsamp = (t % subsamp_bins == 0)
        new_accu_m, new_accu_v, new_accu_c, new_mu, new_sigma = jax.lax.cond(
            is_subsamp,
            compute_costs_at_timestep,
            no_costs
        )

        return (
            (u_new, eta_new, new_accu_m, new_accu_v, new_accu_c),
            (new_mu, new_sigma)
        )

    # Integrar dinámica timestep por timestep
    # Ref: objective.ml:532-556 (función accumulate recursiva)
    carry_final, (mu_traj, sigma_traj) = jax.lax.scan(
        timestep_update,
        (u_current, eta_current, accu_mean, accu_var, accu_cov),
        jnp.arange(n_time_bins)
    )
    u_final, eta_final, accu_mean, accu_var, accu_cov = carry_final

    # Momentos del último timestep para retornar
    # Ref: objective.ml:551
    mu_empirical = mu_traj[-1]
    sigma_empirical = sigma_traj[-1]

    # Costo total acumulado
    # Ref: objective.ml:581-583
    total_cost = accu_mean + accu_var + accu_cov

    return total_cost, mu_empirical, sigma_empirical
