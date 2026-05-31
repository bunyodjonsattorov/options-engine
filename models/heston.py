"""Heston (1993) stochastic-volatility model.

Risk-neutral dynamics:
    dS_t  = (r - q) S_t dt + sqrt(v_t) S_t dW_1
    dv_t  = kappa * (theta - v_t) dt + sigma_v * sqrt(v_t) dW_2
    d<W_1, W_2>_t = rho dt

Parameters:
    kappa   : mean-reversion speed of variance
    theta   : long-run variance
    sigma_v : vol-of-vol
    rho     : correlation between price and variance shocks
              (negative for equities -> leverage effect / negative skew)
    v_0     : initial instantaneous variance
Feller condition:  2 kappa theta > sigma_v^2  (keeps v_t > 0 a.s.).
Usually softly violated by market-calibrated params; not catastrophic if
you avoid simulating with non-positivity-preserving Euler.

Pricing here:
    We compute European call/put via the Lewis (2001) formula, which
    integrates the characteristic function of log(S_T):

        C(K) = S e^{-qT} - sqrt(S K) / pi * e^{-(r + q) T / 2}
               * Re { integral_0^inf e^{-i u (log(K/F) - i/2 * ...)} * ... du }

    More precisely (cleaner form):
        C(K) = S e^{-qT}
               - K e^{-rT} / pi * Re { integral_0^inf
                       e^{-i u log(K/F)} * phi(u - i/2; T) / (u^2 + 1/4)
                   du }
    where F = S e^{(r-q)T} and phi is the char. fn. of log(S_T)/log(F)
    (we shift conventions inside the implementation). Quadrature is
    scipy.integrate.quad — fast enough for calibration with ~50 strikes.

    For pricing a whole vol surface efficiently in production you would
    swap the per-strike quadrature for Carr-Madan FFT (gives all strikes
    in one shot). Same math, different numerical packaging.

Why Heston is the right "stretch goal" to put on a Jane Street CV:
    * It's the canonical stoch-vol model — everyone in derivatives knows it.
    * It has a real characteristic function => good excuse to talk about
      Fourier methods.
    * It DOESN'T fit short-dated index skew well (vol of vol can't be
      high enough without blowing variance negative) — being able to say
      THAT is the difference between knowing it and using it.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


# ---------- characteristic function (Little Heston Trap form) ----------
def heston_char_function(
    u: complex,
    T: float,
    S0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
) -> complex:
    """Characteristic function of x_T = log(S_T) under risk-neutral measure.

    Uses the "Little Heston Trap" form (Albrecher et al 2007) — picks the
    sign of d so that g* in (0,1), avoiding branch-cut discontinuities
    that plague the original Heston (1993) form for long maturities.
    """
    iu = 1j * u
    xi = kappa - sigma_v * rho * iu
    d = np.sqrt(xi**2 + (sigma_v**2) * (iu + u**2))
    # Little Heston Trap: g_2 = (xi - d) / (xi + d)  (NOT the reciprocal)
    g2 = (xi - d) / (xi + d)
    exp_dT = np.exp(-d * T)

    C = (
        (r - q) * iu * T
        + (kappa * theta / sigma_v**2)
        * ((xi - d) * T - 2.0 * np.log((1.0 - g2 * exp_dT) / (1.0 - g2)))
    )
    D = ((xi - d) / sigma_v**2) * ((1.0 - exp_dT) / (1.0 - g2 * exp_dT))

    return np.exp(C + D * v0 + iu * np.log(S0))


# ---------- pricer (Carr-Madan damped call) ---------------------------
def heston_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    v0: float,
    option: str = "call",
    *,
    alpha: float = 1.5,
    u_upper: float = 200.0,
) -> float:
    """European call/put under Heston via the Carr-Madan damped integral.

    Carr-Madan (1999) recipe: the modified call price
        c_T(k) := e^{alpha k} * C_T(e^k)
    is square-integrable for alpha > 0 chosen large enough that
    E[S_T^{alpha+1}] is finite. Its Fourier transform has a closed form
    in terms of the log-spot char fn:
        psi_T(v) = e^{-rT} * phi(v - (alpha+1) i) /
                   ( alpha^2 + alpha - v^2 + i (2 alpha + 1) v )
    Then
        C_T(K) = e^{-alpha k} / pi * Re { integral_0^inf e^{-i v k} psi_T(v) dv }
    with k = log(K). Put follows by put-call parity.

    Why Carr-Madan (vs Lewis):
        * Numerically very stable for typical alpha in [1.0, 1.75].
        * Same formula gives a whole strip of strikes via FFT (we use
          scipy.integrate.quad here — calibration with ~50 strikes is
          fast enough). Swapping in pyfftw/scipy.fft is a one-line change
          for production speed.
    """
    k = np.log(K)

    def integrand(v):
        phi = heston_char_function(
            v - (alpha + 1) * 1j, T, S0, r, q, kappa, theta, sigma_v, rho, v0,
        )
        psi = np.exp(-r * T) * phi / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
        return np.real(np.exp(-1j * v * k) * psi)

    integral, _ = quad(integrand, 0.0, u_upper, limit=200)
    call = np.exp(-alpha * k) / np.pi * integral
    if option == "call":
        return float(call)
    # put-call parity: C - P = S e^{-qT} - K e^{-rT}
    put = call - S0 * np.exp(-q * T) + K * np.exp(-r * T)
    return float(put)
