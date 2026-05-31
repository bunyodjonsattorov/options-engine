"""SVI (Stochastic Volatility Inspired) smile parameterisation — Gatheral.

For a single maturity slice T, model the TOTAL implied variance
w(k) = sigma_imp(k)^2 * T as a function of log-moneyness k = log(K/F):

    w(k) = a + b * ( rho * (k - m) + sqrt( (k - m)^2 + sigma^2 ) )

Five parameters per slice:
    a in R       : vertical offset
    b >= 0       : slope magnitude (controls wing steepness)
    rho in (-1,1): tilt (negative for equity skew)
    m in R       : horizontal shift of the kink
    sigma > 0    : curvature / smoothness around the kink

Why this parameterisation (vs e.g. a polynomial):
    * Linear in the wings as |k| -> infinity (matches Lee's moment formula
      requirement that vol grows at most like sqrt(|k|)).
    * Easy to enforce no-arbitrage in the wings via parameter constraints.
    * Calibrates fast — 5 params, well-behaved nonlinear least squares.

Interview hooks:
    * "What's the difference between SVI and SABR?" — SABR is a stoch
      vol MODEL, SVI is a smile PARAMETERISATION. Different jobs.
    * Gatheral's "SSVI" extension links slices across maturities for a
      full no-arb surface (we don't implement that here).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def svi_total_variance(k: np.ndarray, params: np.ndarray) -> np.ndarray:
    """w(k) given SVI params [a, b, rho, m, sigma]."""
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def svi_implied_vol(
    k: np.ndarray, T: float, params: np.ndarray
) -> np.ndarray:
    """Convert SVI total variance back to implied vol."""
    w = svi_total_variance(k, params)
    return np.sqrt(np.maximum(w, 1e-12) / T)


def fit_svi_slice(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    *,
    weights: np.ndarray | None = None,
) -> dict:
    """Least-squares fit of SVI to a single-maturity smile.

    Returns dict with: params [a, b, rho, m, sigma], rmse_in_vol, success.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    mask = np.isfinite(k) & np.isfinite(iv) & (iv > 0)
    k, iv = k[mask], iv[mask]
    if weights is None:
        weights = np.ones_like(k)
    else:
        weights = np.asarray(weights, dtype=float)[mask]

    w_target = iv**2 * T  # total variance — fit in variance space

    # Initial guess
    a0 = max(np.min(w_target), 1e-4)
    b0 = 0.1
    rho0 = -0.5         # equities: negative correlation w/ vol
    m0 = 0.0
    sig0 = 0.1
    x0 = np.array([a0, b0, rho0, m0, sig0])

    def loss(p):
        a, b, rho, m, sigma = p
        if b < 0 or sigma <= 0 or not (-0.999 < rho < 0.999):
            return 1e10
        w_model = svi_total_variance(k, p)
        # Penalise negative implied variance
        if np.any(w_model <= 0):
            return 1e10
        resid = (w_model - w_target) * weights
        return float(np.mean(resid**2))

    res = minimize(
        loss, x0, method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 5000},
    )
    params = res.x
    w_fit = svi_total_variance(k, params)
    iv_fit = np.sqrt(np.maximum(w_fit, 1e-12) / T)
    rmse_vol = float(np.sqrt(np.mean((iv_fit - iv) ** 2)))
    return {
        "params": params,
        "param_names": ("a", "b", "rho", "m", "sigma"),
        "rmse_in_vol": rmse_vol,
        "success": bool(res.success),
    }
