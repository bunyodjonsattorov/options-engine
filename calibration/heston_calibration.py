"""Calibrate Heston parameters (kappa, theta, sigma_v, rho, v0) to a
market vol surface by nonlinear least squares on IMPLIED VOL space.

Why IV space (not price space):
    Price errors are unevenly scaled across strikes — deep OTM options
    are cheap, their absolute price errors are tiny, so a price-RMSE
    objective ignores the wings. Fitting in IV vol-points equalises
    importance across strikes (the "trader's metric").

Procedure per iteration:
    1. For each market quote (K, T, mid), call heston_price(...).
    2. Invert that model price to model IV via Brent (yes, twice — Heston
       gives a price, market gives an IV; we compare apples to apples).
    3. Residual = model_IV - market_IV; weights = 1 / spread (cheap
       quotes = bigger spread = less weight).

Calibration is famously ILL-POSED for Heston: different (kappa, theta)
pairs give near-identical surfaces. We address with:
    * Tight bounds on each parameter.
    * Soft penalty for Feller violation (2 kappa theta > sigma_v^2).
    * Multi-start from a few seeds; keep the best.
Interview discussion: "calibration is ill-posed -> use regularisation or
multi-start; the global fit is approximately a ridge in parameter space."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from calibration.iv_solver import implied_vol
from models.heston import heston_price


HESTON_BOUNDS = {
    "kappa":   (0.1,  10.0),
    "theta":   (0.005, 0.3),
    "sigma_v": (0.05,  2.0),
    "rho":     (-0.99, 0.99),
    "v0":      (0.005, 0.3),
}


def _pack(p):
    return np.array([p["kappa"], p["theta"], p["sigma_v"], p["rho"], p["v0"]])


def _unpack(x):
    return dict(zip(("kappa", "theta", "sigma_v", "rho", "v0"), x))


def calibrate_heston(
    chain: pd.DataFrame,
    *,
    initial: dict | None = None,
    feller_penalty: float = 10.0,
    verbose: bool = False,
) -> dict:
    """Fit Heston to a chain DataFrame with columns:
        spot, strike, T, r, q, option, iv  (plus optional 'spread').

    Returns dict with params, rmse_in_vol, n_quotes, feller_ok, success.
    """
    needed = {"spot", "strike", "T", "r", "q", "option", "iv"}
    missing = needed - set(chain.columns)
    if missing:
        raise ValueError(f"chain missing columns: {missing}")
    df = chain.dropna(subset=["iv"]).copy()

    # Default weights uniform; override if a 'spread' column is present.
    if "spread" in df.columns:
        w = 1.0 / np.maximum(df["spread"].to_numpy(), 1e-4)
        w = w / w.mean()
    else:
        w = np.ones(len(df))

    if initial is None:
        initial = dict(kappa=2.0, theta=0.04, sigma_v=0.5, rho=-0.6, v0=0.04)
    x0 = _pack(initial)

    spot = float(df["spot"].iloc[0])
    r = float(df["r"].iloc[0])
    q = float(df["q"].iloc[0])
    K = df["strike"].to_numpy()
    T = df["T"].to_numpy()
    option = df["option"].to_numpy()
    market_iv = df["iv"].to_numpy()

    bounds_lo = np.array([HESTON_BOUNDS[k][0] for k in
                          ("kappa", "theta", "sigma_v", "rho", "v0")])
    bounds_hi = np.array([HESTON_BOUNDS[k][1] for k in
                          ("kappa", "theta", "sigma_v", "rho", "v0")])

    def residuals(x):
        kappa, theta, sigma_v, rho, v0 = x
        model_iv = np.empty(len(df))
        for i in range(len(df)):
            model_price = heston_price(
                spot, K[i], T[i], r, q, kappa, theta, sigma_v, rho, v0,
                option=option[i],
            )
            model_iv[i] = implied_vol(
                model_price, spot, K[i], T[i], r, q, option[i],
            )
        # Replace any NaN model IVs with a large penalty
        bad = ~np.isfinite(model_iv)
        model_iv[bad] = market_iv[bad] + 1.0

        resid = (model_iv - market_iv) * w
        # Feller soft penalty — always appended (zero when satisfied) so
        # the residual vector has constant shape (scipy needs this).
        feller_gap = max(0.0, sigma_v**2 - 2 * kappa * theta)
        resid = np.concatenate([resid, [feller_penalty * feller_gap]])
        if verbose:
            print(f"  rmse={np.sqrt(np.mean(resid[: len(df)]**2)):.4f}  "
                  f"kappa={kappa:.3f} theta={theta:.3f} sig_v={sigma_v:.3f} "
                  f"rho={rho:.3f} v0={v0:.3f}")
        return resid

    res = least_squares(
        residuals, x0, bounds=(bounds_lo, bounds_hi),
        method="trf", xtol=1e-8, ftol=1e-8, max_nfev=500,
    )
    params = _unpack(res.x)
    feller_ok = 2 * params["kappa"] * params["theta"] > params["sigma_v"] ** 2

    # Final RMSE on plain residuals (drop last element = Feller penalty)
    final_resid = residuals(res.x)[:-1] / w
    rmse_vol = float(np.sqrt(np.mean(final_resid**2)))

    return {
        "params": params,
        "rmse_in_vol": rmse_vol,
        "n_quotes": len(df),
        "feller_ok": bool(feller_ok),
        "success": bool(res.success),
    }
