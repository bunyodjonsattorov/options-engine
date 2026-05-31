"""Monte Carlo pricing: vanilla European, exotics, with variance reduction.

Key results to remember for interview:
    * Plain MC error ~ sigma_payoff / sqrt(N). To halve the error, 4x the
      paths. This is brutal — variance reduction is how you get usable MC.
    * Antithetic variates: for each draw Z also use -Z. Cuts variance iff
      payoff is monotonic in Z (true for vanilla call/put). For symmetric
      payoffs (e.g. straddle around ATM) it can do nothing or even hurt.
    * Control variate: estimator
            V_hat_cv = mean(V_i) - beta * (mean(C_i) - E[C])
      where C is a related r.v. with known mean and beta = Cov(V,C)/Var(C).
      For an arithmetic Asian, the GEOMETRIC Asian has a closed form and
      correlates ~0.99 with the arithmetic version => massive variance cut.

All estimators return (price, standard_error) so you can build CIs:
    95% CI ~ price +/- 1.96 * standard_error.
"""

from __future__ import annotations

import numpy as np

from models.gbm import simulate_gbm_paths, simulate_gbm_terminal
from pricers.black_scholes import bs_price


# ----------------------------- helpers --------------------------------- #
def _stderr(samples: np.ndarray) -> float:
    """Standard error of the mean of a 1D sample (unbiased sample std / sqrt(N))."""
    return float(np.std(samples, ddof=1) / np.sqrt(samples.size))


def _european_payoff(S_T: np.ndarray, K: float, option: str) -> np.ndarray:
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    if option == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError("option must be 'call' or 'put'")


# ----------------------------- European MC ----------------------------- #
def mc_european(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    n_paths: int = 100_000,
    *,
    antithetic: bool = False,
    seed: int | None = None,
) -> tuple[float, float]:
    """European option price by MC. Returns (price, stderr).

    When antithetic=True we report stderr based on PAIR averages
    (n_paths/2 observations), not individual paths — the two halves of a
    pair are negatively correlated, so treating them as independent
    samples would overstate the variance.
    """
    rng = np.random.default_rng(seed)
    S_T, _ = simulate_gbm_terminal(S0, r, q, sigma, T, n_paths,
                                   antithetic=antithetic, rng=rng)
    payoff = _european_payoff(S_T, K, option)
    discounted = np.exp(-r * T) * payoff

    if antithetic:
        half = n_paths // 2
        pair_avg = 0.5 * (discounted[:half] + discounted[half:])
        return float(pair_avg.mean()), _stderr(pair_avg)
    return float(discounted.mean()), _stderr(discounted)


def mc_european_with_cv(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    n_paths: int = 100_000,
    seed: int | None = None,
) -> tuple[float, float]:
    """European MC with the underlying S_T itself as a control variate.

    Why this works: we know E^Q[e^{-rT} S_T] = S_0 e^{-qT}  (martingale).
    So C_i = e^{-rT} S_T,i has a known mean and is correlated with the
    discounted payoff. Optimal beta = Cov(V, C) / Var(C).

    Variance reduction here is modest (S_T is not THAT correlated with the
    call payoff once you strip out the indicator), but the same machinery
    is exactly how we'll handle the Asian / Heston cases below.
    """
    rng = np.random.default_rng(seed)
    S_T, _ = simulate_gbm_terminal(S0, r, q, sigma, T, n_paths, rng=rng)
    payoff = _european_payoff(S_T, K, option)
    V = np.exp(-r * T) * payoff
    C = np.exp(-r * T) * S_T
    EC = S0 * np.exp(-q * T)

    cov = np.cov(V, C, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    V_cv = V - beta * (C - EC)
    return float(V_cv.mean()), _stderr(V_cv)


# ----------------------------- Asian options --------------------------- #
def _geometric_asian_bs(
    S0, K, T, r, sigma, q, n_avg, option
) -> float:
    """Closed-form price of a DISCRETE-monitored geometric-average Asian.

    Geometric Asian under GBM is itself log-normal, so it reduces to a
    Black-Scholes call/put with adjusted vol and drift. We use this as
    the analytic mean for the control variate in arithmetic-Asian MC.

    Sigma_G  = sigma * sqrt( (n+1)(2n+1) / (6 n^2) )
    mu_G     = (r - q - 0.5 sigma^2)*(n+1)/(2n) + 0.5 Sigma_G^2
    The 'effective' BS with vol Sigma_G and rate r* such that
    exp(-rT) S_0 exp(mu_G T) = "forward of geometric average".
    """
    n = n_avg
    sig_g = sigma * np.sqrt((n + 1.0) * (2 * n + 1.0) / (6.0 * n * n))
    # drift of the log of the geometric average
    mu_g = (r - q - 0.5 * sigma**2) * (n + 1.0) / (2.0 * n) + 0.5 * sig_g**2
    # treat as BS with adjusted drift => synthetic dividend yield
    q_eff = r - mu_g
    return float(bs_price(S0, K, T, r, sig_g, q_eff, option))


def mc_asian_arithmetic(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option: str = "call",
    n_paths: int = 50_000,
    n_steps: int = 50,
    *,
    antithetic: bool = True,
    control_variate: bool = True,
    seed: int | None = None,
) -> tuple[float, float]:
    """Arithmetic-average Asian, optionally with geometric-Asian CV.

    With CV on, expect variance reduction of ~100-1000x vs plain MC for
    typical parameters. This is the canonical example interviewers love.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_paths, n_steps,
                               antithetic=antithetic, rng=rng)
    avg_path = paths[:, 1:]  # exclude S0 from averaging
    arith = avg_path.mean(axis=1)
    payoff_arith = _european_payoff(arith, K, option)
    V = np.exp(-r * T) * payoff_arith

    def _collapse(x):
        """Average antithetic pairs into single observations for stderr."""
        if not antithetic:
            return x
        half = n_paths // 2
        return 0.5 * (x[:half] + x[half:])

    if not control_variate:
        Vc = _collapse(V)
        return float(Vc.mean()), _stderr(Vc)

    # geometric average and its analytic price as CV
    log_avg = np.log(avg_path).mean(axis=1)
    geo = np.exp(log_avg)
    payoff_geo = _european_payoff(geo, K, option)
    C = np.exp(-r * T) * payoff_geo
    EC = _geometric_asian_bs(S0, K, T, r, sigma, q, n_steps, option)

    cov = np.cov(V, C, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    V_cv = V - beta * (C - EC)
    V_cv = _collapse(V_cv)
    return float(V_cv.mean()), _stderr(V_cv)


# ----------------------------- Barrier (up-and-out) -------------------- #
def mc_barrier_up_and_out_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    B: float,
    q: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    *,
    rebate: float = 0.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Up-and-out European call: payoff = max(S_T - K, 0) * I(max_t S_t < B),
    plus a rebate paid at hit-time if knocked out (we discount it at T for
    simplicity — production code would discount at the actual hit time).

    Discrete monitoring => price strictly LESS than continuous-monitored
    barrier (more hits caught with continuous). For very high n_steps the
    two converge; the gap is the well-known 'discrete barrier correction'.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_paths, n_steps, rng=rng)
    max_path = paths.max(axis=1)
    S_T = paths[:, -1]
    alive = max_path < B
    payoff = np.where(alive, np.maximum(S_T - K, 0.0), rebate)
    discounted = np.exp(-r * T) * payoff
    return float(discounted.mean()), _stderr(discounted)


# ----------------------------- Lookback (floating strike) -------------- #
def mc_lookback_floating_call(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_paths: int = 50_000,
    n_steps: int = 252,
    *,
    seed: int | None = None,
) -> tuple[float, float]:
    """Floating-strike lookback call: payoff = S_T - min_t S_t.

    The optimal hindsight trade — always positive, never worthless => price
    >> ATM vanilla call. Useful intuition pump: max value of optionality.
    """
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(S0, r, q, sigma, T, n_paths, n_steps, rng=rng)
    min_path = paths.min(axis=1)
    payoff = paths[:, -1] - min_path
    discounted = np.exp(-r * T) * payoff
    return float(discounted.mean()), _stderr(discounted)
