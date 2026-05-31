# Options Pricing & Trading Engine

A from-scratch Python implementation of the core machinery a derivatives
desk uses to price, risk, and hedge options. Built deliberately without
QuantLib / py_vollib — every primitive (Black-Scholes, Monte Carlo,
binomial tree, Longstaff-Schwartz, implied vol solver, SVI smile fit,
Heston characteristic function + calibration, delta-hedge backtest) is
implemented and tested here.

## Why this exists

A portfolio project that survives a quantitative-trading interview's
deep dive: each component has a one-page derivation in its module
docstring, every claim is cross-verified by an independent pricer, and
the delta-hedge backtest reproduces the gamma-scalping P&L formula
exactly to within numerical residual.

## Quick start

```bash
git clone <this repo>
cd options-engine
pip install -r requirements.txt    # or:  pip install numpy scipy pandas matplotlib

# Run all examples (each one is standalone):
python3 examples/01_black_scholes.py
python3 examples/02_mc_convergence.py
python3 examples/03_american_options.py
python3 examples/04_vol_surface.py
python3 examples/05_delta_hedge_backtest.py

# Run the test suite (17 tests, no pytest required):
python3 -c "
import sys; sys.path.insert(0,'.')
import tests.test_pricers as T
for n in sorted(dir(T)):
    if n.startswith('test_'):
        try: getattr(T, n)(); print(f'PASS  {n}')
        except Exception as e: print(f'FAIL  {n}: {e}')
"
```

## Project layout

```
options-engine/
├── pricers/
│   ├── black_scholes.py        # Closed-form European pricing + all 5 Greeks
│   ├── monte_carlo.py          # MC + antithetic + control variate + exotics
│   ├── binomial.py             # CRR tree + American backward induction
│   └── longstaff_schwartz.py   # LSM regression for American MC
├── models/
│   ├── gbm.py                  # Exact-step GBM path simulators
│   └── heston.py               # Heston char fn (Little Heston Trap) + Carr-Madan
├── greeks/
│   ├── finite_difference.py    # Central-difference Greeks (model-agnostic)
│   └── pathwise.py             # Pathwise + likelihood-ratio MC Greeks
├── calibration/
│   ├── iv_solver.py            # Brent's method implied vol
│   ├── svi.py                  # Gatheral SVI smile parameterisation
│   └── heston_calibration.py   # Least-squares calibration in IV space
├── strategies/
│   └── delta_hedge.py          # Short-straddle simulator + PnL decomposition
├── data/
│   └── market_data.py          # yfinance wrapper + synthetic-chain fallback
├── tests/test_pricers.py       # 17 cross-validation tests
└── examples/                   # Runnable demos per phase
```

## What's in each phase

### 1. Black-Scholes & analytical Greeks

European call/put under GBM, plus Δ, Γ, Vega, Θ, ρ — all derived in
the module docstring, all verified against central-difference Greeks
and put-call parity.

```python
from pricers.black_scholes import bs_price, bs_greeks
call = bs_price(S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.0)
g = bs_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.20, q=0.0, option="call")
# call = 4.6150;  delta = 0.5695, gamma = 0.0397, vega = 0.1985, ...
```

### 2. Monte Carlo + variance reduction + exotics

Vectorised GBM, antithetic variates, control variates, and the exotic
options (Asian / barrier / lookback) where MC is actually necessary.
Plus pathwise vs likelihood-ratio Greeks — including the LR delta of a
digital, the textbook case where pathwise fails.

Convergence demonstrated:

| N            | MC price  | 1.96·SE  | error vs BS |
|--------------|-----------|----------|-------------|
| 1,000        | 6.45      | 0.45     | -0.36       |
| 10,000       | 6.84      | 0.15     | +0.03       |
| 100,000      | 6.79      | 0.046    | -0.02       |
| 1,000,000    | 6.81      | 0.015    | -0.00       |

(error halves when N quadruples — the textbook 1/√N rate)

Variance reduction:

| Method                    | SE       | Variance cut |
|---------------------------|----------|--------------|
| Plain MC                  | 0.020    | 1×           |
| Antithetic variates       | 0.014    | 2×           |
| Control variate on S_T    | 0.009    | 5×           |
| Geometric-Asian CV (arith Asian) | 0.0004 | **5,500×**  |

### 3. Binomial tree + Longstaff-Schwartz American

CRR tree for European + American options. Convergence to BS verified.
American put > European put (early exercise premium); American call =
European call when q=0 (Merton's theorem). LSM MC matches binomial
American put within MC stderr.

### 4. Implied vol, SVI surface, Heston calibration

* Brent's method IV solver (handles arbitrage bounds).
* SVI parameterisation fits a smile slice to <0.1% vol.
* Heston char fn in "Little Heston Trap" form (avoids branch cuts).
* Carr-Madan damped-call integral for European pricing.
* Least-squares calibration in IV space (the trader's metric).

Calibration is famously ill-posed for Heston — many `(kappa, theta)`
pairs give nearly identical surfaces. The example flags this explicitly:
`kappa` typically pins at its upper bound, which is a great talking
point on regularisation and multi-start methods.

### 5. Delta-hedged short straddle + PnL decomposition

Implements the core market-maker P&L identity
`dPnL ≈ ½·Γ·[(ΔS)² − σ_imp²·S²·dt]`. Decomposes total PnL into
gamma / vega / transaction-cost / residual buckets and shows them
summing to the bookkeeping total exactly. Result on 1,000 simulated paths:

| Vol regime                        | mean PnL | std   | win rate |
|-----------------------------------|----------|-------|----------|
| realized = implied (20%)          | +0.03    | 0.70  | 51.9%    |
| sold rich (imp 25%, real 15%)     | **+2.29** | 0.82 | **100%** |
| sold cheap (imp 15%, real 25%)    | **-2.21** | 1.34 | **0%**   |

This is the *entire trade thesis* of short-vol strategies, isolated.

## Tests

17 cross-validation tests covering every phase. All pass:

```
test_analytical_vs_fd_greeks_call      test_iv_round_trip
test_analytical_vs_fd_greeks_put       test_lsm_matches_binomial_american_put
test_american_call_equals_european_when_no_dividend
test_american_put_premium_over_european
test_binomial_converges_to_bs          test_mc_antithetic_cuts_variance
test_call_delta_bounds                 test_mc_european_within_3se_of_bs
test_delta_hedge_decomposition_sums_to_total
test_delta_hedge_realized_equals_implied_zero_mean
test_heston_bs_limit                   test_pathwise_delta_matches_bs
test_intrinsic_at_expiry               test_put_call_parity
test_vectorised_pricing
```

## What I learned (defensible in interview)

* **Stochastic calculus essentials**: Ito's lemma, GBM SDE, BS PDE
  derivation, risk-neutral measure change.
* **Numerical methods**: Brent root-finding, least-squares calibration,
  backward induction, regression-based LSM, complex-integration for
  Heston via Carr-Madan.
* **Monte Carlo theory**: error ~ σ/√N, antithetic/control-variate
  conditions, pathwise vs likelihood-ratio Greeks and when each breaks.
* **Vol surface reality**: implied vol is the "wrong number in the wrong
  formula"; equity skew comes from leverage effect; SVI handles wings.
* **Trading PnL**: gamma scalping formula, why short vol pays for being
  short tails, transaction-cost drag, ill-posed calibration.

## Things explicitly out of scope (for honesty)

* Jump diffusion (Merton), SABR, rough vol, GPU MC — possible extensions.
* American MC duality bounds (Andersen-Broadie) — LSM here gives a
  lower bound only.
* Production-grade FFT pricing — Carr-Madan is implemented via
  scipy.quad per strike, fast enough for ~50 strikes; swapping in
  scipy.fft would price a whole surface in one shot.
* Smile-arbitrage checks (butterfly + calendar) — SVI fits don't
  guarantee no-arb without additional constraints; production code
  needs eSSVI or similar.

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib`, optionally `yfinance` (for
real chains; a synthetic chain is provided as a fallback so all
examples run offline).
