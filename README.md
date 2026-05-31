# options-engine

From-scratch Python implementation of the core machinery a derivatives desk uses to price, risk, and hedge options — no QuantLib, no py_vollib. Every pricer is cross-validated against at least one independent method; 17 tests, all passing.

## What's implemented

| Module | What it does |
|---|---|
| `pricers/black_scholes.py` | Closed-form European call/put + all 5 Greeks (Δ, Γ, V, Θ, ρ) |
| `pricers/monte_carlo.py` | Vectorised GBM, antithetic variates, control variates, Asian / barrier / lookback |
| `pricers/binomial.py` | CRR tree, American backward induction, early-exercise boundary |
| `pricers/longstaff_schwartz.py` | LSM regression-based American MC (lower bound) |
| `models/heston.py` | Characteristic function via Little Heston Trap + Carr-Madan integration |
| `calibration/iv_solver.py` | Brent's method implied vol solver |
| `calibration/svi.py` | Gatheral SVI smile parameterisation |
| `calibration/heston_calibration.py` | Least-squares calibration in IV space |
| `greeks/finite_difference.py` | Central-difference Greeks (model-agnostic) |
| `greeks/pathwise.py` | Pathwise and likelihood-ratio MC Greeks |
| `strategies/delta_hedge.py` | Short-straddle simulator + full PnL decomposition |

## Quick start

```bash
git clone https://github.com/bunyodjonsattorov/options-engine
cd options-engine
pip install numpy scipy pandas matplotlib

python3 examples/01_black_scholes.py
python3 examples/02_mc_convergence.py
python3 examples/03_american_options.py
python3 examples/04_vol_surface.py
python3 examples/05_delta_hedge_backtest.py
```

Tests (no pytest required):

```bash
python3 -c "
import sys; sys.path.insert(0,'.')
import tests.test_pricers as T
for n in sorted(dir(T)):
    if n.startswith('test_'):
        try: getattr(T,n)(); print(f'PASS  {n}')
        except Exception as e: print(f'FAIL  {n}: {e}')
"
```

---

## Results

### 1. Black-Scholes — analytical vs finite-difference Greeks

ATM call (S=K=100, T=3m, σ=20%, r=5%):

```
Call price : 4.6150
Put price  : 3.3728
Put-call parity residual: -7.11e-15  (~machine zero)

Greek      analytic      fin-diff     abs err
delta      0.569460      0.569460    1.80e-08
gamma      0.039288      0.039288    2.74e-09
vega      19.644000     19.644000    4.03e-08
theta    -10.474151    -10.474271    1.20e-04
rho       13.082755     13.082755    2.67e-08
```

> Vega is reported per unit vol (×0.01 for per-1%-vol-point). Theta is per year (÷365 for per-day). Conventions are derived and explained in the module docstring.

---

### 2. Monte Carlo convergence and variance reduction

European call at S=K=100, T=6m, σ=20%, r=5% (BS = 6.8887):

**Convergence — error halves when N quadruples (1/√N):**

| N | MC price | 1.96·SE | error vs BS |
|---|---|---|---|
| 1,000 | 6.3245 | 0.5807 | −0.5642 |
| 10,000 | 6.7930 | 0.1902 | −0.0958 |
| 100,000 | 6.8260 | 0.0605 | −0.0628 |
| 1,000,000 | 6.8728 | 0.0192 | −0.0159 |

**Variance reduction (N = 100k):**

| Method | SE | Variance cut |
|---|---|---|
| Plain MC | 0.0309 | 1× |
| Antithetic variates | 0.0220 | ~2× |
| Control variate on S_T | 0.0130 | ~5.6× |
| Geometric-Asian CV (arithmetic Asian) | 0.0009 | **~2,700×** |

Antithetic variates require the payoff to be monotonic in the Brownian draw — true for vanilla calls/puts, not guaranteed for all exotics. The geometric-Asian control variate works because the geometric average of GBM is itself log-normal (closed form exists) and correlates ~0.99 with the arithmetic average.

**Pathwise vs likelihood-ratio Greeks** (delta, N=200k):

```
Delta BS (true)  : 0.597734
Delta pathwise   : 0.597337 ± 0.00243
Delta LR         : 0.597617 ± 0.00598   (higher SE — LR noisier when both apply)
LR delta digital : 0.027349 ± 0.00018   (pathwise can't differentiate I(S_T > K); LR can)
```

---

### 3. American options

Binomial CRR convergence to BS (European put, T=1y, σ=20%, r=6%):

| n steps | binomial | error vs BS |
|---|---|---|
| 50 | 5.1258 | −4.02e-02 |
| 100 | 5.1459 | −2.01e-02 |
| 500 | 5.1620 | −4.03e-03 |
| 1,000 | 5.1640 | −2.01e-03 |
| 5,000 | 5.1656 | −4.03e-04 |

American vs European (same params):

| | Price | vs European |
|---|---|---|
| European put (BS) | 5.1660 | — |
| American put (binomial n=1000) | 5.7982 | **+0.6322 early-exercise premium** |
| American call (binomial n=1000) | 10.9875 | = European (Merton's theorem, q=0) |
| LSM American put | 5.740 ± 0.043 | matches binomial within MC error |

Early-exercise boundary rises toward K as t → T (classic shape):

```
t = 0.20y   S*(t) = 82.04
t = 0.40y   S*(t) = 84.39
t = 0.60y   S*(t) = 84.39
t = 0.80y   S*(t) = 86.81
t = 0.98y   S*(t) = 93.17
```

---

### 4. Implied vol, SVI surface, Heston calibration

**SVI smile fit per maturity slice:**

| Maturity | Quotes | RMSE | a | b | ρ |
|---|---|---|---|---|---|
| 1m | 21 | 0.119% | 0.0074 | 0.0063 | −0.999 |
| 3m | 21 | 0.031% | −0.1981 | 0.1026 | −0.162 |
| 6m | 21 | 0.029% | −0.2355 | 0.1474 | −0.191 |
| 12m | 21 | 0.029% | −0.5482 | 0.2955 | −0.176 |

Negative ρ across all tenors is the equity leverage effect — vol rises as the stock falls.

**Heston calibration** fits κ, θ, σᵥ, ρ, v₀ to the full surface via nonlinear least squares *in IV space* — price-space errors are unevenly scaled across strikes, so IV-space fitting is the standard in practice.

```
fit RMSE : 0.781% vol
kappa    = 10.000   (hits upper bound — expected; calibration is ill-posed)
theta    = 0.0437   (~20.9% long-run vol)
sigma_v  = 0.775    (vol of vol)
rho      = −0.309   (price-vol correlation; negative = equity leverage)
v0       = 0.1256   (~35.4% initial vol)
Feller:  True       (2κθ > σᵥ² — variance process stays positive a.s.)
```

Calibration is ill-posed: many (κ, θ) pairs produce near-identical surfaces. κ hitting its upper bound is the canonical symptom. Fixes in production: multi-start optimisation, fix κ from the term structure, or regularise toward a prior.

---

### 5. Delta-hedged short straddle — PnL decomposition

The central market-maker identity (from Ito + Black-Scholes PDE):

```
dPnL ≈ ½ · Γ_pos · [(ΔS)² − σ_imp² · S² · dt]  +  Vega_pos · dσ_imp  −  tx costs
```

Delta-hedging strips out direction; what remains is purely a bet on realized vol vs implied vol.

**1,000 simulated 30-day paths:**

| Vol regime | Mean PnL | Std | Win rate |
|---|---|---|---|
| Realized = implied (20%) | +0.03 | 0.70 | 51.9% |
| Sold rich — imp 25%, real 15% | **+2.29** | 0.82 | **100%** |
| Sold cheap — imp 15%, real 25% | **−2.21** | 1.34 | **0%** |
| Rich + 5bp half-spread | +2.12 | 0.82 | 100% |
| Rich + $0.01/share tx cost | +2.27 | 0.82 | 100% |

**Realized vs implied sweep** (σ_imp fixed at 20%):

| σ_real | Mean PnL | Win rate |
|---|---|---|
| 10% | +2.28 | 100% |
| 15% | +1.15 | 97.6% |
| 20% | +0.01 | 51.4% |
| 25% | −1.13 | 7.6% |
| 30% | −2.28 | 0.6% |

**Single-path PnL decomposition** (vol rich, seed=42):

```
total final PnL                = +4.2566
  gamma component (vs implied) = +4.0811
  vega component (IV moves)    =  0.0000   (flat IV by construction)
  transaction costs            =  0.0000
  residual (discrete hedge)    = +0.1755
  --- sum ---                  = +4.2566   ✓
```

The four components sum to the bookkeeping total exactly (verified in `test_delta_hedge_decomposition_sums_to_total`).

---

## Out of scope (by design)

- **Jump diffusion / SABR / rough vol** — possible extensions
- **American MC upper bounds** — LSM gives a lower bound only; the Andersen-Broadie dual approach gives an upper bound
- **Production FFT pricing** — Carr-Madan is implemented per-strike via `scipy.quad`; replacing with FFT would price a whole surface in one call
- **Smile-arbitrage checks** — SVI fits don't guarantee no-arb without additional constraints (eSSVI or similar)
- **GPU MC** — `simulate_gbm_paths` is the hot path; architecture supports a drop-in numpy → cupy swap

## Dependencies

`numpy`, `scipy`, `pandas`, `matplotlib` — optionally `yfinance` for real option chains (a synthetic chain is provided as fallback so all examples run offline).
