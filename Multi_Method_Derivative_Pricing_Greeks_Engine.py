"""
================================================================================
Project I-2: Multi-Method Derivative Pricing and Greeks Engine
================================================================================

This module implements a complete, self-contained option pricing engine with
three independent numerical methods that triangulate against one another:

    1. Black-Scholes-Merton (BSM) closed-form pricing with continuous
       dividend yield, together with the full set of analytic Greeks
       (delta, gamma, vega, theta, rho).
    2. Cox-Ross-Rubinstein (CRR) binomial trees supporting both European
       and American exercise.
    3. Risk-neutral Monte Carlo simulation under geometric Brownian motion
       (GBM) with antithetic variates and a European-call control variate.

It also provides an implied volatility solver that combines Brent's
bracketed root finding with a Newton-Raphson polish step, a triangulation
harness across moneyness, maturity, and volatility grids, optional QuantLib
benchmarking (soft dependency only), and a battery of correctness checks:

    * put-call parity to machine precision,
    * American option value >= European option value,
    * analytic Greeks agree with central finite differences,
    * the delta parity identity delta(call) - delta(put) = e^{-qT},
    * Monte Carlo error decays like N^(-1/2),
    * implied volatility round-trip recovers the input volatility,
    * Monte Carlo confidence bands contain the BSM price roughly 95% of
      the time.

It also builds an implied volatility smile, preferring a live option chain
(via yfinance) with a clearly labeled synthetic fallback when no market data
connection is available.

All pricing, Greeks, and correctness-check outputs in the demo use clearly
labeled SYNTHETIC market parameters; no live market data is required for
any of them. The implied volatility smile (Section 9) attempts a live
option chain via yfinance and falls back to an explicitly labeled synthetic
chain if no network connection is available. Nothing in the demo should be
interpreted as a real-world trading signal.

Author: Quantitative Finance Project Series (Project I-2)
License: MIT
================================================================================
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Matplotlib is used only for the demo plots; a non-interactive backend keeps
# the script safe to run on headless machines (servers, CI, notebooks).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# QuantLib is an OPTIONAL benchmark. If it is not installed, every feature of
# this engine still works; the comparison section is simply skipped.
try:
    import QuantLib as ql  # type: ignore
    HAS_QUANTLIB = True
except ImportError:
    HAS_QUANTLIB = False


# ==============================================================================
# Section 1: Option specification (single interface consumed by every pricer)
# ==============================================================================

@dataclass
class OptionSpec:
    """A single, validated specification object shared by every pricer.

    Parameters
    ----------
    S0 : float
        Spot price of the underlying asset. Must be positive.
    K : float
        Strike price of the option. Must be positive.
    T : float
        Time to maturity in years. Must be strictly positive.
    r : float
        Continuously compounded risk-free rate (may be negative).
    sigma : float
        Annualized volatility of the underlying. Must be strictly positive.
    q : float
        Continuously compounded dividend yield. Defaults to 0.
    option_type : str
        Either "call" or "put" (case-insensitive).
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: str = "call"

    def __post_init__(self) -> None:
        # ---- Validation: reject financially meaningless inputs up front. ----
        if self.S0 <= 0:
            raise ValueError("Spot price S0 must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price K must be positive.")
        if self.T <= 0:
            raise ValueError("Time to maturity T must be strictly positive.")
        if self.sigma <= 0:
            raise ValueError("Volatility sigma must be strictly positive.")
        ot = self.option_type.lower()
        if ot not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        self.option_type = ot

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    def payoff(self, S_T: np.ndarray) -> np.ndarray:
        """Terminal payoff vectorized over an array of terminal prices."""
        if self.is_call:
            return np.maximum(S_T - self.K, 0.0)
        return np.maximum(self.K - S_T, 0.0)

    def intrinsic(self, S: float) -> float:
        """Intrinsic (early-exercise) value at spot level S."""
        if self.is_call:
            return max(S - self.K, 0.0)
        return max(self.K - S, 0.0)


# ==============================================================================
# Section 2: Black-Scholes-Merton analytic pricer and Greeks
# ==============================================================================

def _d1_d2(spec: OptionSpec) -> Tuple[float, float]:
    """Compute the d1 and d2 terms of the BSM formula with dividend yield."""
    sqrt_T = math.sqrt(spec.T)
    d1 = (math.log(spec.S0 / spec.K)
          + (spec.r - spec.q + 0.5 * spec.sigma ** 2) * spec.T) / (spec.sigma * sqrt_T)
    d2 = d1 - spec.sigma * sqrt_T
    return d1, d2


def bs_price(spec: OptionSpec) -> float:
    """Black-Scholes-Merton price with continuous dividend yield q.

    C = S0 e^{-qT} N(d1) - K e^{-rT} N(d2)
    P = K e^{-rT} N(-d2) - S0 e^{-qT} N(-d1)
    """
    d1, d2 = _d1_d2(spec)
    disc_r = math.exp(-spec.r * spec.T)
    disc_q = math.exp(-spec.q * spec.T)
    if spec.is_call:
        price = spec.S0 * disc_q * norm.cdf(d1) - spec.K * disc_r * norm.cdf(d2)
    else:
        price = spec.K * disc_r * norm.cdf(-d2) - spec.S0 * disc_q * norm.cdf(-d1)
    # Clamp tiny negative values arising from floating-point round-off; a
    # negative option price is financially impossible.
    return max(price, 0.0)


def bs_greeks(spec: OptionSpec) -> Dict[str, float]:
    """Analytic Greeks for the BSM model with continuous dividend yield.

    Returns a dict with delta, gamma, vega, theta, rho. Vega and rho are
    quoted per 1.00 unit move in sigma and r respectively; divide by 100 for
    per-vol-point or per-percent conventions if desired.
    """
    d1, d2 = _d1_d2(spec)
    S0, K, T, r, q, sig = spec.S0, spec.K, spec.T, spec.r, spec.q, spec.sigma
    sqrt_T = math.sqrt(T)
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = norm.pdf(d1)

    # Gamma and vega are identical for calls and puts.
    gamma = disc_q * pdf_d1 / (S0 * sig * sqrt_T)
    vega = S0 * disc_q * pdf_d1 * sqrt_T

    if spec.is_call:
        delta = disc_q * norm.cdf(d1)
        theta = (-(S0 * disc_q * pdf_d1 * sig) / (2.0 * sqrt_T)
                 - r * K * disc_r * norm.cdf(d2)
                 + q * S0 * disc_q * norm.cdf(d1))
        rho = K * T * disc_r * norm.cdf(d2)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (-(S0 * disc_q * pdf_d1 * sig) / (2.0 * sqrt_T)
                 + r * K * disc_r * norm.cdf(-d2)
                 - q * S0 * disc_q * norm.cdf(-d1))
        rho = -K * T * disc_r * norm.cdf(-d2)

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


def fd_greeks(spec: OptionSpec, pricer: Callable[[OptionSpec], float] = bs_price,
              rel_eps: float = 1e-4) -> Dict[str, float]:
    """Central finite-difference Greeks, used to validate the analytic ones.

    Central differences have O(eps^2) truncation error; a relative bump of
    1e-4 balances truncation error against floating-point cancellation.
    """
    def bump(field_name: str, signed_eps: float) -> OptionSpec:
        return OptionSpec(**{**spec.__dict__, field_name:
                             getattr(spec, field_name) + signed_eps})

    dS = rel_eps * spec.S0
    dSig = rel_eps * spec.sigma
    dr = rel_eps * max(abs(spec.r), 0.01)
    dT = 1.0 / 365.0 / 24.0  # roughly one hour for a stable theta difference

    up_S, dn_S = bump("S0", dS), bump("S0", -dS)
    delta = (pricer(up_S) - pricer(dn_S)) / (2.0 * dS)
    gamma = (pricer(up_S) - 2.0 * pricer(spec) + pricer(dn_S)) / (dS ** 2)

    up_v, dn_v = bump("sigma", dSig), bump("sigma", -dSig)
    vega = (pricer(up_v) - pricer(dn_v)) / (2.0 * dSig)

    # Theta is dV/dt with t = calendar time, so dV/dT enters with a minus sign.
    up_T = OptionSpec(**{**spec.__dict__, "T": spec.T + dT})
    dn_T = OptionSpec(**{**spec.__dict__, "T": spec.T - dT})
    theta = -(pricer(up_T) - pricer(dn_T)) / (2.0 * dT)

    up_r, dn_r = bump("r", dr), bump("r", -dr)
    rho = (pricer(up_r) - pricer(dn_r)) / (2.0 * dr)

    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}


# ==============================================================================
# Section 3: Cox-Ross-Rubinstein binomial tree (European and American)
# ==============================================================================

def crr_price(spec: OptionSpec, steps: int = 500, american: bool = False) -> float:
    """Price an option on a Cox-Ross-Rubinstein recombining binomial tree.

    Parameters
    ----------
    spec : OptionSpec
        The option specification.
    steps : int
        Number of time steps in the tree. More steps reduce discretization
        error at the cost of O(steps) memory and time in this vectorized
        backward induction.
    american : bool
        If True, early exercise is checked at every node (American style).
        If False, the tree computes the European price.

    Returns
    -------
    float
        The option price, clamped to be nonnegative.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1.")

    dt = spec.T / steps
    u = math.exp(spec.sigma * math.sqrt(dt))   # up factor
    d = 1.0 / u                                 # down factor (CRR choice)
    disc = math.exp(-spec.r * dt)               # one-period discount factor
    # Risk-neutral up probability under continuous dividend yield q.
    growth = math.exp((spec.r - spec.q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "Risk-neutral probability fell outside [0, 1]; increase steps or "
            "check that sigma*sqrt(dt) is large enough relative to (r-q)*dt.")

    # Terminal asset prices S0 * u^j * d^(steps-j), j = 0..steps.
    j = np.arange(steps + 1)
    S_T = spec.S0 * (u ** j) * (d ** (steps - j))
    values = spec.payoff(S_T)

    # Vectorized backward induction through the tree.
    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            # Reconstruct the asset price at every node of this time slice.
            jj = np.arange(step + 1)
            S_node = spec.S0 * (u ** jj) * (d ** (step - jj))
            if spec.is_call:
                exercise = np.maximum(S_node - spec.K, 0.0)
            else:
                exercise = np.maximum(spec.K - S_node, 0.0)
            # American rule: never let continuation value drop below the
            # value of exercising immediately.
            values = np.maximum(values, exercise)

    return max(float(values[0]), 0.0)


# ==============================================================================
# Section 4: Monte Carlo pricer with antithetic variates and control variate
# ==============================================================================

def mc_price(spec: OptionSpec, n_paths: int = 100_000, seed: Optional[int] = None,
             antithetic: bool = True, control_variate: bool = True
             ) -> Dict[str, float]:
    """Price a European option by Monte Carlo under risk-neutral GBM.

    Variance reduction stack:
      * Antithetic variates: pair each draw Z with -Z, which makes the two
        simulated terminal prices negatively correlated and cancels the
        first-order sampling noise. The unit of observation for statistics
        (mean, variance, standard error) is the PAIR average, not the
        individual path, because pooling the two halves of a pair as if
        they were independent draws would misstate both the standard error
        and, when a control variate is layered on top, the covariance used
        to estimate the optimal control coefficient.
      * Control variate: use the discounted payoff of the European CALL with
        the same strike as a control, because its risk-neutral expectation is
        known exactly from the BSM formula. The control-adjusted estimator is
            Y_cv = Y - c * (X - E[X]),
        where X is the call payoff sample (paired with Y when antithetic is
        on) and E[X] is the BSM call price, with the optimal coefficient
        c* = Cov(Y, X) / Var(X) estimated from the pilot sample itself.

    Returns
    -------
    dict with keys: price, stderr, ci_low, ci_high (a 95% confidence band).
    """
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")

    rng = np.random.default_rng(seed)
    disc = math.exp(-spec.r * spec.T)
    drift = (spec.r - spec.q - 0.5 * spec.sigma ** 2) * spec.T
    sqrt_T = math.sqrt(spec.T)

    if antithetic:
        n_half = n_paths // 2
        Z = rng.standard_normal(n_half)
        S_up = spec.S0 * np.exp(drift + spec.sigma * sqrt_T * Z)
        S_dn = spec.S0 * np.exp(drift - spec.sigma * sqrt_T * Z)
        # Pair-average the target payoff FIRST; each pair average is one
        # (approximately) independent observation for variance purposes.
        payoffs = disc * 0.5 * (spec.payoff(S_up) + spec.payoff(S_dn))
        if control_variate:
            call_payoffs = disc * 0.5 * (np.maximum(S_up - spec.K, 0.0)
                                         + np.maximum(S_dn - spec.K, 0.0))
    else:
        Z = rng.standard_normal(n_paths)
        S_T = spec.S0 * np.exp(drift + spec.sigma * sqrt_T * Z)
        payoffs = disc * spec.payoff(S_T)
        if control_variate:
            call_payoffs = disc * np.maximum(S_T - spec.K, 0.0)

    if control_variate:
        # Control: discounted European-call payoff with known BSM expectation.
        expected_control = bs_price(OptionSpec(spec.S0, spec.K, spec.T, spec.r,
                                               spec.sigma, spec.q, "call"))
        var_x = np.var(call_payoffs, ddof=1)
        if var_x > 0.0:
            cov_yx = np.cov(payoffs, call_payoffs, ddof=1)[0, 1]
            c_star = cov_yx / var_x
            payoffs = payoffs - c_star * (call_payoffs - expected_control)

    price = float(np.mean(payoffs))
    stderr = float(np.std(payoffs, ddof=1) / math.sqrt(payoffs.size))
    price = max(price, 0.0)  # prices cannot be negative
    return {
        "price": price,
        "stderr": stderr,
        "ci_low": max(price - 1.96 * stderr, 0.0),
        "ci_high": price + 1.96 * stderr,
        # Number of independent observations actually averaged: raw paths
        # when antithetic is off, antithetic PAIRS when it is on (each pair
        # counts once because the pair average is the independent unit).
        "n_paths": int(payoffs.size),
        "n_simulated_paths": n_paths,
    }


# ==============================================================================
# Section 5: Implied volatility (Brent bracket + Newton-Raphson polish)
# ==============================================================================

def implied_volatility(price: float, spec: OptionSpec,
                       vol_low: float = 1e-6, vol_high: float = 5.0,
                       tol: float = 1e-10, max_newton: int = 20) -> float:
    """Recover the volatility that makes the BSM price equal to `price`.

    Strategy:
      1. Bracket the root of f(sigma) = BS(sigma) - price with Brent's
         method on [vol_low, vol_high]. Brent is robust and cannot diverge.
      2. Polish with a few safeguarded Newton-Raphson steps that use the
         analytic vega as the derivative. Newton steps that leave the bracket
         are rejected, so the solver stays bounded by construction.

    Raises
    ------
    ValueError
        If the target price is below intrinsic value (arbitrage-inconsistent)
        or no root exists inside the volatility bracket.
    """
    if price < 0.0:
        raise ValueError("Target price is negative; no implied vol exists.")

    def objective(vol: float) -> float:
        trial = OptionSpec(spec.S0, spec.K, spec.T, spec.r, vol, spec.q,
                           spec.option_type)
        return bs_price(trial) - price

    f_low, f_high = objective(vol_low), objective(vol_high)
    if f_low * f_high > 0.0:
        raise ValueError(
            "Implied volatility root is not bracketed: the target price lies "
            "outside the range of BSM prices for vol in "
            f"[{vol_low}, {vol_high}]. Check the input price for arbitrage "
            "inconsistency (e.g., below intrinsic or discounted bounds).")

    # Step 1: robust bracketed solve.
    vol_brent = brentq(objective, vol_low, vol_high, xtol=tol, rtol=tol)

    # Step 2: safeguarded Newton polish using analytic vega.
    vol = vol_brent
    lo, hi = vol_low, vol_high
    for _ in range(max_newton):
        trial = OptionSpec(spec.S0, spec.K, spec.T, spec.r, vol, spec.q,
                           spec.option_type)
        f_val = bs_price(trial) - price
        if abs(f_val) < tol:
            break
        vega = bs_greeks(trial)["vega"]
        if vega <= 1e-12:
            break  # vega too small for a meaningful Newton step
        candidate = vol - f_val / vega
        # Keep the iterate strictly inside the Brent bracket for safety.
        if not (lo < candidate < hi):
            break
        # Tighten the bracket using the sign of the residual.
        if f_val > 0:
            hi = vol
        else:
            lo = vol
        vol = candidate
    return vol


# ==============================================================================
# Section 6: Correctness checks (the validation battery)
# ==============================================================================

def check_put_call_parity(spec: OptionSpec) -> float:
    """Verify C - P = S0 e^{-qT} - K e^{-rT} to machine precision."""
    call = bs_price(OptionSpec(spec.S0, spec.K, spec.T, spec.r, spec.sigma,
                               spec.q, "call"))
    put = bs_price(OptionSpec(spec.S0, spec.K, spec.T, spec.r, spec.sigma,
                              spec.q, "put"))
    lhs = call - put
    rhs = spec.S0 * math.exp(-spec.q * spec.T) - spec.K * math.exp(-spec.r * spec.T)
    return abs(lhs - rhs)


def check_delta_parity(spec: OptionSpec) -> float:
    """Verify the delta parity identity delta(call) - delta(put) = e^{-qT}.

    This follows directly from differentiating put-call parity
    C - P = S0 e^{-qT} - K e^{-rT} with respect to S0, and is an
    independent check on the analytic Greek formulas (as distinct from
    check_put_call_parity, which checks the price formulas)."""
    call = OptionSpec(spec.S0, spec.K, spec.T, spec.r, spec.sigma, spec.q, "call")
    put = OptionSpec(spec.S0, spec.K, spec.T, spec.r, spec.sigma, spec.q, "put")
    lhs = bs_greeks(call)["delta"] - bs_greeks(put)["delta"]
    rhs = math.exp(-spec.q * spec.T)
    return abs(lhs - rhs)


def check_american_ge_european(spec: OptionSpec, steps: int = 400) -> bool:
    """An American option carries all European rights plus early exercise,
    so its value can never be lower than the European value."""
    euro = crr_price(spec, steps=steps, american=False)
    amer = crr_price(spec, steps=steps, american=True)
    return amer >= euro - 1e-12


def check_tree_convergence(spec: OptionSpec,
                           step_counts: List[int] = None
                           ) -> Dict[str, object]:
    """Verify the CRR European tree converges to the BSM price as steps grow.

    The CRR tree is known to converge at O(1/steps) but with the classic
    "sawtooth" oscillation (the error alternates sign as the moneyness moves
    in and out of alignment with the tree's node lattice), so error is not
    monotone in steps even though its envelope shrinks like 1/steps. We
    report the raw signed and absolute error at each step count, from 50 to
    1000 as specified, rather than asserting monotonicity that the theory
    does not promise.
    """
    if step_counts is None:
        step_counts = [50, 100, 200, 300, 400, 500, 600, 800, 1000]
    exact = bs_price(spec)
    signed_errors, abs_errors = [], []
    for n in step_counts:
        tree_price = crr_price(spec, steps=n, american=False)
        signed_errors.append(tree_price - exact)
        abs_errors.append(abs(tree_price - exact))
    return {"step_counts": step_counts, "signed_errors": signed_errors,
            "abs_errors": abs_errors, "exact": exact,
            "pass": abs_errors[-1] < abs_errors[0]}  # 1000 steps beats 50


def check_greeks_vs_finite_difference(spec: OptionSpec,
                                      rel_tol: float = 1e-3) -> Dict[str, Dict[str, float]]:
    """Compare analytic Greeks with central finite differences."""
    analytic = bs_greeks(spec)
    numeric = fd_greeks(spec)
    report: Dict[str, Dict[str, float]] = {}
    for key in analytic:
        a, n = analytic[key], numeric[key]
        denom = max(abs(a), 1e-8)
        report[key] = {"analytic": a, "finite_diff": n,
                       "rel_error": abs(a - n) / denom,
                       "pass": abs(a - n) / denom < rel_tol}
    return report


def check_mc_convergence(spec: OptionSpec,
                         path_counts: List[int] = None,
                         seed: int = 7) -> Dict[str, object]:
    """Verify that MC standard error decays like N^(-1/2).

    We estimate the slope of log(stderr) versus log(N). A clean N^(-1/2)
    decay corresponds to a slope of -0.5. Variance reduction rescales the
    constant in front of the error but not the -1/2 exponent.
    """
    if path_counts is None:
        path_counts = [1_000, 4_000, 16_000, 64_000]
    exact = bs_price(spec)
    errs, stderrs = [], []
    for i, n in enumerate(path_counts):
        # Distinct but deterministic seeds per rung avoid reusing draws.
        res = mc_price(spec, n_paths=n, seed=seed + i,
                       antithetic=True, control_variate=False)
        stderrs.append(res["stderr"])
        errs.append(abs(res["price"] - exact))
    log_n = np.log(np.asarray(path_counts, dtype=float))
    log_se = np.log(np.asarray(stderrs, dtype=float))
    slope = float(np.polyfit(log_n, log_se, 1)[0])
    return {"path_counts": path_counts, "abs_errors": errs,
            "stderrs": stderrs, "loglog_slope": slope,
            "pass": -0.65 < slope < -0.35}


def check_iv_round_trip(spec: OptionSpec, tol: float = 1e-6) -> Dict[str, float]:
    """Price with sigma, then recover sigma from the price."""
    price = bs_price(spec)
    recovered = implied_volatility(price, spec)
    return {"input_sigma": spec.sigma, "recovered_sigma": recovered,
            "abs_error": abs(recovered - spec.sigma),
            "pass": abs(recovered - spec.sigma) < tol}


def quantify_variance_reduction(spec: OptionSpec,
                                path_counts: List[int] = None,
                                n_trials: int = 30
                                ) -> Dict[str, object]:
    """Quantify variance reduction with a real computed statistic.

    A single Monte Carlo draw's absolute error is itself a noisy random
    variable, so comparing plain vs. antithetic vs. control-variate MC on
    one draw each is not a statistically honest comparison (it can show
    either method "winning" by chance). Instead, for every (method, path
    count) cell we run n_trials independent repetitions, using identical
    seeds across the three methods at each trial so the comparison is
    paired, and report the root-mean-square error across trials. RMSE is
    an unbiased, low-noise estimator of the true standard error and must
    decrease monotonically as variance reduction techniques are layered on.
    """
    if path_counts is None:
        path_counts = [1_000, 4_000, 16_000, 64_000]
    exact = bs_price(spec)
    methods = [("plain", False, False), ("antithetic", True, False),
              ("antithetic_cv", True, True)]
    result: Dict[str, object] = {"path_counts": path_counts, "rmse": {}}
    for label, anti, cv in methods:
        rmses = []
        for n in path_counts:
            sq_errs = []
            for trial in range(n_trials):
                r_ = mc_price(spec, n_paths=n, seed=1000 * trial + n,
                              antithetic=anti, control_variate=cv)
                sq_errs.append((r_["price"] - exact) ** 2)
            rmses.append(max(math.sqrt(float(np.mean(sq_errs))), 1e-14))
        result["rmse"][label] = rmses
    # Reduction factors at the largest path count, the headline numbers.
    plain_rmse = result["rmse"]["plain"][-1]
    anti_rmse = result["rmse"]["antithetic"][-1]
    cv_rmse = result["rmse"]["antithetic_cv"][-1]
    result["n_at_headline"] = path_counts[-1]
    result["antithetic_reduction_factor"] = plain_rmse / anti_rmse
    result["control_variate_reduction_factor"] = plain_rmse / cv_rmse
    return result


def check_mc_coverage(spec: OptionSpec, n_trials: int = 200,
                      n_paths: int = 2_000, seed: int = 123
                      ) -> Dict[str, float]:
    """Empirical coverage of the 95% MC confidence band around the BS price.

    With correct standard errors, the band should contain the exact BSM price
    in approximately 95% of independent simulations. We disable the control
    variate here because the call control variate collapses the variance of a
    call target to nearly zero, which would make coverage trivially 100%.
    """
    exact = bs_price(spec)
    hits = 0
    for i in range(n_trials):
        res = mc_price(spec, n_paths=n_paths, seed=seed + i,
                       antithetic=True, control_variate=False)
        if res["ci_low"] <= exact <= res["ci_high"]:
            hits += 1
    coverage = hits / n_trials
    return {"coverage": coverage, "n_trials": n_trials,
            "pass": 0.88 <= coverage <= 1.0}


# ==============================================================================
# Section 7: Triangulation harness across moneyness, maturity, volatility
# ==============================================================================

MONEYNESS_GRID = [0.8, 0.9, 1.0, 1.1, 1.2]            # defined as S0 / K
MATURITY_GRID = {"1M": 1.0 / 12.0, "3M": 0.25, "6M": 0.5,
                 "1Y": 1.0, "2Y": 2.0}
VOL_GRID = [0.10, 0.20, 0.40]


def run_triangulation(S0: float = 100.0, r: float = 0.05, q: float = 0.02,
                      option_type: str = "call",
                      tree_steps: int = 300, mc_paths: int = 50_000,
                      seed: int = 42) -> List[Dict[str, float]]:
    """Price every grid cell with BSM, CRR tree, and Monte Carlo.

    Moneyness is defined as S0 / K, so K = S0 / moneyness. For each cell we
    also recover the implied volatility from the tree price as an end-to-end
    self-consistency probe: if the tree and the IV solver are both correct,
    the recovered vol must equal the grid vol.
    """
    rows: List[Dict[str, float]] = []
    run = 0
    for m in MONEYNESS_GRID:
        K = S0 / m
        for mat_name, T in MATURITY_GRID.items():
            for sig in VOL_GRID:
                run += 1
                spec = OptionSpec(S0, K, T, r, sig, q, option_type)
                p_bs = bs_price(spec)
                p_tree = crr_price(spec, steps=tree_steps, american=False)
                mc = mc_price(spec, n_paths=mc_paths, seed=seed + run)
                iv_from_tree = implied_volatility(p_tree, spec)
                rows.append({
                    "moneyness": m, "maturity": mat_name, "T": T,
                    "sigma": sig, "bs": p_bs, "tree": p_tree,
                    "mc": mc["price"], "mc_stderr": mc["stderr"],
                    "tree_vs_bs_abs": abs(p_tree - p_bs),
                    "mc_vs_bs_abs": abs(mc["price"] - p_bs),
                    "iv_from_tree": iv_from_tree,
                    "iv_error": abs(iv_from_tree - sig),
                })
    return rows


def print_triangulation_table(rows: List[Dict[str, float]]) -> None:
    """Pretty-print the triangulation results as an aligned text table."""
    header = (f"{'M(S/K)':>7} {'Mat':>4} {'sigma':>6} {'BS':>10} "
              f"{'CRR':>10} {'MC':>10} {'MC se':>8} "
              f"{'|CRR-BS|':>10} {'|MC-BS|':>10} {'IV err':>9}")
    print(header)
    print("-" * len(header))
    for r_ in rows:
        print(f"{r_['moneyness']:>7.2f} {r_['maturity']:>4} "
              f"{r_['sigma']:>6.2f} {r_['bs']:>10.5f} {r_['tree']:>10.5f} "
              f"{r_['mc']:>10.5f} {r_['mc_stderr']:>8.5f} "
              f"{r_['tree_vs_bs_abs']:>10.6f} {r_['mc_vs_bs_abs']:>10.6f} "
              f"{r_['iv_error']:>9.2e}")


# ==============================================================================
# Section 8: Optional QuantLib benchmark (soft dependency)
# ==============================================================================

def quantlib_price(spec: OptionSpec) -> Optional[float]:
    """Price the European option with QuantLib if it is installed.

    Returns None when QuantLib is unavailable, keeping it a soft dependency.
    """
    if not HAS_QUANTLIB:
        return None
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today
    maturity = calendar.advance(today, ql.Period(
        int(round(spec.T * 365)), ql.Days))
    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if spec.is_call else ql.Option.Put, spec.K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    spot = ql.QuoteHandle(ql.SimpleQuote(spec.S0))
    div = ql.YieldTermStructureHandle(
        ql.FlatForward(today, spec.q, day_count))
    rate = ql.YieldTermStructureHandle(
        ql.FlatForward(today, spec.r, day_count))
    vol = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, spec.sigma, day_count))
    process = ql.BlackScholesMertonProcess(spot, div, rate, vol)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    # QuantLib uses actual dates; rescale T difference is small for the demo.
    return float(option.NPV())


def run_quantlib_comparison(rows: List[Dict[str, float]], S0: float,
                            r: float, q: float, option_type: str) -> None:
    """Compare a subset of triangulation cells against QuantLib."""
    if not HAS_QUANTLIB:
        print("\n[QuantLib] Not installed. Skipping optional benchmark; "
              "the engine is fully functional without it.")
        return
    print("\n[QuantLib] Benchmark on a subset of triangulation cells:")
    print(f"{'M(S/K)':>7} {'Mat':>4} {'sigma':>6} {'BS (ours)':>12} "
          f"{'QuantLib':>12} {'abs diff':>12}")
    for row in rows[::15]:  # sample every 15th cell to keep output compact
        K = S0 / row["moneyness"]
        spec = OptionSpec(S0, K, row["T"], r, row["sigma"], q, option_type)
        ql_p = quantlib_price(spec)
        if ql_p is not None:
            print(f"{row['moneyness']:>7.2f} {row['maturity']:>4} "
                  f"{row['sigma']:>6.2f} {row['bs']:>12.5f} {ql_p:>12.5f} "
                  f"{abs(ql_p - row['bs']):>12.6f}")


# ==============================================================================
# Section 9: Implied volatility smile (live option chain, synthetic fallback)
# ==============================================================================
#
# The IV round-trip check (Section 6) proves the solver is internally
# consistent (price -> sigma -> price). This section is the complementary,
# externally-facing deliverable: apply the same solver to an actual quoted
# option chain and produce a smile. A live data path is always attempted
# first; if the network call fails or yfinance is not installed, the code
# falls back to a CLEARLY LABELED synthetic chain built from a known skewed
# volatility surface, so the smile machinery is demonstrated either way.

def build_iv_smile(ticker: str = "SPY", target_days: int = 30,
                   r_assumed: float = 0.05, q_assumed: float = 0.015
                   ) -> Dict[str, object]:
    """Build an implied-volatility smile from a live option chain.

    Falls back to a synthetic chain (clearly labeled as such) if a live
    quote cannot be retrieved, e.g. no network access or yfinance missing.
    """
    try:
        import yfinance as yf  # optional, soft dependency

        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            raise RuntimeError("No listed option expirations returned.")

        today = datetime.now(timezone.utc).date()
        exp_dates = [datetime.strptime(e, "%Y-%m-%d").date() for e in expirations]
        days_out = [(d - today).days for d in exp_dates]
        # Pick the expiration closest to target_days, but never same-day/expired.
        candidates = [(abs(d - target_days), e, d) for d, e in
                     zip(days_out, expirations) if d >= 7]
        if not candidates:
            raise RuntimeError("No expiration at least 7 days out.")
        _, expiration, days = min(candidates, key=lambda x: x[0])
        T = days / 365.0

        spot_hist = tk.history(period="1d")
        if spot_hist.empty:
            raise RuntimeError("No spot price history returned.")
        S0 = float(spot_hist["Close"].iloc[-1])

        chain = tk.option_chain(expiration)
        calls = chain.calls
        rows = []
        for _, row in calls.iterrows():
            K = float(row["strike"])
            bid, ask = float(row["bid"]), float(row["ask"])
            if bid <= 0.0 or ask <= 0.0 or ask < bid:
                continue  # skip stale/illiquid quotes
            mid = 0.5 * (bid + ask)
            moneyness = S0 / K
            if not (0.7 <= moneyness <= 1.3):
                continue  # keep the smile near the money where quotes are live
            spec = OptionSpec(S0, K, T, r_assumed, 0.20, q_assumed, "call")
            try:
                iv = implied_volatility(mid, spec)
            except ValueError:
                continue  # price outside the no-arbitrage bracket; skip
            rows.append({"strike": K, "moneyness": moneyness,
                        "market_price": mid, "iv": iv})
        if len(rows) < 5:
            raise RuntimeError("Fewer than 5 usable quotes after filtering.")
        rows.sort(key=lambda d: d["strike"])
        return {"source": "real", "ticker": ticker, "expiration": expiration,
                "days_to_expiry": days, "S0": S0, "r": r_assumed, "q": q_assumed,
                "rows": rows}
    except Exception as exc:
        print(f"    [IV smile] Live chain unavailable ({type(exc).__name__}: "
              f"{exc}). Falling back to a SYNTHETIC labeled chain.")
        return _synthetic_iv_smile(r_assumed=r_assumed, q_assumed=q_assumed)


def _synthetic_iv_smile(S0: float = 100.0, T: float = 30.0 / 365.0,
                        r_assumed: float = 0.05, q_assumed: float = 0.015
                        ) -> Dict[str, object]:
    """Construct a synthetic option chain with a known volatility skew.

    A quadratic-in-log-moneyness skew (higher vol for low strikes, the
    classic equity "smirk") is used to generate SYNTHETIC market prices via
    BSM; those prices are then inverted with the same implied_volatility
    solver used for real data, so the recovered smile should reproduce the
    input skew up to solver tolerance. This is a demonstration data set, not
    a market observation.
    """
    strikes = np.linspace(0.8, 1.2, 17) * S0
    rows = []
    for K in strikes:
        log_m = math.log(S0 / K)
        true_vol = 0.20 - 0.35 * log_m + 0.9 * log_m ** 2  # synthetic skew
        true_vol = max(true_vol, 0.03)
        spec = OptionSpec(S0, float(K), T, r_assumed, true_vol, q_assumed, "call")
        market_price = bs_price(spec)
        iv_spec = OptionSpec(S0, float(K), T, r_assumed, 0.20, q_assumed, "call")
        iv = implied_volatility(market_price, iv_spec)
        rows.append({"strike": float(K), "moneyness": S0 / K,
                    "market_price": market_price, "iv": iv,
                    "true_vol": true_vol})
    return {"source": "synthetic", "ticker": "SYNTHETIC", "expiration": "N/A",
            "days_to_expiry": round(T * 365), "S0": S0, "r": r_assumed,
            "q": q_assumed, "rows": rows}


def plot_iv_smile(smile: Dict[str, object], out_prefix: str = "I2") -> str:
    """Plot implied volatility versus strike and save it to a PNG."""
    rows = smile["rows"]
    strikes = [row["strike"] for row in rows]
    ivs = [row["iv"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if smile["source"] == "real":
        label = f"{smile['ticker']} calls, exp {smile['expiration']}"
        title = (f"Implied volatility smile: LIVE chain "
                 f"({smile['ticker']}, {smile['days_to_expiry']}d to expiry, "
                 f"S0={smile['S0']:.2f})")
    else:
        label = "Synthetic calls (labeled demonstration data)"
        title = (f"Implied volatility smile: SYNTHETIC chain "
                 f"({smile['days_to_expiry']}d to expiry, S0={smile['S0']:.2f})")
    ax.plot(strikes, ivs, "o-", label=label)
    ax.axvline(smile["S0"], color="gray", linestyle=":", alpha=0.6,
              label="Spot")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fpath = f"{out_prefix}_iv_smile.png"
    fig.tight_layout()
    fig.savefig(fpath, dpi=130)
    plt.close(fig)
    return fpath


# ==============================================================================
# Section 10: Demo plots
# ==============================================================================

def make_demo_plots(spec: OptionSpec, out_prefix: str = "I2",
                    vr_result: Optional[Dict[str, object]] = None
                    ) -> List[str]:
    """Generate the demo figures and return their file paths."""
    saved: List[str] = []

    # --- Figure 1: Monte Carlo convergence with N^(-1/2) reference slope ---
    conv = check_mc_convergence(spec)
    ns = np.asarray(conv["path_counts"], dtype=float)
    ses = np.asarray(conv["stderrs"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(ns, ses, "o-", label="MC standard error (antithetic)")
    # Reference line with exact -1/2 slope, anchored to the first point.
    ref = ses[0] * (ns / ns[0]) ** -0.5
    ax.loglog(ns, ref, "--", color="gray", label="N$^{-1/2}$ reference")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Standard error of price estimate")
    ax.set_title(f"MC convergence, fitted slope = {conv['loglog_slope']:.3f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    f1 = f"{out_prefix}_mc_convergence.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    saved.append(f1)

    # --- Figure 2: Variance reduction comparison across path counts ---
    # See quantify_variance_reduction() for why RMSE over repeated trials,
    # rather than a single noisy draw, is the honest metric to plot here.
    # This figure uses a PUT contract (passed in via vr_result from main()):
    # for a call target the call control variate collapses to numerical
    # zero by construction (a degenerate special case, discussed in the
    # README), which would make the chart uninformative. The put is the
    # genuine, non-degenerate test of the control-variate technique.
    if vr_result is None:
        put_spec = OptionSpec(spec.S0, spec.K, spec.T, spec.r, spec.sigma,
                              spec.q, "put")
        vr_result = quantify_variance_reduction(put_spec, conv["path_counts"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_labels = {"plain": "Plain MC", "antithetic": "Antithetic",
                   "antithetic_cv": "Antithetic + control variate"}
    for key, label in plot_labels.items():
        ax.loglog(vr_result["path_counts"], vr_result["rmse"][key], "o-",
                  label=label)
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("RMS error vs BSM over 30 trials (log scale)")
    ax.set_title("Variance reduction effectiveness (put contract)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    f2 = f"{out_prefix}_variance_reduction.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    saved.append(f2)

    # --- Figure 3: European vs American put value across spot levels ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    spots = np.linspace(0.6 * spec.K, 1.4 * spec.K, 25)
    euro_puts, amer_puts = [], []
    for s in spots:
        s_ = OptionSpec(s, spec.K, spec.T, spec.r, spec.sigma, spec.q, "put")
        euro_puts.append(crr_price(s_, steps=200, american=False))
        amer_puts.append(crr_price(s_, steps=200, american=True))
    ax.plot(spots, euro_puts, label="European put (CRR)")
    ax.plot(spots, amer_puts, label="American put (CRR)")
    ax.set_xlabel("Spot price")
    ax.set_ylabel("Option value")
    ax.set_title("Early exercise premium of the American put")
    ax.legend()
    ax.grid(True, alpha=0.3)
    f3 = f"{out_prefix}_american_vs_european.png"
    fig.tight_layout()
    fig.savefig(f3, dpi=130)
    plt.close(fig)
    saved.append(f3)

    # --- Figure 4: CRR tree convergence to BSM as steps grow, 50 to 1000 ---
    tree_conv = check_tree_convergence(spec)
    steps_arr = np.asarray(tree_conv["step_counts"], dtype=float)
    signed_err = np.asarray(tree_conv["signed_errors"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.plot(steps_arr, signed_err, "o-", color="tab:red",
           label="CRR European - BSM (signed)")
    ax.set_xlabel("Number of tree steps")
    ax.set_ylabel("Pricing error vs BSM")
    ax.set_title("CRR tree convergence: signed error, 50 to 1000 steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    f4 = f"{out_prefix}_tree_convergence.png"
    fig.tight_layout()
    fig.savefig(f4, dpi=130)
    plt.close(fig)
    saved.append(f4)

    return saved


# ==============================================================================
# Section 11: Demo driver (synthetic data by default; live smile if reachable)
# ==============================================================================

def main() -> None:
    """Run the full demonstration and validation suite.

    All market inputs below are SYNTHETIC demonstration parameters, chosen to
    represent a plausible equity index option environment. They are not live
    market data and produce no real-world pricing claims.
    """
    t_start = time.time()
    print("=" * 78)
    print("Project I-2: Multi-Method Derivative Pricing and Greeks Engine")
    print("DEMONSTRATION MODE: synthetic market parameters, no live data.")
    print("=" * 78)

    # ---- Synthetic reference contract -------------------------------------
    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
    call = OptionSpec(S0, K, T, r, sigma, q, "call")
    put = OptionSpec(S0, K, T, r, sigma, q, "put")

    print(f"\nReference contract (synthetic): S0={S0}, K={K}, T={T}y, "
          f"r={r:.2%}, sigma={sigma:.2%}, q={q:.2%}")

    # ---- 1. BSM prices and Greeks ------------------------------------------
    print("\n--- Black-Scholes-Merton prices and analytic Greeks ---")
    for spec in (call, put):
        g = bs_greeks(spec)
        print(f"{spec.option_type.upper():>4}: price={bs_price(spec):.6f} | "
              f"delta={g['delta']:+.5f} gamma={g['gamma']:.6f} "
              f"vega={g['vega']:.5f} theta={g['theta']:+.5f} rho={g['rho']:+.5f}")

    # ---- 2. Binomial trees --------------------------------------------------
    print("\n--- Cox-Ross-Rubinstein trees (500 steps) ---")
    for spec in (call, put):
        euro = crr_price(spec, steps=500, american=False)
        amer = crr_price(spec, steps=500, american=True)
        print(f"{spec.option_type.upper():>4}: European={euro:.6f}  "
              f"American={amer:.6f}  "
              f"early-exercise premium={amer - euro:+.6f}")

    tree_conv = check_tree_convergence(call)
    print("    Tree convergence to BSM (European call, 50 to 1000 steps): "
          f"|error| at 50 steps={tree_conv['abs_errors'][0]:.5f}, "
          f"at 1000 steps={tree_conv['abs_errors'][-1]:.5f} "
          f"-> {'PASS' if tree_conv['pass'] else 'FAIL'} "
          "(see I2_tree_convergence.png; CRR error oscillates but its "
          "envelope shrinks like 1/steps, so it is not required to "
          "decrease monotonically at every step count)")

    # ---- 3. Monte Carlo with variance reduction -----------------------------
    print("\n--- Monte Carlo (200k paths, antithetic + control variate) ---")
    for spec in (call, put):
        res = mc_price(spec, n_paths=200_000, seed=2024)
        print(f"{spec.option_type.upper():>4}: price={res['price']:.6f}  "
              f"se={res['stderr']:.6f}  "
              f"95% CI=[{res['ci_low']:.6f}, {res['ci_high']:.6f}]  "
              f"BS={bs_price(spec):.6f}")

    # ---- 4. Validation battery ----------------------------------------------
    print("\n--- Correctness checks ---")
    parity_err = check_put_call_parity(call)
    print(f"[1] Put-call parity |C - P - (S e^-qT - K e^-rT)| = {parity_err:.3e} "
          f"-> {'PASS' if parity_err < 1e-10 else 'FAIL'}")

    ok_am = all(check_american_ge_european(s) for s in (call, put))
    print(f"[2] American >= European (call and put) -> "
          f"{'PASS' if ok_am else 'FAIL'}")

    gcheck = check_greeks_vs_finite_difference(call)
    g_all = all(v["pass"] for v in gcheck.values())
    for k, v in gcheck.items():
        print(f"    {k:>5}: analytic={v['analytic']:+.6f} "
              f"FD={v['finite_diff']:+.6f} rel_err={v['rel_error']:.2e}")
    print(f"[3] Analytic Greeks vs finite differences -> "
          f"{'PASS' if g_all else 'FAIL'}")

    dparity_err = check_delta_parity(call)
    print(f"[4] Delta parity |delta(call) - delta(put) - e^-qT| = "
          f"{dparity_err:.3e} -> {'PASS' if dparity_err < 1e-10 else 'FAIL'}")

    conv = check_mc_convergence(call)
    print(f"[5] MC convergence: log-log slope = {conv['loglog_slope']:.3f} "
          f"(theory: -0.500) -> {'PASS' if conv['pass'] else 'FAIL'}")

    iv = check_iv_round_trip(OptionSpec(95.0, 100.0, 0.5, 0.03, 0.27, 0.01, "put"))
    print(f"[6] IV round-trip: input={iv['input_sigma']:.4f} "
          f"recovered={iv['recovered_sigma']:.10f} -> "
          f"{'PASS' if iv['pass'] else 'FAIL'}")

    cov = check_mc_coverage(call)
    print(f"[7] MC 95% band coverage over {cov['n_trials']} trials: "
          f"{cov['coverage']:.1%} -> {'PASS' if cov['pass'] else 'FAIL'}")

    vr = quantify_variance_reduction(call)
    cv_rmse = vr["rmse"]["antithetic_cv"][-1]
    # Below ~1e-10 the "RMSE" is floating-point noise, not a meaningful
    # sampling error, because the call control variate is the call target
    # itself; reporting a raw ratio there (billions-of-x) would be a
    # meaningless number, so it is reported qualitatively instead.
    cv_desc = ("collapses to floating-point noise (effectively exact, as "
              "expected when the control variate equals the target)"
              if cv_rmse < 1e-10 else
              f"{vr['control_variate_reduction_factor']:.1f}x reduction "
              f"vs plain MC")
    print(f"[8] Variance reduction (CALL target) at N={vr['n_at_headline']:,} "
          f"(RMSE over 30 trials each): "
          f"plain={vr['rmse']['plain'][-1]:.3e}, "
          f"antithetic={vr['rmse']['antithetic'][-1]:.3e} "
          f"({vr['antithetic_reduction_factor']:.2f}x reduction vs plain), "
          f"antithetic+CV={cv_rmse:.3e} ({cv_desc}) -> "
          f"{'PASS' if vr['antithetic_reduction_factor'] >= 1.0 else 'FAIL'}")

    # The call control variate is the call payoff itself, so its variance
    # reduction is a degenerate (perfectly correlated) special case. The put
    # is the genuine test of the control-variate technique: the control
    # (call payoff) is correlated with the put payoff only through put-call
    # parity, so this number is the one that should be compared against the
    # "order of magnitude" rule of thumb for a good control variate.
    vr_put = quantify_variance_reduction(put)
    print(f"    Variance reduction (PUT target, genuine test) at "
          f"N={vr_put['n_at_headline']:,}: plain={vr_put['rmse']['plain'][-1]:.3e}, "
          f"antithetic={vr_put['rmse']['antithetic'][-1]:.3e} "
          f"({vr_put['antithetic_reduction_factor']:.2f}x), "
          f"antithetic+CV={vr_put['rmse']['antithetic_cv'][-1]:.3e} "
          f"({vr_put['control_variate_reduction_factor']:.2f}x reduction vs plain)")

    # ---- 5. Triangulation harness -------------------------------------------
    print("\n--- Triangulation harness: BSM vs CRR vs MC across the grid ---")
    print("(75 cells: 5 moneyness x 5 maturities x 3 vols, synthetic calls)")
    rows = run_triangulation(S0=100.0, r=0.05, q=0.02, option_type="call",
                             tree_steps=300, mc_paths=50_000, seed=42)
    print_triangulation_table(rows)
    max_tree = max(r_["tree_vs_bs_abs"] for r_ in rows)
    max_mc = max(r_["mc_vs_bs_abs"] for r_ in rows)
    max_iv = max(r_["iv_error"] for r_ in rows)
    print(f"\nMax |CRR - BS|  = {max_tree:.6f}")
    print(f"Max |MC - BS|   = {max_mc:.6f}")
    print(f"Max IV round-trip error from tree price = {max_iv:.3e}")

    # Grid-level MC coverage: what fraction of the 75 cells have the BS
    # price inside that cell's own two-standard-error MC band? This is
    # computed with the control variate OFF, matching check_mc_coverage:
    # every grid cell here is a call, so a call control variate collapses
    # the band to numerical zero width and would make this check trivial
    # rather than a genuine statistical statement about the estimator.
    grid_hits = 0
    for cell_index, row in enumerate(rows):
        K = 100.0 / row["moneyness"]
        cell_spec = OptionSpec(100.0, K, row["T"], 0.05, row["sigma"], 0.02, "call")
        # Deterministic, index-based seed (not Python's randomized hash()),
        # so this diagnostic reproduces exactly across runs and machines.
        cell_mc = mc_price(cell_spec, n_paths=50_000, seed=5000 + cell_index,
                           antithetic=True, control_variate=False)
        if abs(cell_mc["price"] - row["bs"]) <= 2.0 * cell_mc["stderr"]:
            grid_hits += 1
    grid_coverage = grid_hits / len(rows)
    print(f"Grid-level MC 2-stderr coverage (control variate off, "
          f"{len(rows)} cells): {grid_coverage:.1%} "
          f"({grid_hits}/{len(rows)} cells) -> "
          f"{'PASS' if grid_coverage >= 0.85 else 'FAIL'}")

    # ---- 6. Optional QuantLib benchmark --------------------------------------
    run_quantlib_comparison(rows, S0=100.0, r=0.05, q=0.02, option_type="call")

    # ---- 7. Implied volatility smile (live chain, synthetic fallback) --------
    print("\n--- Implied volatility smile ---")
    smile = build_iv_smile(ticker="SPY", target_days=30)
    if smile["source"] == "real":
        print(f"    Source: LIVE option chain, {smile['ticker']}, "
              f"expiration {smile['expiration']} "
              f"({smile['days_to_expiry']}d), spot={smile['S0']:.2f}, "
              f"assumed r={smile['r']:.2%}, q={smile['q']:.2%}")
    else:
        print(f"    Source: SYNTHETIC labeled chain "
              f"({smile['days_to_expiry']}d to expiry, spot={smile['S0']:.2f})")
    for row in smile["rows"][::3]:  # sample every 3rd strike to keep it compact
        print(f"    K={row['strike']:>8.2f}  moneyness={row['moneyness']:.3f}  "
              f"price={row['market_price']:.4f}  IV={row['iv']:.4f}")

    # ---- 8. Demo plots --------------------------------------------------------
    print("\n--- Generating demo plots ---")
    try:
        saved = make_demo_plots(call, vr_result=vr_put)
        smile_path = plot_iv_smile(smile)
        saved.append(smile_path)
        for f in saved:
            print(f"    saved: {f}")
    except Exception as exc:  # plots are cosmetic; never fail the engine
        print(f"    Plot generation skipped: {exc}")

    print(f"\nTotal runtime: {time.time() - t_start:.1f}s")
    print("=" * 78)
    print("Demo complete. Pricing, Greeks, and validation checks all use "
          "synthetic parameters; the IV smile above is live market data "
          "when reachable and a clearly labeled synthetic chain otherwise. "
          "No output here should be read as a real-time trading signal.")
    print("=" * 78)


if __name__ == "__main__":
    main()
