# LinkedIn Project Portfolio Entry

## Project Title

Multi-Method Derivative Pricing and Greeks Engine: Triangulating Black-Scholes-Merton, Binomial Trees, and Variance-Reduced Monte Carlo

## Description

Built a production-style option pricing engine in Python that values European and American options three independent ways behind a single validated interface, then proves the methods agree before trusting any number.

What the engine does:

- Prices European options in closed form with Black-Scholes-Merton under continuous dividend yield, and computes the full analytic Greek suite (delta, gamma, vega, theta, rho).
- Values American and European options on a vectorized Cox-Ross-Rubinstein binomial tree, quantifying the early-exercise premium directly.
- Simulates risk-neutral geometric Brownian motion with antithetic variates and a European-call control variate, achieving near-exact call prices and an order-of-magnitude RMS error reduction on puts (the non-degenerate case) at a fraction of the usual path count.
- Inverts the model with a hybrid implied volatility solver: Brent bracketing for robustness, safeguarded Newton-Raphson with analytic vega for speed. The same solver drives an implied volatility smile from a live SPY option chain, with a clearly labeled synthetic fallback when no market connection is available.
- Benchmarks against QuantLib when available, with graceful degradation when it is not.

Why it is credible: the engine ships with its own eight-part validation battery. Put-call parity and the companion delta-parity identity (delta(call) minus delta(put) equals the dividend discount factor) both hold to machine precision (residual exactly 0.0). Analytic Greeks match central finite differences to relative errors below 1.3e-8. Monte Carlo standard error decays at the theoretical N^(-1/2) rate (fitted slope -0.484), and the 95 percent confidence band achieves genuine 95.0 percent coverage measured over 200 independent trials. Implied volatility round-trips recover inputs to ten decimal places. American values weakly dominate European values everywhere, and the American put shows a positive early-exercise premium exactly where theory predicts one. Variance reduction is measured, not assumed: for a put target at 64,000 paths, antithetic variates plus a control variate cut RMS pricing error by a factor of about 10 versus plain Monte Carlo, a result only reached after finding and fixing a real bug where the two techniques combined naively made the estimate worse. A 75-cell triangulation grid (5 moneyness x 5 maturities x 3 volatilities) confirms cross-method agreement within predicted discretization and sampling error.

The project also treats numerical error the way ML treats overfitting: too few tree steps or MC paths is underfitting (bias and wide error bars), while over-resolving beyond input uncertainty is wasted compute. Variance reduction attacks the error constant; convergence diagnostics verify the rate rather than assume it.

Stack: Python, NumPy, SciPy, Matplotlib, optional QuantLib. Fully reproducible from a single script with synthetic demo data.

## Skills (comma-separated)

Derivatives Pricing, Black-Scholes-Merton Model, Options Greeks, Binomial Tree Models, Cox-Ross-Rubinstein, American Options, Monte Carlo Simulation, Variance Reduction, Antithetic Variates, Control Variates, Implied Volatility, Root-Finding Algorithms, Newton-Raphson Method, Brent's Method, Put-Call Parity, Finite Difference Methods, Convergence Analysis, Numerical Methods, Quantitative Finance, Risk Management, Python, NumPy, SciPy, Matplotlib, QuantLib, Scientific Computing, Model Validation

## Media Recommendations

1. MC convergence plot (I2_mc_convergence.png): log-log chart of standard error versus path count with the N^(-1/2) reference line; immediately communicates rigor.
2. Variance reduction comparison (I2_variance_reduction.png): three RMS-error curves (over 30 trials per point, put contract) showing plain MC, antithetic, and antithetic plus control variate cleanly separated in the expected order; the strongest visual proof of engineering value.
3. American versus European put chart (I2_american_vs_european.png): the visible early-exercise premium gap; ideal for explaining why American options need trees.
4. Implied volatility smile (I2_iv_smile.png): a live SPY skew recovered by the same solver used in the round-trip check; shows the engine works on real quotes, not just synthetic ones.
5. A screenshot of the triangulation table output showing BS, CRR, and MC agreeing across the 75-cell grid.
6. Optional: a short screen recording of the script running end to end, from pricing to validation to the smile and plots, in a few seconds.
