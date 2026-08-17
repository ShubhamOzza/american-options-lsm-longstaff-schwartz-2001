# Plain English Project Notes

## What we made

We built a Python program that answers one question three different ways: "What is a stock option worth?" An option is a contract that gives you the right, but not the obligation, to buy (a call) or sell (a put) a stock at a fixed price on or before a fixed date. Because this question matters to banks, hedgers, and traders, the program does not just spit out a number. It computes the answer with three independent methods, checks that they all agree, and runs a series of tests that would catch a bug before a human would.

## The core concept

There is no single "true" way to compute an option's value. The famous Black-Scholes-Merton formula gives an exact answer in one line of math, but it only works for the simplest case (European options, which can only be used at expiry). A binomial tree chops time into small steps and works out the value by reasoning backward from all possible ending prices; this handles American options, which can be exercised early. Monte Carlo simulation plays out tens of thousands of random stock-price futures and averages the payoff; this is the most flexible but the noisiest. If all three methods, which share almost no code, land on the same number, you can trust it. That idea is called triangulation, and it is how real quant desks validate their models.

## How we made it, step by step

1. One contract description for everything. We defined a single OptionSpec object holding the stock price, strike, time to expiry, interest rate, volatility, and dividend yield. It refuses nonsense inputs (like zero time or zero volatility) the moment it is created, so every pricer downstream can assume clean data.
2. The formula. We implemented Black-Scholes-Merton including dividends, plus its "Greeks": delta (how much the price moves when the stock moves one dollar), gamma (how fast delta itself changes), vega (sensitivity to volatility), theta (sensitivity to time passing), and rho (sensitivity to interest rates). These are the numbers a trader watches all day.
3. The tree. We built a Cox-Ross-Rubinstein binomial tree with fast vectorized backward induction. Setting a flag to "American" makes the code ask, at every point in time: is it worth more to exercise right now or to hold on? The gap between the American and European answers is the early-exercise premium.
4. The simulation. We simulated the stock under the risk-neutral random walk and averaged discounted payoffs. Raw simulation is noisy, so we added two free accuracy boosts: antithetic variates (every random scenario is paired with its mirror image, canceling noise) and a control variate (we simulate an option whose exact price we know, measure how wrong the simulation is on that one, and correct the target accordingly). With both turned on, call prices come out essentially exact.
5. Running the model backward. Given a market price, what volatility does it imply? We solved this with a two-stage method: a slow but unbreakable bracketing search (Brent), followed by fast Newton steps that are never allowed to leave the safe bracket. We use this same solver on a real, live SPY options chain pulled from the market to draw an actual volatility smile, and fall back to a clearly labeled synthetic chain if the live data is not reachable.
6. Proof, not promises. We built eight automatic tests into the script (listed below) and a 75-contract grid that prices everything with all three methods side by side.

## The eight correctness checks, in plain words

1. Put-call parity: a call minus a put must equal a simple combination of the stock and cash. Our residual was exactly zero.
2. American options, which give you strictly more rights, must never be cheaper than European ones. Checked everywhere.
3. The analytic Greeks matched "bump the input and re-price" approximations to eight decimal places.
4. A companion identity: the call's delta minus the put's delta must equal the dividend discount factor exactly. It did, to machine precision.
5. Doubling the simulation work four times over shrank the error by about the square-root law that statistics predicts (measured slope -0.484, theory -0.500).
6. We priced an option at 27% volatility, then asked the solver to recover it. It returned 0.2700000000.
7. The simulation's stated 95% confidence band really does contain the true answer about 95% of the time. We measured 95.0% over 200 trials, which is about as close to the textbook number as you could ask for.
8. We measured, not assumed, how much the two speed-up tricks actually help. For a put contract (the fair test), the combination of both tricks cut the pricing error by a factor of about ten compared to plain simulation, exactly the "order of magnitude" a good control variate is supposed to deliver.

## A bug we found and fixed along the way

While measuring check 8 honestly (running the same comparison many times instead of trusting a single lucky or unlucky run), we noticed the two speed-up tricks combined were making the put price worse, not better, than one trick alone. Tracing it down: when you pair up random draws and their mirror images (the first trick) and then also use a control variate (the second trick), the "how much to trust the control variate" number has to be computed from the paired averages, not from the raw individual draws. Computing it the naive way looked reasonable but quietly made the estimate noisier. Fixing that turned a shaky, borderline result into the clean 10x improvement reported above. It is exactly the kind of bug that a single test run would never catch, and it is the reason the "measure it many times, do not trust one draw" discipline exists.

## Interview-ready explanation

"I built a derivative pricing engine that values European and American options with three independent methods: closed-form Black-Scholes-Merton, a vectorized CRR binomial tree, and Monte Carlo under risk-neutral GBM. The interesting part is the validation design. All pricers consume one validated spec object, so I can triangulate: put-call parity and the matching delta-parity identity both hold to machine precision, analytic Greeks agree with finite differences to 1e-8, the American tree never prices below European, and I verified the Monte Carlo standard error decays at the theoretical N to the minus one-half rate by fitting the log-log slope. For speed I stacked antithetic variates with a European-call control variate; when the target is a call the control is perfectly correlated, so the variance collapses to zero, which is itself a nice correctness signal. For puts, where the reduction is genuine rather than degenerate, I measured about a 10x cut in RMS error at fixed path count, after finding and fixing a real bug where combining the two techniques naively made things worse instead of better. Implied vol is a safeguarded hybrid: Brent to bracket the root, Newton with analytic vega to polish, with steps rejected if they leave the bracket; I use the same solver to build an implied volatility smile from a live SPY option chain, with a labeled synthetic fallback. I also think about the bias-variance trade-off in numerical terms: too few tree steps or paths is underfitting, over-resolving beyond input uncertainty is wasted compute, and variance reduction lowers the error constant without changing the convergence rate. QuantLib can benchmark the results, but I kept it a soft dependency so the engine runs anywhere."

## Why this matters

Pricing libraries in industry fail silently all the time: a wrong sign in theta, a tree that ignores dividends, a confidence band that is too narrow. This project shows you can build the math and the safety net in the same codebase, which is exactly the habit that separates a quantitative developer from someone who once read the formula.
