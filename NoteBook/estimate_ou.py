import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ── Functions ─────────────────────────────────────────────────────────────────

# Just like AR(1), however, this model is continuous, and the random variable is a brownian motion model
def simulate_ou(phi, mu, sigma, X0, N, dt=1):
    price = np.zeros(N)
    price[0] = X0
    for t in range(1, N):
        price[t] = price[t-1] + phi * (mu - price[t-1]) * dt + sigma * np.random.normal() * np.sqrt(dt)
    return price


# Now we'll do an OLS regression, because in real life, you dont know the values of the greek letters, we have to estimate them
def estimate_ou(prices):
    # from first bar to later
    y = prices[1:]
    # from last bar to back
    x = prices[:-1]

    # matrix [1,1,..1]
    X = np.column_stack([np.ones(len(x)), x])
    coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
    a = coeffs[0]  # intercept, x = 0
    b = coeffs[1]  # slope = phi, because its how much of yesterday's price affects todays price

    phi = b
    mu = a / (1 - b)
    residuals = y - (a + b * x)
    sigma = np.std(residuals)
    half_life = np.log(2) / np.log(1 / phi)
    return phi, mu, sigma, half_life


def rolling_ou(prices, window=60):
    phis       = np.full(len(prices), np.nan)
    mus        = np.full(len(prices), np.nan)
    sigmas     = np.full(len(prices), np.nan)
    half_lives = np.full(len(prices), np.nan)

    for t in range(window, len(prices)):
        window_slice = prices[t-window:t]
        phi, mu, sigma, half_life = estimate_ou(window_slice)
        phis[t]       = phi
        mus[t]        = mu
        sigmas[t]     = sigma
        half_lives[t] = half_life

    return phis, mus, sigmas, half_lives


def gated_rolling_ou(prices, window=60):
    phis       = np.full(len(prices), np.nan)
    mus        = np.full(len(prices), np.nan)
    sigmas     = np.full(len(prices), np.nan)
    half_lives = np.full(len(prices), np.nan)

    for t in range(window, len(prices)):
        # the last window, not including t
        window_slice = prices[t-window:t]

        # Check gate first
        adf_pvalue = adfuller(window_slice)[1]

        if adf_pvalue < 0.05:
            # Gate open — run OU estimator
            phi, mu, sigma, half_life = estimate_ou(window_slice)
            phis[t]       = phi
            mus[t]        = mu
            sigmas[t]     = sigma
            half_lives[t] = half_life
        # Gate closed — leave as NaN

    return phis, mus, sigmas, half_lives


# ── Main — only runs when this file is executed directly ──────────────────────

if __name__ == "__main__":

    # ── Simulate and plot OU paths ────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    for i in range(30):
        path = simulate_ou(phi=0.1, mu=100, sigma=1, X0=80, N=200)
        plt.plot(path, alpha=0.3, linewidth=0.8, color="steelblue")

    plt.axhline(y=100, color="red", linestyle="--", linewidth=1.5, label="μ = 100")
    plt.title("OU Process — φ=0.1, μ=100, X0=80")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # ── Download QQQ ─────────────────────────────────────────────────────
    import yfinance as yf

    data = yf.download("QQQ", period="7d", interval="1m", auto_adjust=True, prepost=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Extended hours: 4:00 AM to 8:00 PM
    data = data.between_time("04:00", "20:00")
    prices = data["Close"].dropna().values.flatten()

    print(f"Total bars: {len(prices)}")

    # ── Estimate on full series ───────────────────────────────────────────
    phi, mu, sigma, half_life = estimate_ou(prices)

    print(f"φ  (phi):       {phi:.4f}")
    print(f"μ  (mu):        {mu:.4f}")
    print(f"σ  (sigma):     {sigma:.4f}")
    print(f"Half-life:      {half_life:.1f} bars ({half_life:.1f} minutes)")

    # ── Rolling OU — ungated ──────────────────────────────────────────────
    phis, mus, sigmas, half_lives = rolling_ou(prices, window=60)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle("Rolling OU Parameters — QQQ 1min Extended Hours", fontsize=13)

    axes[0].plot(phis, linewidth=0.8, color="steelblue")
    axes[0].axhline(y=1.0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("φ (phi) — mean reversion speed")
    axes[0].set_ylabel("φ")

    axes[1].plot(mus, linewidth=0.8, color="green")
    axes[1].plot(prices, linewidth=0.8, color="steelblue", alpha=0.5)
    axes[1].set_title("μ (mu) — estimated mean vs actual price")
    axes[1].set_ylabel("Price")

    axes[2].plot(sigmas, linewidth=0.8, color="orange")
    axes[2].set_title("σ (sigma) — noise level")
    axes[2].set_ylabel("σ")

    axes[3].plot(half_lives, linewidth=0.8, color="purple")
    axes[3].axhline(y=20, color="red", linestyle="--", linewidth=1, label="20 bar max")
    axes[3].set_title("Half-life (bars)")
    axes[3].set_ylabel("Bars")
    axes[3].legend()

    plt.tight_layout()
    plt.show()

    # ── Gated rolling OU ─────────────────────────────────────────────────
    phis, mus, sigmas, half_lives = gated_rolling_ou(prices, window=60)

    gate_open_pct = np.sum(~np.isnan(phis)) / len(phis) * 100
    print(f"Gate OPEN  {gate_open_pct:.1f}% of bars")
    print(f"Gate CLOSED {100 - gate_open_pct:.1f}% of bars")

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    axes[0].plot(prices, linewidth=0.8, color="steelblue")
    axes[0].plot(mus, linewidth=1.0, color="red", alpha=0.7, label="μ (when gate open)")
    axes[0].set_title("QQQ Price + OU Mean (only when gate open)")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].plot(phis, linewidth=0.8, color="steelblue")
    axes[1].axhline(y=1.0, color="red", linestyle="--", linewidth=1, label="φ = 1 (random walk)")
    axes[1].set_title("φ — only filled when gate open")
    axes[1].set_ylabel("φ")
    axes[1].legend()

    axes[2].plot(sigmas, linewidth=0.8, color="orange")
    axes[2].set_title("σ — noise level")
    axes[2].set_ylabel("σ")

    axes[3].plot(half_lives, linewidth=0.8, color="purple")
    axes[3].axhline(y=20, color="red", linestyle="--", linewidth=1, label="20 bar max")
    axes[3].set_title("Half-life (bars) — only when gate open")
    axes[3].set_ylabel("Bars")
    axes[3].set_ylim(0, 100)
    axes[3].legend()

    gate_open = (~np.isnan(phis)).astype(int)
    axes[4].fill_between(range(len(gate_open)), gate_open, alpha=0.6, color="green", label="Gate OPEN")
    axes[4].set_title(f"Stationarity Gate — OPEN {gate_open_pct:.1f}% of bars")
    axes[4].set_ylabel("Gate")
    axes[4].set_yticks([0, 1])
    axes[4].set_yticklabels(["CLOSED", "OPEN"])
    axes[4].legend()

    plt.tight_layout()
    plt.show()