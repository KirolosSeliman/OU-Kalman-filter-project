import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize

# ── 1. Download ───────────────────────────────────────────────────────────────
data = yf.download("QQQ", period="7d", interval="1m", auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.index = pd.to_datetime(data.index)
if data.index.tz is not None:
    data.index = data.index.tz_localize(None)

data = data.between_time("09:30", "16:00")
data["date"] = data.index.date

# ── 2. Build clean per-session prices ────────────────────────────────────────
all_prices = []
for date, session in data.groupby("date"):
    session_prices = session["Close"].dropna().values.flatten()
    if len(session_prices) < 60:
        continue
    # Demean each session — removes overnight gaps
    session_prices = session_prices - session_prices[0]
    all_prices.extend(session_prices)

prices = np.array(all_prices, dtype=float)
print(f"Total bars: {len(prices)}")
print(f"Price min:  {np.min(prices):.4f}")
print(f"Price max:  {np.max(prices):.4f}")
print(f"Max jump:   {np.max(np.abs(np.diff(prices))):.4f}")
print(f"Jumps > 1:  {np.sum(np.abs(np.diff(prices)) > 1.0)}")

# ── 2. Kalman Filter ──────────────────────────────────────────────────────────
def kalman_filter(prices, Q, R):
    x_hat       = np.full(len(prices), np.nan)
    P           = np.full(len(prices), np.nan)
    K_arr       = np.full(len(prices), np.nan)
    innovations = np.full(len(prices), np.nan)

    x_hat[0] = prices[0]
    P[0]     = 1.0

    for t in range(1, len(prices)):
        x_pred          = x_hat[t-1]
        P_pred          = P[t-1] + Q
        innovation      = prices[t] - x_pred
        K               = P_pred / (P_pred + R)
        x_hat[t]        = x_pred + K * innovation
        P[t]            = (1 - K) * P_pred
        K_arr[t]        = K
        innovations[t]  = innovation

    return x_hat, P, K_arr, innovations

# ── 3. MLE Initialization ─────────────────────────────────────────────────────
def mle_init(prices):
    price_diff_var = np.var(np.diff(prices))
    
    def neg_log_likelihood(params):
        Q = params[0]
        R = params[1]
        if Q <= 0 or R <= 0:
            return 1e10
        x_hat, P, K, innovations = kalman_filter(prices, Q, R)
        S = P + R
        valid = np.isfinite(S[1:]) & np.isfinite(innovations[1:]) & (S[1:] > 0)
        if np.sum(valid) < 10:
            return 1e10
        log_likelihood = -0.5 * np.sum(
            np.log(S[1:][valid]) + innovations[1:][valid]**2 / S[1:][valid]
        )
        return -log_likelihood

    # Bounded optimization — Q and R cannot exceed price variance
    from scipy.optimize import minimize
    bounds = [(1e-6, price_diff_var), (1e-6, price_diff_var)]
    initial_params = [price_diff_var * 0.1, price_diff_var * 0.1]
    
    result = minimize(
        neg_log_likelihood, 
        initial_params, 
        method="L-BFGS-B",  # supports bounds unlike Nelder-Mead
        bounds=bounds
    )
    
    return result.x[0], result.x[1]
# ── 4. VB-AKF ────────────────────────────────────────────────────────────────
def vb_akf(prices, Q_init, R_init, lambda_=0.98):
    x_hat       = np.full(len(prices), np.nan)
    P           = np.full(len(prices), np.nan)
    innovations = np.full(len(prices), np.nan)
    Q_arr       = np.full(len(prices), np.nan)
    R_arr       = np.full(len(prices), np.nan)

    alpha_R = 10
    beta_R  = 10 * R_init
    alpha_Q = 10
    beta_Q  = 10 * Q_init

    x_hat[0] = prices[0]
    P[0]     = 1.0
    Q_k      = Q_init
    R_k      = R_init

    for t in range(1, len(prices)):
        # PREDICT
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + Q_k
        S      = P_pred + R_k

        # RAW INNOVATION — used for Kalman update
        innovation = prices[t] - x_pred

        # KALMAN UPDATE — always uses raw innovation
        K              = P_pred / (P_pred + R_k)
        x_hat[t]       = x_pred + K * innovation
        P[t]           = (1 - K) * P_pred
        innovations[t] = innovation
        x_hat_change   = x_hat[t] - x_hat[t-1]

        # CAPPED INNOVATION — only for VB parameter update
        if t > 20:
            sigma_recent = np.nanstd(innovations[max(0, t-20):t])
        else:
            sigma_recent = np.nanstd(prices[:t])

        if abs(innovation) > 4 * sigma_recent and sigma_recent > 0:
            innovation_vb = np.sign(innovation) * 4 * sigma_recent
        else:
            innovation_vb = innovation

        # VB UPDATE — R
        alpha_R = lambda_ * alpha_R + 0.5
        beta_R  = lambda_ * beta_R  + 0.5 * (innovation_vb**2 + S)
        R_k     = beta_R / max(alpha_R - 1, 1e-6)

        # VB UPDATE — Q
        alpha_Q = lambda_ * alpha_Q + 0.5
        beta_Q  = lambda_ * beta_Q  + 0.5 * (x_hat_change**2)
        Q_k     = beta_Q / max(alpha_Q - 1, 1e-6)

        Q_arr[t] = Q_k
        R_arr[t] = R_k

    return x_hat, P, innovations, Q_arr, R_arr

# ── 5. Run MLE then VB-AKF ───────────────────────────────────────────────────
# ── Run MLE and VB-AKF per session ───────────────────────────────────────────
all_x_hat    = []
all_Q        = []
all_R        = []
all_prices_clean = []

for date, session in data.groupby("date"):
    session_prices = session["Close"].dropna().values.flatten()
    if len(session_prices) < 60:
        continue

    # MLE on this session
    Q_opt, R_opt = mle_init(session_prices)

    # VB-AKF on this session
    x_hat_s, P_s, innov_s, Q_arr_s, R_arr_s = vb_akf(session_prices, Q_opt, R_opt)

    all_x_hat.extend(x_hat_s)
    all_Q.extend(Q_arr_s)
    all_R.extend(R_arr_s)
    all_prices_clean.extend(session_prices)

prices      = np.array(all_prices_clean, dtype=float)
x_hat_vb    = np.array(all_x_hat,       dtype=float)
Q_arr       = np.array(all_Q,           dtype=float)
R_arr       = np.array(all_R,           dtype=float)

print(f"Q range: {np.nanmin(Q_arr):.6f} to {np.nanmax(Q_arr):.6f}")
print(f"R range: {np.nanmin(R_arr):.6f} to {np.nanmax(R_arr):.6f}")
print(f"Q mean:  {np.nanmean(Q_arr):.6f}")
print(f"R mean:  {np.nanmean(R_arr):.6f}")

# ── 6. Plot VB-AKF results ───────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(14, 12))
fig.suptitle("VB-AKF — Adaptive Kalman Filter", fontsize=13)

axes[0].plot(prices, linewidth=0.8, alpha=0.4, color="steelblue", label="Raw prices")
axes[0].plot(x_hat_vb, linewidth=1.2, color="red", label="VB-AKF estimate")
axes[0].set_title("Raw Price vs VB-AKF Estimate")
axes[0].legend()

axes[1].plot(Q_arr, linewidth=0.8, color="orange")
axes[1].set_title("Q_k — Process Noise (adaptive)")
axes[1].set_ylabel("Q")

axes[2].plot(R_arr, linewidth=0.8, color="green")
axes[2].set_title("R_k — Measurement Noise (adaptive)")
axes[2].set_ylabel("R")

axes[3].plot(Q_arr / (R_arr + 1e-10), linewidth=0.8, color="purple")
axes[3].set_title("Q/R Ratio — Filter Reactivity")
axes[3].set_ylabel("Q/R")

plt.tight_layout()
plt.show()

print(f"Price min:  {np.min(prices):.4f}")
print(f"Price max:  {np.max(prices):.4f}")
print(f"Price mean: {np.mean(prices):.4f}")
print(f"Price std:  {np.std(prices):.4f}")

# Check for jumps between sessions
diffs = np.abs(np.diff(prices))
print(f"\nMax price jump: {np.max(diffs):.4f}")
print(f"Mean price jump: {np.mean(diffs):.4f}")
print(f"Jumps > 1.0: {np.sum(diffs > 1.0)}")
print(f"Jumps > 5.0: {np.sum(diffs > 5.0)}")

print(f"Max price jump: {np.max(np.abs(np.diff(prices))):.4f}")
print(f"Jumps > 1.0: {np.sum(np.abs(np.diff(prices)) > 1.0)}")
print(f"Jumps > 5.0: {np.sum(np.abs(np.diff(prices)) > 5.0)}")

