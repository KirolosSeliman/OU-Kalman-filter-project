import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize


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




# X_hat, is the estimate of the true value
# X_hate_pred is the estimated prediction of the true value
def kalman_filter(prices, Q, R):
    x_hat      = np.full(len(prices), np.nan)
    P          = np.full(len(prices), np.nan)
    K_arr      = np.full(len(prices), np.nan)
    innovations = np.full(len(prices), np.nan)  # ← must be an array

    x_hat[0] = prices[0]
    P[0]     = 1.0

    for t in range(1, len(prices)):
        # PREDICT
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + Q

        # UPDATE
        innovation      = prices[t] - x_pred  # ← scalar this bar
        K               = P_pred / (P_pred + R)
        x_hat[t]        = x_pred + K * innovation
        P[t]            = (1 - K) * P_pred
        K_arr[t]        = K
        innovations[t]  = innovation  # ← store in array

    return x_hat, P, K_arr, innovations

x_hat, P, K_arr, innovations = kalman_filter(prices, Q=0.01, R=1.0)




plt.figure(figsize=(14, 5))
plt.plot(prices, linewidth=0.8, alpha=0.5, color="steelblue", label="Raw prices")
plt.plot(x_hat, linewidth=1.2, color="red", label="Kalman estimate")
plt.title("Kalman Filter — Q=0.01, R=1.0")
plt.legend()
plt.show()


x_hat, P, K_arr, innovations= kalman_filter(prices, Q=0.01, R=1.0)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Price vs filtered estimate
axes[0].plot(prices, linewidth=0.8, alpha=0.5, color="steelblue", label="Raw prices")
axes[0].plot(x_hat, linewidth=1.2, color="red", label="Kalman x̂")
axes[0].set_title("Raw Price vs Kalman Estimate")
axes[0].legend()

# Uncertainty P over time
axes[1].plot(P, linewidth=0.8, color="orange")
axes[1].set_title("P — Filter Uncertainty (should drop and stabilize)")
axes[1].set_ylabel("P")

# Kalman Gain K over time
axes[2].plot(K_arr, linewidth=0.8, color="green")
axes[2].set_title("K — Kalman Gain (how much filter trusts new observations)")
axes[2].set_ylabel("K")

plt.tight_layout()
plt.show()

x_hat2, P2, K2, innovations = kalman_filter(prices, Q=1.0, R=0.01)

plt.figure(figsize=(14, 4))
plt.plot(prices, linewidth=0.8, alpha=0.5, color="steelblue", label="Raw prices")
plt.plot(x_hat, linewidth=1.2, color="red", label="Q=0.01 R=1.0 (smooth)")
plt.plot(x_hat2, linewidth=1.2, color="green", label="Q=1.0 R=0.01 (reactive)")
plt.title("Q/R Tradeoff")
plt.legend()
plt.show()


x_hat1, _, _ ,_= kalman_filter(prices, Q=0.01, R=1.0)   # very smooth
x_hat2, _, _,_ = kalman_filter(prices, Q=0.1,  R=1.0)   # moderate
x_hat3, _, _ ,_= kalman_filter(prices, Q=1.0,  R=1.0)   # reactive

plt.figure(figsize=(14, 5))
plt.plot(prices, linewidth=0.8, alpha=0.4, color="steelblue", label="Raw")
plt.plot(x_hat1, linewidth=1.2, color="red",    label="Q=0.01 (smooth)")
plt.plot(x_hat2, linewidth=1.2, color="orange", label="Q=0.1  (moderate)")
plt.plot(x_hat3, linewidth=1.2, color="green",  label="Q=1.0  (reactive)")
plt.title("Kalman Filter — Q/R Tradeoff")
plt.legend()
plt.show()



def mle_init(prices):
    
    def neg_log_likelihood(params):
        Q = params[0]
        R = params[1]
        
        # guard against negative values
        if Q <= 0 or R <= 0:
            return 1e10
        
        # run filter with this Q and R
        x_hat, P, K, innovations = kalman_filter(prices, Q, R)
        
        # compute log-likelihood
        S = P + R  # innovation covariance     [1:] means skip the first element
        log_likelihood = -0.5 * np.sum(np.log(S[1:]) + innovations[1:]**2 / S[1:])
        # its a sum of : 1. how wrong was the prediction at this bar divided by how wrong we expected to be 
        # so it gives you a normalized answer, was the error bigger, or smaller then expected
        
        return -log_likelihood  # negative because scipy minimizes
    
    # starting guess
    initial_params = [0.1, 0.1]
    
    result = minimize(neg_log_likelihood, initial_params, method="Nelder-Mead")
    
    Q_opt = result.x[0]
    R_opt = result.x[1]
    
    return Q_opt, R_opt

Q_opt, R_opt = mle_init(prices[:200])  # use first 200 bars so it runs fast

print(f"Optimal Q: {Q_opt:.6f}")
print(f"Optimal R: {R_opt:.6f}")
print(f"Q/R ratio: {Q_opt/R_opt:.4f}")

# Compare filters side by side
x_hat_manual, _, _, _ = kalman_filter(prices, Q=0.1, R=1.0)
x_hat_mle,    _, _, _ = kalman_filter(prices, Q=Q_opt, R=R_opt)

plt.figure(figsize=(14, 5))
plt.plot(prices[:500], linewidth=0.8, alpha=0.4, color="steelblue", label="Raw")
plt.plot(x_hat_manual[:500], linewidth=1.2, color="red",   label="Manual Q=0.1 R=1.0")
plt.plot(x_hat_mle[:500],    linewidth=1.2, color="green", label=f"MLE Q={Q_opt:.4f} R={R_opt:.4f}")
plt.title("Manual Q/R vs MLE-derived Q/R")
plt.legend()
plt.show()

