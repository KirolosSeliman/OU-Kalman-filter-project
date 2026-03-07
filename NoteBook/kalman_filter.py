import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

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
    x_hat = np.full(len(prices), np.nan)
    P     = np.full(len(prices), np.nan)
    K_arr = np.full(len(prices), np.nan)

    # Initialize
    x_hat[0] = prices[0]
    P[0]     = 1.0

    for t in range(1, len(prices)):
        # PREDICT
        x_pred = x_hat[t-1]
        P_pred = P[t-1] + Q

        # UPDATE
        K         = P_pred / (P_pred + R)
        x_hat[t]  = x_pred + K * (prices[t] - x_pred)
        P[t]      = (1 - K) * P_pred
        K_arr[t]  = K

    return x_hat, P, K_arr

x_hat, P, K_arr = kalman_filter(prices, Q=0.01, R=1.0)

plt.figure(figsize=(14, 5))
plt.plot(prices, linewidth=0.8, alpha=0.5, color="steelblue", label="Raw prices")
plt.plot(x_hat, linewidth=1.2, color="red", label="Kalman estimate")
plt.title("Kalman Filter — Q=0.01, R=1.0")
plt.legend()
plt.show()



    