import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

# ── 1. Download Data ──────────────────────────────────────────────────────────
data = yf.download("AAPL", period="1mo", interval="5m", auto_adjust=True)

print("Columns:", data.columns.tolist())
print("Index type:", type(data.index[0]))
print("Index tz:", data.index.tz)
print("Shape:", data.shape)
print(data.head(3))

# Fix 1: flatten multi-level columns yfinance sometimes returns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Fix 2: strip timezone so between_time works
data.index = pd.to_datetime(data.index)
if data.index.tz is not None:
    data.index = data.index.tz_localize(None)

data = data.between_time("09:30", "16:00")
data["date"] = data.index.date

print(f"Total bars: {len(data)}")
print(f"Sessions found: {data['date'].nunique()}")

# ── 2. Rolling ADF Function ───────────────────────────────────────────────────
def rolling_adf(prices, window=15):
    pvalues = np.full(len(prices), np.nan)
    for t in range(window, len(prices)):
        window_slice = prices[t-window:t]
        result = adfuller(window_slice)
        pvalues[t] = result[1]
    return pvalues

# ── 3. Classifier ─────────────────────────────────────────────────────────────
def classify(pvalue):
    if np.isnan(pvalue):
        return "NOT_ENOUGH_DATA"
    elif pvalue < 0.05:
        return "MEAN_REVERTING"
    else:
        return "RANDOM_WALK"

# ── 4. Run Per Session ────────────────────────────────────────────────────────
all_pvalues = []
all_prices = []

for date, session in data.groupby("date"):
    session_prices = session["Close"].dropna().values.flatten()
    if len(session_prices) < 15:
        continue
    pvalues = rolling_adf(session_prices, window=15)
    all_pvalues.extend(pvalues)
    all_prices.extend(session_prices)

all_pvalues = np.array(all_pvalues, dtype=float)
all_prices = np.array(all_prices, dtype=float)

print(f"Total bars processed: {len(all_prices)}")

# ── 5. Gate Stats ─────────────────────────────────────────────────────────────
valid = ~np.isnan(all_pvalues)
gate_open_pct = np.mean(all_pvalues[valid] < 0.05) * 100
print(f"Gate OPEN  {gate_open_pct:.1f}% of bars")
print(f"Gate CLOSED {100 - gate_open_pct:.1f}% of bars")

# ── 6. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle("AAPL Stationarity Gate — 5min bars, 60 day window", fontsize=13)

axes[0].plot(all_prices, linewidth=0.8, color="steelblue")
axes[0].set_title("AAPL Price (market hours only, gaps removed)")
axes[0].set_ylabel("Price")

axes[1].plot(all_pvalues, linewidth=0.8, color="steelblue")
axes[1].axhline(y=0.05, color="red", linestyle="--", linewidth=1.5, label="p = 0.05 threshold")
axes[1].fill_between(range(len(all_pvalues)), 0, 0.05, alpha=0.15, color="green", label=f"MEAN_REVERTING ({gate_open_pct:.1f}% of bars)")
axes[1].set_title("Rolling ADF P-Value (window=60 bars)")
axes[1].set_ylabel("P-Value")
axes[1].set_xlabel("Bar")
axes[1].legend()

plt.tight_layout()
plt.show()


# the null hypothethis is that the serie is a random walk,
# the p result is how likely the hypothethis is true, so a low p result ,means not a random walk