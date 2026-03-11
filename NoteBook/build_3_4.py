"""
BUILD 3.4 — Kalman + OU Pipeline
=================================
Feed VB-AKF filtered prices into the OU parameter estimator.
Compare OU parameters estimated on:
  (a) raw prices
  (b) VB-AKF filtered prices

The core question: does filtering reduce parameter variance?
A smoother phi = more reliable half-life = better exit timing.

Pipeline per session:
  raw prices  →  rolling OLS  →  (phi_raw, mu_raw, sigma_raw, HL_raw)
  raw prices  →  VB-AKF  →  x_hat  →  rolling OLS  →  (phi_filt, ...)

Run from NoteBook folder:
  python build_3_4.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import sys
import os

# ── Import from existing modules ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kalman_filter import mle_init, vb_akf
from estimate_ou import estimate_ou

# ── Config ────────────────────────────────────────────────────────────────────
TICKER       = "QQQ"
WINDOW       = 60     # bars for rolling OU estimation
MIN_SESSION  = 120    # skip sessions shorter than this (not enough data)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="7d", interval="1m", auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data.between_time("09:30", "16:00")
    data = data[["Close"]].dropna()
    data["date"] = data.index.date
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING OU — applies to any price array
# ═══════════════════════════════════════════════════════════════════════════════

def rolling_ou_params(prices: np.ndarray, window: int = 60):
    """
    Slide a window of length `window` across prices.
    At each bar t >= window, estimate OU on prices[t-window:t].

    Returns arrays of length len(prices), with NaN for t < window.
    """
    N = len(prices)
    phi_arr = np.full(N, np.nan)
    mu_arr  = np.full(N, np.nan)
    sig_arr = np.full(N, np.nan)
    hl_arr  = np.full(N, np.nan)

    for t in range(window, N):
        window_prices = prices[t - window : t]
        try:
            phi, mu, sigma, hl = estimate_ou(window_prices)
            # Validity check — discard degenerate estimates
            if not (0.0 < phi < 1.0):
                continue
            if hl <= 0 or hl > 200:
                continue
            phi_arr[t] = phi
            mu_arr[t]  = mu
            sig_arr[t] = sigma
            hl_arr[t]  = hl
        except Exception:
            continue

    return phi_arr, mu_arr, sig_arr, hl_arr


# ═══════════════════════════════════════════════════════════════════════════════
# PER-SESSION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_all_sessions(data: pd.DataFrame, window: int = 60):
    """
    For each session:
      1. Run MLE → VB-AKF → get x_hat
      2. Run rolling OU on raw prices
      3. Run rolling OU on x_hat (filtered)

    Returns dict of per-session results assembled into full arrays.
    """
    raw_prices_all  = []
    filt_prices_all = []
    phi_raw_all     = []
    phi_filt_all    = []
    mu_raw_all      = []
    mu_filt_all     = []
    sig_raw_all     = []
    sig_filt_all    = []
    hl_raw_all      = []
    hl_filt_all     = []
    bar_indices     = []
    session_bounds  = []   # (start_idx, end_idx, date) for shading

    global_idx = 0

    for date, session in data.groupby("date"):
        prices = session["Close"].dropna().values.flatten().astype(float)
        N = len(prices)

        if N < MIN_SESSION:
            print(f"  Skipping {date} — only {N} bars")
            continue

        # ── Step 1: VB-AKF on this session ────────────────────────────────
        Q_init, R_init = mle_init(prices)
        x_hat, P, innovations, Q_arr, R_arr = vb_akf(prices, Q_init, R_init)

        # ── Step 2: Rolling OU on raw prices ──────────────────────────────
        phi_r, mu_r, sig_r, hl_r = rolling_ou_params(prices, window)

        # ── Step 3: Rolling OU on filtered prices ─────────────────────────
        phi_f, mu_f, sig_f, hl_f = rolling_ou_params(x_hat, window)

        # ── Collect ───────────────────────────────────────────────────────
        session_start = global_idx
        raw_prices_all.extend(prices)
        filt_prices_all.extend(x_hat)
        phi_raw_all.extend(phi_r)
        phi_filt_all.extend(phi_f)
        mu_raw_all.extend(mu_r)
        mu_filt_all.extend(mu_f)
        sig_raw_all.extend(sig_r)
        sig_filt_all.extend(sig_f)
        hl_raw_all.extend(hl_r)
        hl_filt_all.extend(hl_f)
        bar_indices.extend(range(global_idx, global_idx + N))
        session_bounds.append((session_start, global_idx + N - 1, str(date)))
        global_idx += N

    results = {
        "raw"      : np.array(raw_prices_all,  dtype=float),
        "filtered" : np.array(filt_prices_all, dtype=float),
        "phi_raw"  : np.array(phi_raw_all,     dtype=float),
        "phi_filt" : np.array(phi_filt_all,    dtype=float),
        "mu_raw"   : np.array(mu_raw_all,      dtype=float),
        "mu_filt"  : np.array(mu_filt_all,     dtype=float),
        "sig_raw"  : np.array(sig_raw_all,     dtype=float),
        "sig_filt" : np.array(sig_filt_all,    dtype=float),
        "hl_raw"   : np.array(hl_raw_all,      dtype=float),
        "hl_filt"  : np.array(hl_filt_all,     dtype=float),
        "sessions" : session_bounds,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS — the quantitative case for filtering
# ═══════════════════════════════════════════════════════════════════════════════

def print_comparison_stats(results: dict):
    """
    Print mean, std, and variance-reduction ratio for each parameter.
    The key metric is std_raw / std_filt — how much smoother is the
    filtered version? A ratio > 1 means filtering is working.
    """
    params = [
        ("phi",   "φ (mean-reversion speed)",     results["phi_raw"],  results["phi_filt"]),
        ("mu",    "μ (long-run mean)",             results["mu_raw"],   results["mu_filt"]),
        ("sigma", "σ (residual std)",              results["sig_raw"],  results["sig_filt"]),
        ("hl",    "Half-life (bars)",              results["hl_raw"],   results["hl_filt"]),
    ]

    print("\n" + "═" * 70)
    print("  PARAMETER STABILITY: RAW vs VB-AKF FILTERED")
    print("═" * 70)
    print(f"  {'Parameter':<28} {'Raw std':>10} {'Filt std':>10} {'Ratio':>10}")
    print("─" * 70)

    for key, label, raw, filt in params:
        valid_raw  = raw[np.isfinite(raw)]
        valid_filt = filt[np.isfinite(filt)]
        if len(valid_raw) < 10 or len(valid_filt) < 10:
            print(f"  {label:<28}  insufficient data")
            continue
        std_raw  = np.std(valid_raw)
        std_filt = np.std(valid_filt)
        ratio    = std_raw / std_filt if std_filt > 0 else np.nan
        print(f"  {label:<28} {std_raw:>10.4f} {std_filt:>10.4f} {ratio:>9.2f}x")

    print("═" * 70)
    print("  Ratio > 1 means filtering reduced parameter variance.")
    print("  Ratio on φ is most important — it drives half-life reliability.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(results: dict):
    x     = np.arange(len(results["raw"]))
    sesh  = results["sessions"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Build 3.4 — Raw vs VB-AKF Filtered OU Parameters\n"
        f"Rolling window = {WINDOW} bars  |  {TICKER} 1-min",
        fontsize=14, fontweight="bold"
    )

    gs = gridspec.GridSpec(5, 1, hspace=0.55)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])
    ax4 = fig.add_subplot(gs[4])

    # ── Shade sessions alternately ────────────────────────────────────────
    colors_shade = ["#f0f4ff", "#fff8f0"]
    for axes in [ax0, ax1, ax2, ax3, ax4]:
        for i, (s, e, _) in enumerate(sesh):
            axes.axvspan(s, e, alpha=0.25, color=colors_shade[i % 2], linewidth=0)

    # ── Session boundary vertical lines ───────────────────────────────────
    for axes in [ax1, ax2, ax3, ax4]:
        for i, (s, e, d) in enumerate(sesh):
            axes.axvline(s, color="gray", lw=0.6, ls="--", alpha=0.5)

    # ── Panel 0: Price — raw vs filtered ──────────────────────────────────
    ax0.plot(x, results["raw"],      color="#aaaaaa", lw=0.6, label="Raw price", alpha=0.8)
    ax0.plot(x, results["filtered"], color="#cc0000", lw=1.2, label="VB-AKF x̂")
    ax0.set_ylabel("Price ($)")
    ax0.set_title("Price: Raw vs VB-AKF Filtered", fontsize=10)
    ax0.legend(loc="upper left", fontsize=8)
    ax0.set_xlim(0, len(x))

    # ── Panel 1: φ ────────────────────────────────────────────────────────
    ax1.plot(x, results["phi_raw"],  color="#6699cc", lw=0.8, label="φ raw",       alpha=0.7)
    ax1.plot(x, results["phi_filt"], color="#003399", lw=1.4, label="φ filtered")
    ax1.axhline(1.0, color="red",  lw=0.8, ls=":", alpha=0.6, label="φ=1 (RW)")
    ax1.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax1.set_ylabel("φ")
    ax1.set_ylim(-0.3, 1.3)
    ax1.set_title("φ — Mean-Reversion Speed  (smoother = more reliable half-life)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(0, len(x))

    # ── Panel 2: μ ────────────────────────────────────────────────────────
    ax2.plot(x, results["mu_raw"],  color="#99bb55", lw=0.8, label="μ raw",       alpha=0.7)
    ax2.plot(x, results["mu_filt"], color="#336600", lw=1.4, label="μ filtered")
    ax2.plot(x, results["raw"],     color="#cccccc", lw=0.5, label="raw price",    alpha=0.5)
    ax2.set_ylabel("μ ($)")
    ax2.set_title("μ — Long-Run Mean  (should track price, not diverge from it)", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_xlim(0, len(x))

    # ── Panel 3: σ ────────────────────────────────────────────────────────
    ax3.plot(x, results["sig_raw"],  color="#cc8800", lw=0.8, label="σ raw",       alpha=0.7)
    ax3.plot(x, results["sig_filt"], color="#884400", lw=1.4, label="σ filtered")
    ax3.set_ylabel("σ")
    ax3.set_title("σ — Residual Std  (filtered should be lower — tick noise removed)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlim(0, len(x))

    # ── Panel 4: Half-life ────────────────────────────────────────────────
    ax4.plot(x, results["hl_raw"],  color="#cc44aa", lw=0.8, label="HL raw",       alpha=0.7)
    ax4.plot(x, results["hl_filt"], color="#660044", lw=1.4, label="HL filtered")
    ax4.axhline(4,  color="red",   lw=0.8, ls=":", alpha=0.6, label="4 bars (scalp)")
    ax4.axhline(30, color="orange",lw=0.8, ls=":", alpha=0.6, label="30 bars (max)")
    ax4.set_ylabel("HL (bars)")
    ax4.set_xlabel("Bar index (across all sessions)")
    ax4.set_ylim(0, 80)
    ax4.set_title("Half-Life (bars)  (smoother filtered line = more reliable exit timing)", fontsize=10)
    ax4.legend(loc="upper right", fontsize=8)
    ax4.set_xlim(0, len(x))

    # ── Session date labels on bottom panel ───────────────────────────────
    for s, e, d in sesh:
        mid = (s + e) // 2
        ax4.text(mid, 73, d, ha="center", va="top", fontsize=7, color="#444444")

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Downloading {TICKER} 1-min data...")
    data = load_data(TICKER)
    print(f"Total bars loaded: {len(data)}")
    print(f"Sessions: {data['date'].nunique()}")

    print("\nProcessing sessions...")
    results = process_all_sessions(data, window=WINDOW)

    total_valid_raw  = np.sum(np.isfinite(results["phi_raw"]))
    total_valid_filt = np.sum(np.isfinite(results["phi_filt"]))
    print(f"Valid OU estimates — raw: {total_valid_raw}  |  filtered: {total_valid_filt}")

    print_comparison_stats(results)
    plot_results(results)
