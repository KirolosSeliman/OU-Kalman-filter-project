"""
IMM FILTER — Interacting Multiple Models
==========================================
Module 3A Layer 3 from the Rigorous Strategy Document.

Architecture:
    Three parallel VB-AKF filters run simultaneously every bar.
    Each represents a distinct market regime:
        Model 1 — Mean-reverting  : low process noise, filter trusts its prediction
        Model 2 — Transitional    : balanced, no strong regime view
        Model 3 — Trending/Volatile: high process noise, filter tracks price aggressively

    At every bar, Bayes theorem reweights each model based on how well it
    predicted the new observation. The model that was least surprised gets
    more probability mass.

    Output:
        x_hat_k  — combined price estimate (weighted average across all 3 models)
        P_k      — combined uncertainty (accounts for inter-model spread)
        p1_arr   — Regime_Score = P(mean-reverting regime) at every bar
                   This replaces the manual ADX/VR score in position sizing.

Key equations:
    Likelihood:     L_i,k  = N(v_i,k ; 0, S_i,k)
                           = (1/sqrt(2π·S_i,k)) · exp(-v_i,k² / (2·S_i,k))
    Probability:    p_i,k  = L_i,k · p_i,k-1  (then normalize to sum=1)
    Combined state: x̂_k   = Σ p_i,k · x̂_i,k
    Combined P:     P_k    = Σ p_i,k · [P_i,k + (x̂_i,k - x̂_k)²]

Integration:
    from imm_filter import imm_filter
    x_hat, P_combined, p1 = imm_filter(session_prices, Q_mle, R_mle)

Run from NoteBook folder:
    python imm_filter.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: SINGLE-BAR VB-AKF STEP
# ═══════════════════════════════════════════════════════════════════════════════

def _vb_step(price_t, state, lambda_=0.98):
    """
    Execute one bar of a VB-AKF filter.

    This is the per-bar version of vb_akf — instead of processing a full
    array, it takes a single new price and the current filter state dict,
    updates it in-place, and returns the updated state plus this bar's
    innovation and innovation covariance.

    Parameters
    ----------
    price_t : float
        New observed price at bar t.
    state : dict
        Current filter state. Keys:
            x_hat   — current state estimate
            P       — current error covariance
            Q_k     — current process noise estimate
            R_k     — current measurement noise estimate
            alpha_R — VB shape parameter for R
            beta_R  — VB rate parameter for R
            alpha_Q — VB shape parameter for Q
            beta_Q  — VB rate parameter for Q
            prev_x  — x_hat from previous bar (for Q update)
    lambda_ : float
        Forgetting factor. Default 0.98 ≈ 50-bar memory.

    Returns
    -------
    state : dict
        Updated in-place.
    innovation : float
        v_k = price_t - x_hat_predicted
    S : float
        Innovation covariance = P_pred + R_k
    """
    # ── PREDICT ───────────────────────────────────────────────────────────────
    x_pred = state["x_hat"]
    P_pred = state["P"] + state["Q_k"]
    S      = P_pred + state["R_k"]

    # ── INNOVATION ────────────────────────────────────────────────────────────
    innovation = price_t - x_pred

    # ── ROBUST GATE: cap innovation for VB update only, not Kalman update ────
    # Compute recent innovation std from the state's innovation history
    sigma_innov = state.get("sigma_innov", abs(innovation) + 1e-8)
    if abs(innovation) > 4.0 * sigma_innov and sigma_innov > 0:
        innovation_vb = np.sign(innovation) * 4.0 * sigma_innov
    else:
        innovation_vb = innovation
    # Update rolling sigma estimate (exponential moving std)
    state["sigma_innov"] = 0.95 * sigma_innov + 0.05 * abs(innovation)

    # ── KALMAN UPDATE (raw innovation) ────────────────────────────────────────
    K               = P_pred / (P_pred + state["R_k"])
    x_hat_new       = x_pred + K * innovation
    P_new           = (1.0 - K) * P_pred
    x_hat_change    = x_hat_new - state["x_hat"]

    # ── VB UPDATE — R (capped innovation) ─────────────────────────────────────
    state["alpha_R"] = lambda_ * state["alpha_R"] + 0.5
    state["beta_R"]  = lambda_ * state["beta_R"]  + 0.5 * (innovation_vb**2 + S)
    state["R_k"]     = state["beta_R"] / max(state["alpha_R"] - 1.0, 1e-8)

    # ── VB UPDATE — Q (x_hat change) ──────────────────────────────────────────
    state["alpha_Q"] = lambda_ * state["alpha_Q"] + 0.5
    state["beta_Q"]  = lambda_ * state["beta_Q"]  + 0.5 * (x_hat_change**2)
    state["Q_k"]     = state["beta_Q"] / max(state["alpha_Q"] - 1.0, 1e-8)

    # ── STORE UPDATED STATE ───────────────────────────────────────────────────
    state["x_hat"] = x_hat_new
    state["P"]     = P_new

    return state, innovation, S


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALISE A VB-AKF STATE DICT FROM MLE VALUES
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state(price_0, Q_init, R_init, alpha_prior=10.0):
    """
    Create a fresh VB-AKF state dict for one IMM model.

    Parameters
    ----------
    price_0 : float
        First observed price — used to initialise x_hat.
    Q_init : float
        Initial process noise (from MLE or scaled from MLE).
    R_init : float
        Initial measurement noise (from MLE or scaled from MLE).
    alpha_prior : float
        Prior strength in bars. Default 10 means prior worth ~10 observations.
        By bar 30 the data dominates regardless of this choice.
    """
    return {
        "x_hat"      : price_0,
        "P"          : 1.0,
        "Q_k"        : Q_init,
        "R_k"        : R_init,
        "alpha_R"    : alpha_prior,
        "beta_R"     : alpha_prior * R_init,
        "alpha_Q"    : alpha_prior,
        "beta_Q"     : alpha_prior * Q_init,
        "sigma_innov": 1e-4,          # warm start for robust gate
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MLE INITIALISATION (copied from kalman_filter.py for standalone operation)
# ═══════════════════════════════════════════════════════════════════════════════

def _kalman_filter_basic(prices, Q, R):
    """Basic 1D Kalman filter — used inside MLE optimizer."""
    N           = len(prices)
    x_hat       = np.full(N, np.nan)
    P           = np.full(N, np.nan)
    innovations = np.full(N, np.nan)

    x_hat[0] = prices[0]
    P[0]     = 1.0

    for t in range(1, N):
        x_pred         = x_hat[t-1]
        P_pred         = P[t-1] + Q
        innovation     = prices[t] - x_pred
        K              = P_pred / (P_pred + R)
        x_hat[t]       = x_pred + K * innovation
        P[t]           = (1.0 - K) * P_pred
        innovations[t] = innovation

    return x_hat, P, innovations


def mle_init(prices):
    """
    Find optimal Q and R by maximising the log-likelihood of the
    innovation sequence. Uses L-BFGS-B with bounds to prevent the
    optimizer from exploring degenerate regions.
    """
    price_diff_var = np.var(np.diff(prices))

    def neg_log_likelihood(params):
        Q, R = params[0], params[1]
        if Q <= 0 or R <= 0:
            return 1e10
        _, P, innovations = _kalman_filter_basic(prices, Q, R)
        S     = P + R
        valid = np.isfinite(S[1:]) & np.isfinite(innovations[1:]) & (S[1:] > 0)
        if valid.sum() < 10:
            return 1e10
        ll = -0.5 * np.sum(
            np.log(S[1:][valid]) + innovations[1:][valid]**2 / S[1:][valid]
        )
        return -ll

    bounds  = [(1e-8, price_diff_var), (1e-8, price_diff_var)]
    x0      = [price_diff_var * 0.1, price_diff_var * 0.1]
    result  = minimize(neg_log_likelihood, x0, method="L-BFGS-B", bounds=bounds)
    return result.x[0], result.x[1]


# ═══════════════════════════════════════════════════════════════════════════════
# IMM FILTER — MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def imm_filter(prices, Q_mle, R_mle, lambda_=0.98, p_init=None):
    """
    Interacting Multiple Models filter.

    Runs three parallel VB-AKF filters, each representing a different
    market regime. Bayesian reweighting every bar produces a continuous
    probability of the mean-reverting regime.

    Parameters
    ----------
    prices : np.ndarray
        Clean price series for one session (no overnight gaps).
    Q_mle : float
        MLE-derived process noise — used as the baseline for model scaling.
    R_mle : float
        MLE-derived measurement noise — used as the baseline.
    lambda_ : float
        VB forgetting factor. Default 0.98.
    p_init : list of 3 floats, optional
        Initial model probabilities [p1, p2, p3]. Must sum to 1.
        Default: equal priors [1/3, 1/3, 1/3].

    Returns
    -------
    x_hat_combined : np.ndarray
        Combined state estimate — weighted average across all 3 models.
    P_combined : np.ndarray
        Combined error covariance — accounts for inter-model spread.
    p1_arr : np.ndarray
        Regime_Score = P(mean-reverting model) at every bar.
        Range [0, 1]. Use this in Module 5 position sizing.
    p_all : np.ndarray, shape (N, 3)
        Full probability vector for all 3 models at every bar.
        Columns: [p_mean_reverting, p_transitional, p_trending].

    Model Initialisation
    --------------------
    Model 1 (Mean-reverting):  Q = 0.3 × Q_mle  R = 1.0 × R_mle
        Low process noise → filter trusts its own prediction more.
        Appropriate when price oscillates around a stable mean.

    Model 2 (Transitional):    Q = 1.0 × Q_mle  R = 1.0 × R_mle
        Balanced. Acts as a buffer. Gains probability during regime
        transitions when neither model 1 nor model 3 fits well.

    Model 3 (Trending):        Q = 3.0 × Q_mle  R = 0.5 × R_mle
        High process noise → filter tracks price aggressively.
        Appropriate when genuine large moves are occurring.
    """
    N = len(prices)

    # ── Model Q/R scaling ────────────────────────────────────────────────────
    # These multipliers are from the strategy document (Module 3A, Table).
    model_configs = [
        {"Q_scale": 0.3, "R_scale": 1.0, "name": "Mean-reverting"},
        {"Q_scale": 1.0, "R_scale": 1.0, "name": "Transitional"},
        {"Q_scale": 3.0, "R_scale": 0.5, "name": "Trending"},
    ]

    # ── Initialise 3 filter states ────────────────────────────────────────────
    states = [
        _init_state(
            prices[0],
            Q_init = cfg["Q_scale"] * Q_mle,
            R_init = cfg["R_scale"] * R_mle,
        )
        for cfg in model_configs
    ]

    # ── Initialise model probabilities ────────────────────────────────────────
    if p_init is None:
        probs = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
    else:
        probs = np.array(p_init, dtype=float)
        probs /= probs.sum()  # enforce normalisation

    # ── Output arrays ─────────────────────────────────────────────────────────
    x_hat_combined = np.full(N, np.nan)
    P_combined     = np.full(N, np.nan)
    p1_arr         = np.full(N, np.nan)
    p_all          = np.full((N, 3), np.nan)

    # Bar 0 — initialisation
    x_hat_combined[0] = prices[0]
    P_combined[0]     = 1.0
    p1_arr[0]         = probs[0]
    p_all[0]          = probs

    # ── Main loop ─────────────────────────────────────────────────────────────
    for t in range(1, N):
        price_t = prices[t]

        # Step 1: Run one VB-AKF step for each model independently
        innovations = np.zeros(3)
        S_vals      = np.zeros(3)
        x_hats      = np.zeros(3)
        P_vals      = np.zeros(3)

        for i, state in enumerate(states):
            states[i], innov, S = _vb_step(price_t, state, lambda_)
            innovations[i] = innov
            S_vals[i]      = S
            x_hats[i]      = states[i]["x_hat"]
            P_vals[i]      = states[i]["P"]

        # Step 2: Bayesian probability update
        # Likelihood: how probable is this innovation under each model?
        # L_i = N(v_i; 0, S_i) = exp(-v_i²/(2·S_i)) / sqrt(2π·S_i)
        S_safe = np.maximum(S_vals, 1e-10)
        log_likelihoods = (
            -0.5 * np.log(2.0 * np.pi * S_safe)
            - 0.5 * innovations**2 / S_safe
        )
        # Subtract max before exp to prevent numerical underflow
        log_likelihoods -= log_likelihoods.max()
        likelihoods = np.exp(log_likelihoods)

        # Multiply by prior probabilities
        unnorm = likelihoods * probs
        total  = unnorm.sum()

        if total < 1e-15:
            # All three models were equally wrong — reset to equal priors
            # This happens during extreme outlier bars
            probs = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        else:
            probs = unnorm / total

        # Step 3: Combined state estimate (weighted average)
        x_hat_k = np.sum(probs * x_hats)

        # Combined P accounts for spread between model estimates
        # P_k = Σ p_i · [P_i + (x̂_i - x̂_k)²]
        P_k = np.sum(probs * (P_vals + (x_hats - x_hat_k)**2))

        # Step 4: Store outputs
        x_hat_combined[t] = x_hat_k
        P_combined[t]     = P_k
        p1_arr[t]         = probs[0]   # mean-reverting probability
        p_all[t]          = probs

    return x_hat_combined, P_combined, p1_arr, p_all


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (same pattern as kalman_filter.py)
# ═══════════════════════════════════════════════════════════════════════════════

def load_sessions(ticker="GLD", period="7d", interval="1m"):
    """
    Download data and return a dict of {date: price_array} per session.
    Using GLD as the default — first test of the new instrument.
    """
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = data.between_time("09:30", "16:00")
    data["date"] = data.index.date

    sessions = {}
    for date, session in data.groupby("date"):
        prices = session["Close"].dropna().values.flatten().astype(float)
        if len(prices) >= 60:
            sessions[date] = prices

    return sessions


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION: DOES p1 CORRELATE WITH ACTUAL SIGNAL QUALITY?
# ═══════════════════════════════════════════════════════════════════════════════

def validate_regime_score(x_hat, p1, prices, window=10):
    """
    Empirically validate that high p1 (mean-reverting regime) correlates
    with actual mean-reverting behavior in the subsequent window.

    For each bar where p1 > 0.6, measure whether the price 10 bars later
    is closer to the current mean than the current price is.
    This is the empirical test of whether p1 is informative.
    """
    N = len(prices)
    high_p1_bars  = np.where(p1 > 0.6)[0]
    low_p1_bars   = np.where(p1 < 0.33)[0]

    def mean_reversion_rate(bars):
        rates = []
        for t in bars:
            if t + window >= N:
                continue
            local_mean  = x_hat[t]
            gap_now     = abs(prices[t]    - local_mean)
            gap_later   = abs(prices[t + window] - local_mean)
            if gap_now > 1e-6:
                rates.append(gap_later / gap_now)
        return np.array(rates)

    high_rates = mean_reversion_rate(high_p1_bars)
    low_rates  = mean_reversion_rate(low_p1_bars)

    print("\n" + "═" * 60)
    print("  IMM REGIME SCORE VALIDATION")
    print("═" * 60)
    print(f"  High p1 (>0.6) bars  : {len(high_p1_bars)}")
    print(f"  Low  p1 (<0.33) bars : {len(low_p1_bars)}")

    if len(high_rates) > 5:
        print(f"\n  Gap ratio after {window} bars (closer to 1 = no reversion, <1 = reversion):")
        print(f"  High p1 mean gap ratio : {np.mean(high_rates):.3f}  (lower = better reversion)")
        print(f"  Low  p1 mean gap ratio : {np.mean(low_rates):.3f}  (should be higher)")
        if np.mean(high_rates) < np.mean(low_rates):
            print("  ✓ CONFIRMED: High p1 bars show better subsequent mean-reversion")
        else:
            print("  ✗ NOT CONFIRMED: p1 is not predicting reversion on this data")
    else:
        print("  Insufficient bars to validate — need more data")
    print("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_imm_results(prices, x_hat, P, p1, p_all, session_date):
    """
    5-panel plot showing IMM output for one session:
      1. Price vs combined estimate
      2. Individual model probabilities (all 3)
      3. Regime score p1 (mean-reverting probability)
      4. Combined P_k (filter uncertainty — use for confidence gate)
      5. Q/R ratio for model 1 (mean-reverting model)
    """
    N = len(prices)
    x = np.arange(N)

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        f"IMM Filter — Interacting Multiple Models\n"
        f"Session: {session_date}  |  GLD 1-min",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(4, 1, hspace=0.55)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    # Panel 0: Price vs combined estimate
    ax0.plot(x, prices, color="#aaaaaa", lw=0.7, label="Raw price", alpha=0.9)
    ax0.plot(x, x_hat,  color="#cc0000", lw=1.4, label="IMM combined x̂")
    ax0.fill_between(x,
        x_hat - 2*np.sqrt(np.maximum(P, 0)),
        x_hat + 2*np.sqrt(np.maximum(P, 0)),
        alpha=0.12, color="#cc0000", label="±2σ (P_k)"
    )
    ax0.set_ylabel("Price ($)")
    ax0.set_title("Price vs IMM Combined Estimate (±2σ uncertainty band)", fontsize=10)
    ax0.legend(loc="upper left", fontsize=8)

    # Panel 1: All 3 model probabilities
    ax1.plot(x, p_all[:, 0], color="#003399", lw=1.2, label="p₁ Mean-reverting")
    ax1.plot(x, p_all[:, 1], color="#888800", lw=1.0, label="p₂ Transitional",  alpha=0.8)
    ax1.plot(x, p_all[:, 2], color="#cc3300", lw=1.0, label="p₃ Trending",      alpha=0.8)
    ax1.axhline(1/3, color="gray", lw=0.6, ls="--", alpha=0.5, label="Equal prior (1/3)")
    ax1.set_ylabel("Probability")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("Model Probabilities — Bayesian update every bar", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8, ncol=2)

    # Panel 2: Regime score with trading threshold
    ax2.fill_between(x, 0, p1, where=(p1 >= 0.5),
                     alpha=0.35, color="#003399", label="p₁ ≥ 0.5 (trade)")
    ax2.fill_between(x, 0, p1, where=(p1 < 0.5),
                     alpha=0.15, color="#cc3300", label="p₁ < 0.5 (suppress)")
    ax2.plot(x, p1, color="#003399", lw=1.2)
    ax2.axhline(0.5, color="red",    lw=0.8, ls="--", label="Threshold 0.5")
    ax2.axhline(0.7, color="green",  lw=0.6, ls=":",  label="High conviction 0.7")
    ax2.set_ylabel("p₁ (Regime Score)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Regime Score = P(Mean-Reverting)  →  replaces manual ADX/VR score", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)

    # Panel 3: Combined P_k
    P_plot = np.sqrt(np.maximum(P, 0))  # express as std dev
    ax3.plot(x, P_plot, color="#884400", lw=1.0)
    p_max_line = np.nanpercentile(P_plot, 80)  # suggest P_max as 80th percentile
    ax3.axhline(p_max_line, color="red", lw=0.8, ls="--",
                label=f"Suggested P_max = {p_max_line:.4f} (80th pct)")
    ax3.set_ylabel("√P_k (uncertainty)")
    ax3.set_xlabel(f"Bar index (09:30 to 16:00)")
    ax3.set_title("Filter Uncertainty √P_k  →  use P_max threshold to gate z-score signals", fontsize=10)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("Downloading GLD 1-min data...")
    sessions = load_sessions(ticker="GLD", period="7d", interval="1m")
    print(f"Sessions loaded: {len(sessions)}")

    all_p1     = []
    all_prices = []

    for date, prices in sessions.items():
        print(f"\nProcessing {date} — {len(prices)} bars")

        # MLE initialisation for this session
        Q_mle, R_mle = mle_init(prices)
        print(f"  MLE → Q={Q_mle:.6f}  R={R_mle:.6f}  Q/R={Q_mle/R_mle:.4f}")

        # Run IMM
        x_hat, P, p1, p_all = imm_filter(prices, Q_mle, R_mle)

        # Statistics for this session
        print(f"  p1 mean: {np.nanmean(p1):.3f}  "
              f"p1 > 0.5: {np.mean(p1[~np.isnan(p1)] > 0.5)*100:.1f}%  "
              f"p1 > 0.7: {np.mean(p1[~np.isnan(p1)] > 0.7)*100:.1f}%")

        all_p1.extend(p1.tolist())
        all_prices.extend(prices.tolist())

    # ── Cross-session summary ─────────────────────────────────────────────────
    all_p1     = np.array(all_p1,     dtype=float)
    all_prices = np.array(all_prices, dtype=float)

    valid = ~np.isnan(all_p1)
    print("\n" + "═" * 60)
    print("  CROSS-SESSION IMM SUMMARY  —  GLD")
    print("═" * 60)
    print(f"  Total bars          : {valid.sum()}")
    print(f"  Mean regime score   : {np.mean(all_p1[valid]):.3f}")
    print(f"  p1 > 0.5 (trade)    : {np.mean(all_p1[valid] > 0.5)*100:.1f}%  of bars")
    print(f"  p1 > 0.7 (high conf): {np.mean(all_p1[valid] > 0.7)*100:.1f}%  of bars")
    print(f"  p1 < 0.33 (suppress): {np.mean(all_p1[valid] < 0.33)*100:.1f}%  of bars")
    print("═" * 60)
    print("\n  Compare to QQQ: gate was open only 12% of bars.")
    print("  GLD's mean-reverting regime probability should be higher")
    print("  because GLD tracks a mean-reverting physical commodity.")

    # ── Plot the last session in full detail ──────────────────────────────────
    last_date   = list(sessions.keys())[-1]
    last_prices = sessions[last_date]
    Q_mle, R_mle = mle_init(last_prices)
    x_hat, P, p1, p_all = imm_filter(last_prices, Q_mle, R_mle)

    validate_regime_score(x_hat, p1, last_prices, window=10)
    plot_imm_results(last_prices, x_hat, P, p1, p_all, last_date)
