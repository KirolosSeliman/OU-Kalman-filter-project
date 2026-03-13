"""
Phase 4–5: Signal, Risk, Execution, Adaptive Weights, and Walk-Forward

This module assumes upstream pipeline provides, per session, a dict `result` with at least:
    - 'spread' : np.ndarray, spread prices (GLD - beta*IAU) or similar
    - 'x_hat'  : np.ndarray, Kalman/IMM combined estimate
    - 'P'      : np.ndarray, Kalman error covariance
    - 'p1'     : np.ndarray, mean-reverting regime probability (IMM)
    - 'z_score': np.ndarray, (spread - x_hat)/sqrt(P)
    - 'beta'   : float, hedge ratio
    - 'GLD'    : np.ndarray, GLD prices
    - 'IAU'    : np.ndarray, IAU prices

Upstream functions expected (from your existing code):
    from imm_filter import load_spread_data, split_sessions, run_pipeline, ou_params_final
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import yfinance as yf
from imm_filter import load_spread_data, split_sessions, run_pipeline, ou_params_final  # adjust import if needed

from datetime import datetime, timedelta, timezone

print("run_pipeline from module:", run_pipeline.__module__)

def load_gld_iau_intraday_chunked(
    days_back=60,
    chunk_days=7,
    interval="1m",
    tz="America/New_York",
):
    """
    Download GLD/IAU intraday data in chunks to respect Yahoo's 1m limit (~7-8 days).

    Parameters
    ----------
    days_back : int
        How many calendar days back from today.
    chunk_days : int
        Length of each chunk in days (<= 7 recommended for 1m).
    """

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    all_chunks = []

    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=chunk_days), end)

        data = yf.download(
            tickers=["GLD", "IAU"],
            start=cur_start,
            end=cur_end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"].copy()
                close = close.rename(columns={c: c for c in close.columns})
            else:
                # Single-ticker fallback (shouldn't happen here)
                close = data[["Close"]].copy()
                close.columns = ["GLD"]

            all_chunks.append(close)

        cur_start = cur_end

    if not all_chunks:
        print("load_gld_iau_intraday_chunked: no data downloaded.")
        return pd.DataFrame()

    combined = pd.concat(all_chunks).sort_index()
    combined.index = pd.to_datetime(combined.index)

    if combined.index.tz is None:
        combined = combined.tz_localize("UTC").tz_convert(tz)
    else:
        combined = combined.tz_convert(tz)

    # Regular US session only
    combined = combined.between_time("09:30", "16:00")
    combined = combined.dropna(subset=["GLD", "IAU"])

    return combined

def load_gld_iau_intraday(period="60d", interval="1m", tz="America/New_York"):
    """
    Download and concatenate GLD/IAU intraday data for up to `period` (max 60d for interval<1d).

    Returns
    -------
    combined : pd.DataFrame
        Columns: ['GLD', 'IAU'], index: timezone-aware DatetimeIndex (NY time),
        filtered to regular session 09:30–16:00.
    """
    tickers = ["GLD", "IAU"]

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Handle MultiIndex columns (ticker, field) → keep Close
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
        close = close.rename(columns={c: c for c in close.columns})
    else:
        # Single ticker case fallback
        close = data[["Close"]].copy()
        close.columns = tickers[:1]

    # Ensure timezone and regular session only
    close.index = pd.to_datetime(close.index)
    if close.index.tz is None:
        close = close.tz_localize("UTC").tz_convert(tz)
    else:
        close = close.tz_convert(tz)

    # Filter to regular US hours 09:30–16:00
    close = close.between_time("09:30", "16:00")

    # Drop rows where either leg is missing
    combined = close.dropna(subset=["GLD", "IAU"])

    return combined
# --------------------------------------------------------------------------
# Layer 1: Raw signal and validity factors
# --------------------------------------------------------------------------

def raw_signal(z_score, z_scale=2.0):
    """
    Raw directional signal: -tanh(z / z_scale).
    Positive when price below mean (buy), negative when above (sell).
    """
    z_score = np.asarray(z_score, dtype=float)
    return +np.tanh(z_score / z_scale)


def layer1_validity_factors(kalman_result, ou_params, p_max_factor=3.0, gate_open=True):
    """
    Layer 1 validity factors: Gate, Filter, Regime, Cost.

    GateFactor:   1 if stationarity gate open, else 0 (placeholder flag here).
    FilterFactor: max(0, 1 - P / P_max), where P_max = p_max_factor * median(P).
    RegimeFactor: p1 (mean-reverting regime probability).
    CostFactor:   1 - tc / expected_edge, clipped at 0.

    Returns
    -------
    gate_factor, filter_factor, regime_factor, cost_factor : np.ndarray
    """
    P = np.asarray(kalman_result["P"], dtype=float)
    p1 = np.asarray(kalman_result["p1"], dtype=float)
    z = np.asarray(kalman_result["z_score"], dtype=float)

    # Gate: for now, a simple boolean flag applied to all bars.
    gate_factor = np.ones_like(P) if gate_open else np.zeros_like(P)

    # Filter: 0 when P very large, 1 when P small.
    P_med = np.median(P)
    P_max = p_max_factor * P_med
    filter_factor = np.maximum(0.0, 1.0 - P / np.maximum(P_max, 1e-12))

    # Regime: direct p1
    regime_factor = np.clip(p1, 0.0, 1.0)

    # Cost: based on OU parameters
    cost_factor = compute_cost_factor(z, ou_params)

    return gate_factor, filter_factor, regime_factor, cost_factor


def compute_cost_factor(z_score, ou_params, tc_bps=5):
    """
    Transaction cost adjustment.

    expected_edge ∝ |z| * kappa * HL_min
    CostFactor = max(0, 1 - tc / expected_edge)
    """
    z = np.asarray(z_score, dtype=float)
    kappa = float(ou_params["kappa"])
    hl_min = float(ou_params["half_life_min"])

    tc_decimal = tc_bps / 10_000.0  # 5 bps -> 0.0005
    expected_edge = np.abs(z) * kappa * hl_min
    cost_factor = np.maximum(0.0, 1.0 - tc_decimal / np.maximum(expected_edge, 1e-6))
    return cost_factor


def layer1_complete(kalman_result, ou_params, p_max_factor=3.0, gate_open=True):
    """
    Full Layer 1 SignalScore: RawSignal * Gate * Filter * Regime * Cost.
    """
    z = np.asarray(kalman_result["z_score"], dtype=float)
    raw = raw_signal(z, z_scale=2.0)
    gate, filt, regime, cost = layer1_validity_factors(
        kalman_result, ou_params, p_max_factor=p_max_factor, gate_open=gate_open
    )
    signal_score = raw * gate * filt * regime * cost
    return signal_score, (gate, filt, regime, cost)


# --------------------------------------------------------------------------
# Trade-level state: HL_entry and trade_age
# --------------------------------------------------------------------------

def build_entry_state(signal_scores, hl_series, entry_threshold=0.30):
    """
    Build trade-level HL_entry and trade_age from SignalScore and current HL.

    Simple single-position logic:
        - When flat and |SignalScore| >= entry_threshold -> enter.
        - Direction = sign(SignalScore)
        - HL_entry locked from hl_series at entry bar.
        - trade_age increments each bar while in trade.
        - Exit when SignalScore changes sign (direction conflict).
    """
    sig = np.asarray(signal_scores, dtype=float)
    hl_series = np.asarray(hl_series, dtype=float)

    N = len(sig)
    hl_series[hl_series <= 0] = np.nan

    hl_entry = np.zeros(N, dtype=float)
    trade_age = np.zeros(N, dtype=float)
    position_state = np.zeros(N, dtype=int)

    in_trade = False
    current_dir = 0
    current_hl_entry = 0.0
    current_age = 0

    # Fallback HL if needed
    hl_fallback = np.nanmedian(hl_series)
    if not np.isfinite(hl_fallback):
        hl_fallback = 20.0

    for t in range(N):
        s = sig[t]

        if not in_trade:
            if np.abs(s) >= entry_threshold:
                in_trade = True
                current_dir = 1 if s > 0 else -1
                hl_t = hl_series[t]
                if not np.isfinite(hl_t):
                    hl_t = hl_fallback
                current_hl_entry = max(hl_t, 1.0)
                current_age = 0
        else:
            # Hard exit: signal conflict
            if s * current_dir <= 0.0:
                in_trade = False
                current_dir = 0
                current_hl_entry = 0.0
                current_age = 0

        if in_trade:
            hl_entry[t] = current_hl_entry
            trade_age[t] = current_age
            position_state[t] = current_dir
            current_age += 1
        else:
            hl_entry[t] = 0.0
            trade_age[t] = 0.0
            position_state[t] = 0

    return hl_entry, trade_age, position_state


# --------------------------------------------------------------------------
# Adaptive Weight Estimator (RiskScore weights)
# --------------------------------------------------------------------------

class AdaptiveWeightEstimator:
    """
    Adaptive weights for 5 risk factors:

        [ZRisk, RegimeRisk, FilterRisk, TimeRisk, HLJumpRisk]

    Layer A: empirical Spearman correlation with adverse move.
    Layer B: regime-conditional scaling.
    Layer C: stability-based blend with prior weights.
    """

    def __init__(self, window=60, stability_min_obs=30, dampening=0.70):
        self.window = window
        self.stability_min_obs = stability_min_obs
        self.dampening = dampening

        # Prior weights (mean-reverting regime) from design doc
        self.prior_weights = np.array([0.35, 0.15, 0.20, 0.20, 0.10], dtype=float)

        # History buffers
        self.factor_history = []   # list of 5-element vectors
        self.outcome_history = []  # list of scalars

        self.weights = self.prior_weights.copy()

    def record_outcome(self, factors_t, outcome_t):
        """
        Record factor vector and realized adverse outcome.

        factors_t : array-like of length 5
        outcome_t : float
        """
        f = np.asarray(factors_t, dtype=float).ravel()
        if f.shape[0] != 5:
            raise ValueError("factors_t must have length 5")
        self.factor_history.append(f)
        self.outcome_history.append(float(outcome_t))

        if len(self.factor_history) > self.window:
            self.factor_history = self.factor_history[-self.window:]
            self.outcome_history = self.outcome_history[-self.window:]

    def update_weights(self, regime_score, adf_pvalue=None):
        """
        Update weights based on historical factor-outcome pairs and current regime.

        regime_score : float in [0, 1]
        adf_pvalue   : float or None
        """
        n = len(self.factor_history)
        if n < self.stability_min_obs:
            self.weights = self.prior_weights.copy()
            return self.weights

        X = np.vstack(self.factor_history)         # [n, 5]
        y = np.asarray(self.outcome_history)       # [n]

        # Layer A: Spearman |rho| per factor
        layerA = np.zeros(5)
        for i in range(5):
            rho, _ = stats.spearmanr(X[:, i], y)
            if not np.isfinite(rho):
                rho = 0.0
            layerA[i] = abs(rho)

        if layerA.sum() > 0:
            layerA /= layerA.sum()
        else:
            layerA[:] = 1.0 / 5.0

        # Layer B: regime-conditional scaling
        layerB = np.ones(5)
        # ZRisk more important when regime is strongly mean-reverting and stationary
        if regime_score >= 0.6 and (adf_pvalue is None or adf_pvalue < 0.05):
            layerB[0] *= 1.3
        # RegimeRisk more important when regime score is ambiguous
        if 0.3 <= regime_score <= 0.7:
            layerB[1] *= 1.3
        # FilterRisk slightly emphasized always
        layerB[2] *= 1.1
        # TimeRisk more important when regime is stable
        if regime_score >= 0.7:
            layerB[3] *= 1.2
        # HLJumpRisk more important when mean-reverting regime score is low
        if regime_score <= 0.4:
            layerB[4] *= 1.3

        layerB /= layerB.sum()

        est = layerA * layerB
        if est.sum() > 0:
            est /= est.sum()
        else:
            est = self.prior_weights.copy()

        # Layer C: stability dampening
        stability_factor = min(1.0, n / float(self.window))
        blend = self.dampening * stability_factor

        self.weights = (1.0 - blend) * self.prior_weights + blend * est
        self.weights /= self.weights.sum()
        return self.weights


# --------------------------------------------------------------------------
# Layer 2: Target Position (Mode B with full RiskScore)
# --------------------------------------------------------------------------

def layer2_target_position(
    signal_score,
    kalman_result,
    ou_params,
    mode="B",
    risk_pct_fixed=0.02,
    risk_pct_min=0.0075,
    risk_pct_max=0.02,
    p_max_factor=3.0,
    hl_entry=None,
    trade_age=None,
    weights=None,
):
    """
    Layer 2: SignalScore -> TargetPosition (fraction of notional).

    Parameters
    ----------
    signal_score : np.ndarray
        Layer 1 SignalScore, one-dimensional.
    kalman_result : dict
        Must contain 'z_score', 'p1', 'P', and 'GLD'.
    ou_params : dict
        Must contain 'kappa', 'half_life_min'.
    mode : {"A","B"}
        A: fixed risk, B: z-scaled risk.
    weights : np.ndarray or None
        5-element risk weights; if None, uses prior weights.

    Returns
    -------
    target_position : np.ndarray
    risk_score      : np.ndarray
    factors         : np.ndarray, shape (N, 5)
                      [ZRisk, RegimeRisk, FilterRisk, TimeRisk, HLJumpRisk]
    """
    z = np.asarray(kalman_result["z_score"], dtype=float)
    p1 = np.asarray(kalman_result["p1"], dtype=float)
    P = np.asarray(kalman_result["P"], dtype=float)
    signal_score = np.asarray(signal_score, dtype=float)

    N = len(z)

    # 1. SizeScalar
    if mode == "A":
        gld_level = np.mean(kalman_result["GLD"]) if "GLD" in kalman_result else 470.0
        dollar_vol = 0.002 * gld_level
        size_scalar_val = risk_pct_fixed / max(dollar_vol, 1e-6)
        size_scalar = np.full(N, size_scalar_val, dtype=float)
    else:
        # Mode B: risk % scales with |z|
        z_abs = np.abs(z)
        z_min, z_max = 1.5, 3.5
        z_scaled = np.clip((z_abs - z_min) / (z_max - z_min), 0.0, 1.0)
        risk_pct = risk_pct_min + z_scaled * (risk_pct_max - risk_pct_min)

        gld_level = np.mean(kalman_result["GLD"]) if "GLD" in kalman_result else 470.0
        dollar_vol = 0.002 * gld_level
        size_scalar = risk_pct / max(dollar_vol, 1e-6)

    # 2. RiskScore components
    # ZRisk: 0 when |z| <=2, 1 when |z| >=3.5
    z_risk = np.clip((np.abs(z) - 2.0) / (3.5 - 2.0), 0.0, 1.0)

    # RegimeRisk: 0 when p1>=0.7, 1 when p1<=0
    regime_risk = np.clip((0.7 - p1) / 0.7, 0.0, 1.0)

    # FilterRisk: 0 at median(P), 1 at P_max = p_max_factor*median(P)
    P_med = np.median(P)
    P_max = p_max_factor * P_med
    filter_risk = np.clip((P - P_med) / max(P_max - P_med, 1e-12), 0.0, 1.0)

    # TimeRisk and HLJumpRisk need trade-level HL_entry and age
    if hl_entry is None:
        hl_entry_arr = np.full(N, ou_params["half_life_min"], dtype=float)
    else:
        hl_entry_arr = np.asarray(hl_entry, dtype=float)
        hl_entry_arr[hl_entry_arr <= 0] = ou_params["half_life_min"]

    if trade_age is None:
        age = np.zeros(N, dtype=float)
    else:
        age = np.asarray(trade_age, dtype=float)
        age[age < 0] = 0.0

    time_risk = np.clip((age / (2.0 * hl_entry_arr)) ** 2, 0.0, 1.0)

    if "half_life_bars" in kalman_result:
        hl_current = np.asarray(kalman_result["half_life_bars"], dtype=float)
        hl_current[hl_current <= 0] = hl_entry_arr
    else:
        hl_current = hl_entry_arr.copy()

    hl_ratio = hl_current / np.maximum(hl_entry_arr, 1e-12)
    hl_jump_risk = np.clip((hl_ratio - 1.0) / (3.0 - 1.0), 0.0, 1.0)

    # Factors matrix
    factors = np.column_stack([z_risk, regime_risk, filter_risk, time_risk, hl_jump_risk])

    # Weights
    if weights is None:
        weights = np.array([0.35, 0.15, 0.20, 0.20, 0.10], dtype=float)

    risk_score = np.clip(factors @ weights, 0.0, 1.0)

    # 3. TimeDecay: linear to 0 at age = 3*HL_entry
    time_decay = np.clip(1.0 - age / (3.0 * hl_entry_arr), 0.0, 1.0)

    # 4. TargetPosition
    position_scalar = 1.0 - np.sqrt(risk_score)
    target_position = signal_score * size_scalar * position_scalar * time_decay

    return target_position, risk_score, factors


# --------------------------------------------------------------------------
# Layer 3: Execution (single instrument, simple Δ-trading)
# --------------------------------------------------------------------------

def layer3_execution(target_position, current_position, min_delta=0.005):
    """
    Translate TargetPosition into orders with an anti-whipsaw threshold.

    Parameters
    ----------
    target_position : np.ndarray
        Desired position per bar (fraction of notional).
    current_position : np.ndarray
        Actual position per bar.
    min_delta : float
        Minimum abs(delta) to execute.

    Returns
    -------
    orders      : np.ndarray
    new_position: np.ndarray
    """
    target_position = np.asarray(target_position, dtype=float)
    current_position = np.asarray(current_position, dtype=float)

    current_pos_padded = np.zeros_like(target_position)
    if len(current_position) > 0:
        current_pos_padded[1:] = current_position[:-1]

    delta = target_position - current_pos_padded

    execute_mask = np.abs(delta) > min_delta
    orders = np.zeros_like(delta)
    orders[execute_mask] = delta[execute_mask]

    new_position = current_pos_padded + orders
    return orders, new_position


# --------------------------------------------------------------------------
# PnL and Walk-forward (Phase 5)
# --------------------------------------------------------------------------
def proper_spread_pnl(session_data, target_position, beta):
    """
    GLD - beta * IAU PnL in bps, using lagged TargetPosition.
    Signal at t earns return from t to t+1.
    """
    gld = np.asarray(session_data["GLD"], dtype=float)
    iau = np.asarray(session_data["IAU"], dtype=float)

    gld_ret = np.diff(np.log(gld))          # length N-1, return t-1 -> t
    iau_ret = np.diff(np.log(iau))
    spread_ret = gld_ret - beta * iau_ret

    tp = np.asarray(target_position, dtype=float)

    # Use position known after bar t-1 to earn return t-1 -> t
    pnl = tp[:-1] * spread_ret * 10_000.0
    return pnl


def phase5_walkforward_full(results, sessions, risk_pct=0.02):
    """
    Phase 5: Full walk-forward using Phase 4 with adaptive weights.
    """
    print(f"phase5_walkforward_full: got {len(results)} results, {len(sessions)} sessions")

    estimator = AdaptiveWeightEstimator(window=60, stability_min_obs=30, dampening=0.70)

    all_pnl = []
    session_sharpes = []

    for i, result in enumerate(results):
        session = sessions[i]
        result["GLD"] = session["GLD"].values
        result["IAU"] = session["IAU"].values

        # OU params from spread/Kalman residuals
        spread = np.asarray(result["spread"], dtype=float)
        x_hat = np.asarray(result["x_hat"], dtype=float)
        ou_params = ou_params_final(np.std(spread - x_hat))

        # Layer 1
        signal_scores, _ = layer1_complete(result, ou_params, p_max_factor=3.0, gate_open=True)

        # Trade state
        hl_current = np.full_like(signal_scores, ou_params["half_life_min"], dtype=float)
        hl_entry, trade_age, position_state = build_entry_state(
            signal_scores, hl_current, entry_threshold=0.30
        )

        z = np.asarray(result["z_score"], dtype=float)
        prices = spread

        N = len(z)
        target_pos = np.zeros(N, dtype=float)
        risk_scores = np.zeros(N, dtype=float)

        for t in range(N):
            single_result = {
                "z_score": z[t:t+1],
                "p1": result["p1"][t:t+1],
                "P": result["P"][t:t+1],
                "GLD": result["GLD"][t:t+1],
            }

            tp_t, rs_t, factors_t = layer2_target_position(
                signal_score=signal_scores[t:t+1],
                kalman_result=single_result,
                ou_params=ou_params,
                mode="B",
                risk_pct_fixed=risk_pct,
                risk_pct_min=0.0075,
                risk_pct_max=0.02,
                p_max_factor=3.0,
                hl_entry=hl_entry[t:t+1],
                trade_age=trade_age[t:t+1],
                weights=estimator.weights,
            )

            target_pos[t] = tp_t[0]
            risk_scores[t] = rs_t[0]

            if t > 0:
                outcome = abs(prices[t] - prices[t-1])
                estimator.record_outcome(factors_t[0], outcome)
                _ = estimator.update_weights(regime_score=float(result["p1"][t]))

        beta = float(result["beta"])

        gld = np.asarray(result["GLD"], dtype=float)
        iau = np.asarray(result["IAU"], dtype=float)
        gld_ret = np.diff(np.log(gld))
        iau_ret = np.diff(np.log(iau))
        spread_ret = gld_ret - beta * iau_ret
        corr = np.corrcoef(target_pos[:-1], spread_ret)[0, 1]
        print(f"Session {i} corr(target_pos[:-1], spread_ret) = {corr:.4f}")

        pnl = proper_spread_pnl(result, target_pos, beta)

        all_pnl.append(pnl)
        sh = np.mean(pnl) / np.std(pnl) * np.sqrt(252 * 78 / 390) if np.std(pnl) > 0 else 0.0
        session_sharpes.append(sh)
        print(f"Session {i}: PnL={pnl.sum():+.0f} bps, Sharpe={sh:.2f}")

        # Optional: debug plots for first session only
        if i == 0:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            axes[0].plot(spread, label="Spread", color="gray")
            axes[0].plot(x_hat, label="x_hat", color="blue")
            axes[0].fill_between(
                np.arange(len(x_hat)),
                x_hat - 2*np.sqrt(result["P"]),
                x_hat + 2*np.sqrt(result["P"]),
                alpha=0.2, color="blue", label="±2√P",
            )
            axes[0].legend(); axes[0].set_title("Spread & Kalman estimate")

            axes[1].plot(result["z_score"], label="z", color="purple")
            axes[1].plot(signal_scores, label="SignalScore", color="green")
            axes[1].plot(target_pos, label="TargetPos", color="orange")
            axes[1].legend(); axes[1].set_title("Signals & Target Position")

            cum_pnl_0 = np.cumsum(np.concatenate([[0], pnl]))
            axes[2].plot(cum_pnl_0, label="Cum PnL (bps)")
            axes[2].legend(); axes[2].set_title("Cumulative PnL (session 0)")
            plt.tight_layout()
            plt.show()

    # ---- aggregation AFTER the loop ----
    if not all_pnl:
        print("phase5_walkforward_full: no PnL arrays, nothing to aggregate.")
        return np.array([]), []

    all_pnl_flat = np.concatenate(all_pnl)

    if len(all_pnl_flat) > 0 and np.std(all_pnl_flat) > 0:
        total_sharpe = np.mean(all_pnl_flat) / np.std(all_pnl_flat) * np.sqrt(252 * 390)
    else:
        total_sharpe = 0.0

    print("\n=== PHASE 5 COMPLETE ===")
    print(f"Sessions: {len(results)}")
    print(f"Total PnL: {all_pnl_flat.sum():+.0f} bps")
    print(f"Total Sharpe: {total_sharpe:.2f}")
    print(f"Avg session Sharpe: {np.mean(session_sharpes):.2f}")
    win_rate = np.mean(all_pnl_flat > 0) if len(all_pnl_flat) > 0 else 0.0
    print(f"Win rate: {win_rate:.1%}")
    print(f"PnL bars: {len(all_pnl_flat)}, mean={np.mean(all_pnl_flat):.4f}, std={np.std(all_pnl_flat):.4f}")

    if len(all_pnl_flat) > 0:
        plt.figure(figsize=(12, 6))
        cum_pnl = np.cumsum(np.concatenate([[0], all_pnl_flat]))
        plt.plot(cum_pnl, linewidth=2)
        plt.title("Phase 5: Walk-Forward Cumulative PnL (all sessions)")
        plt.ylabel("Cumulative PnL (bps)")
        plt.xlabel("1-min Bars")
        plt.grid(True, alpha=0.3)
        plt.show()

    return all_pnl_flat, session_sharpes



# --------------------------------------------------------------------------
# Main harness (use only when running this file directly)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    combined = load_gld_iau_intraday_chunked(days_back=60, chunk_days=7, interval="1m")
    print(f"Combined shape: {combined.shape}")

    sessions = split_sessions(combined)
    print(f"split_sessions produced {len(sessions)} sessions")
    print("First session length:", len(sessions[0]))
    print("All session lengths:", [len(s) for s in sessions])

    results = run_pipeline(sessions, combined)
    print(f"run_pipeline produced {len(results)} results")

    pnl_series, sharpes = phase5_walkforward_full(results, sessions, risk_pct=0.02)

