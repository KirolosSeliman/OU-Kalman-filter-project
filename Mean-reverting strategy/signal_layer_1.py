"""
signal_layer_1.py
─────────────────────────────────────────────────────────────────────────────
Layer 1: SignalScore = RawSignal × Gate × Filter × Regime × Cost
Layer 2: TargetPosition = SignalScore × SizeScalar × PositionScalar × TimeDecay
Layer 3: Orders with min_delta threshold
PnL:     proper_spread_pnl in bps
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Raw signal
# ─────────────────────────────────────────────────────────────────────────────

def raw_signal(z_score, z_scale=2.0):
    """
    Raw directional signal: +tanh(z / z_scale).
    Positive → spread below Kalman mean → BUY.
    """
    z_score = np.asarray(z_score, dtype=float)
    return +np.tanh(z_score / z_scale)


def compute_cost_factor(z_score, ou_params, tc_bps=5):
    """
    CostFactor = max(0, 1 - tc / expected_edge)
    expected_edge ∝ |z| * kappa * half_life_min
    """
    z         = np.asarray(z_score, dtype=float)
    kappa     = float(ou_params["kappa"])
    hl_min    = float(ou_params["half_life_min"])
    tc        = tc_bps / 10_000.0
    edge      = np.abs(z) * kappa * hl_min
    return np.maximum(0.0, 1.0 - tc / np.maximum(edge, 1e-6))


def layer1_validity_factors(kalman_result, ou_params, p_max_factor=3.0, gate_open=True):
    """
    Returns gate_factor, filter_factor, regime_factor, cost_factor as np.ndarray.
    """
    P  = np.asarray(kalman_result["P"],       dtype=float)
    p1 = np.asarray(kalman_result["p1"],      dtype=float)
    z  = np.asarray(kalman_result["z_score"], dtype=float)

    gate_factor   = np.ones_like(P) if gate_open else np.zeros_like(P)

    P_med         = np.median(P)
    P_max         = p_max_factor * P_med
    filter_factor = np.maximum(0.0, 1.0 - P / np.maximum(P_max, 1e-12))

    regime_factor = np.clip(p1, 0.0, 1.0)
    cost_factor   = compute_cost_factor(z, ou_params)

    return gate_factor, filter_factor, regime_factor, cost_factor


def layer1_complete(kalman_result, ou_params, p_max_factor=3.0, gate_open=True):
    """
    Full Layer 1 SignalScore = RawSignal × Gate × Filter × Regime × Cost.
    Returns (signal_score, (gate, filter, regime, cost)).
    """
    z    = np.asarray(kalman_result["z_score"], dtype=float)
    raw  = raw_signal(z, z_scale=2.0)
    gate, filt, regime, cost = layer1_validity_factors(
        kalman_result, ou_params,
        p_max_factor=p_max_factor,
        gate_open=gate_open,
    )
    signal_score = raw * gate * filt * regime * cost
    return signal_score, (gate, filt, regime, cost)


# ─────────────────────────────────────────────────────────────────────────────
# Trade-level state
# ─────────────────────────────────────────────────────────────────────────────

def build_entry_state(signal_scores, hl_series, entry_threshold=0.30):
    """
    Build hl_entry, trade_age, position_state arrays from SignalScore.

    Entry:  |signal| >= entry_threshold while flat
    Exit:   signal changes sign (direction conflict)
    """
    sig      = np.asarray(signal_scores, dtype=float)
    hl       = np.asarray(hl_series,     dtype=float)
    N        = len(sig)
    hl       = hl.copy()
    hl[hl <= 0] = np.nan

    hl_entry       = np.zeros(N, dtype=float)
    trade_age      = np.zeros(N, dtype=float)
    position_state = np.zeros(N, dtype=int)

    in_trade          = False
    current_dir       = 0
    current_hl_entry  = 0.0
    current_age       = 0

    hl_fallback = float(np.nanmedian(hl))
    if not np.isfinite(hl_fallback):
        hl_fallback = 20.0

    for t in range(N):
        s = sig[t]

        if not in_trade:
            if np.abs(s) >= entry_threshold:
                in_trade         = True
                current_dir      = 1 if s > 0 else -1
                hl_t             = hl[t] if np.isfinite(hl[t]) else hl_fallback
                current_hl_entry = max(hl_t, 1.0)
                current_age      = 0
        else:
            if s * current_dir <= 0.0:
                in_trade         = False
                current_dir      = 0
                current_hl_entry = 0.0
                current_age      = 0

        if in_trade:
            hl_entry[t]       = current_hl_entry
            trade_age[t]      = current_age
            position_state[t] = current_dir
            current_age      += 1
        else:
            hl_entry[t]       = 0.0
            trade_age[t]      = 0.0
            position_state[t] = 0

    return hl_entry, trade_age, position_state


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Weight Estimator
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveWeightEstimator:
    """
    Adaptive weights for 5 risk factors:
    [ZRisk, RegimeRisk, FilterRisk, TimeRisk, HLJumpRisk]

    Layer A: Spearman |rho| with adverse outcome
    Layer B: regime-conditional scaling
    Layer C: stability-based blend with prior weights
    """

    def __init__(self, window=60, stability_min_obs=30, dampening=0.70):
        self.window            = window
        self.stability_min_obs = stability_min_obs
        self.dampening         = dampening
        self.prior_weights     = np.array([0.35, 0.15, 0.20, 0.20, 0.10], dtype=float)
        self.factor_history    = []
        self.outcome_history   = []
        self.weights           = self.prior_weights.copy()

    def record_outcome(self, factors_t, outcome_t):
        f = np.asarray(factors_t, dtype=float).ravel()
        if f.shape[0] != 5:
            raise ValueError("factors_t must have length 5")
        self.factor_history.append(f)
        self.outcome_history.append(float(outcome_t))
        if len(self.factor_history) > self.window:
            self.factor_history  = self.factor_history[-self.window:]
            self.outcome_history = self.outcome_history[-self.window:]

    def update_weights(self, regime_score, adf_pvalue=None):
        n = len(self.factor_history)
        if n < self.stability_min_obs:
            self.weights = self.prior_weights.copy()
            return self.weights

        X = np.vstack(self.factor_history)
        y = np.asarray(self.outcome_history)

        # Layer A: Spearman |rho|
        layerA = np.zeros(5)
        for i in range(5):
            rho, _ = stats.spearmanr(X[:, i], y)
            layerA[i] = abs(rho) if np.isfinite(rho) else 0.0
        if layerA.sum() > 0:
            layerA /= layerA.sum()
        else:
            layerA[:] = 0.2

        # Layer B: regime-conditional scaling
        layerB = np.ones(5)
        if regime_score >= 0.6 and (adf_pvalue is None or adf_pvalue < 0.05):
            layerB[0] *= 1.3
        if 0.3 <= regime_score <= 0.7:
            layerB[1] *= 1.3
        layerB[2] *= 1.1
        if regime_score >= 0.7:
            layerB[3] *= 1.2
        if regime_score <= 0.4:
            layerB[4] *= 1.3
        layerB /= layerB.sum()

        est = layerA * layerB
        est = est / est.sum() if est.sum() > 0 else self.prior_weights.copy()

        # Layer C: stability dampening
        blend        = self.dampening * min(1.0, n / float(self.window))
        self.weights = (1.0 - blend) * self.prior_weights + blend * est
        self.weights /= self.weights.sum()
        return self.weights


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Target Position
# ─────────────────────────────────────────────────────────────────────────────

def layer2_target_position(
    signal_score,
    kalman_result,
    ou_params,
    mode           = "B",
    risk_pct_fixed = 0.02,
    risk_pct_min   = 0.0075,
    risk_pct_max   = 0.02,
    p_max_factor   = 3.0,
    hl_entry       = None,
    trade_age      = None,
    weights        = None,
):
    """
    Returns target_position, risk_score, factors (shape N×5).
    """
    z            = np.asarray(kalman_result["z_score"], dtype=float)
    p1           = np.asarray(kalman_result["p1"],      dtype=float)
    P            = np.asarray(kalman_result["P"],       dtype=float)
    signal_score = np.asarray(signal_score,             dtype=float)
    N            = len(z)

    # ── SizeScalar ────────────────────────────────────────────────────────
    gld_level  = float(np.mean(kalman_result["GLD"])) if "GLD" in kalman_result else 470.0
    dollar_vol = 0.002 * gld_level

    if mode == "A":
        size_scalar = np.full(N, risk_pct_fixed / max(dollar_vol, 1e-6), dtype=float)
    else:
        z_abs       = np.abs(z)
        z_scaled    = np.clip((z_abs - 1.5) / (3.5 - 1.5), 0.0, 1.0)
        risk_pct    = risk_pct_min + z_scaled * (risk_pct_max - risk_pct_min)
        size_scalar = risk_pct / max(dollar_vol, 1e-6)

    # ── Risk factors ──────────────────────────────────────────────────────
    z_risk      = np.clip((np.abs(z) - 2.0) / (3.5 - 2.0), 0.0, 1.0)
    regime_risk = np.clip((0.7 - p1) / 0.7, 0.0, 1.0)

    P_med       = np.median(P)
    P_max       = p_max_factor * P_med
    filter_risk = np.clip((P - P_med) / max(P_max - P_med, 1e-12), 0.0, 1.0)

    hl_min_val       = float(ou_params.get("half_life_min", 20.0))
    hl_entry_arr     = np.asarray(hl_entry, dtype=float) if hl_entry is not None else np.full(N, hl_min_val)
    hl_entry_arr     = np.where(hl_entry_arr > 0, hl_entry_arr, hl_min_val)

    age              = np.asarray(trade_age, dtype=float) if trade_age is not None else np.zeros(N)
    age              = np.maximum(age, 0.0)

    time_risk        = np.clip((age / (2.0 * hl_entry_arr)) ** 2, 0.0, 1.0)

    hl_current       = np.asarray(kalman_result.get("half_life_bars", hl_entry_arr), dtype=float)
    hl_current       = np.where(hl_current > 0, hl_current, hl_entry_arr)
    hl_ratio         = hl_current / np.maximum(hl_entry_arr, 1e-12)
    hl_jump_risk     = np.clip((hl_ratio - 1.0) / 2.0, 0.0, 1.0)

    factors          = np.column_stack([z_risk, regime_risk, filter_risk, time_risk, hl_jump_risk])

    if weights is None:
        weights = np.array([0.35, 0.15, 0.20, 0.20, 0.10], dtype=float)

    risk_score       = np.clip(factors @ weights, 0.0, 1.0)
    time_decay       = np.clip(1.0 - age / (3.0 * hl_entry_arr), 0.0, 1.0)
    position_scalar  = 1.0 - np.sqrt(risk_score)
    target_position  = signal_score * size_scalar * position_scalar * time_decay

    return target_position, risk_score, factors


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Execution
# ─────────────────────────────────────────────────────────────────────────────

def layer3_execution(target_position, current_position, min_delta=0.005):
    """Orders with anti-whipsaw min_delta threshold."""
    target_position  = np.asarray(target_position,  dtype=float)
    current_position = np.asarray(current_position, dtype=float)

    prev_pos         = np.zeros_like(target_position)
    if len(current_position) > 0:
        prev_pos[1:] = current_position[:-1]

    delta        = target_position - prev_pos
    mask         = np.abs(delta) > min_delta
    orders       = np.where(mask, delta, 0.0)
    new_position = prev_pos + orders
    return orders, new_position


# ─────────────────────────────────────────────────────────────────────────────
# PnL
# ─────────────────────────────────────────────────────────────────────────────

def proper_spread_pnl(session_data, target_position, beta):
    """
    PnL in bps. Signal at bar t earns return from t to t+1.
    Returns array of length N-1.
    """
    gld        = np.asarray(session_data["GLD"], dtype=float)
    iau        = np.asarray(session_data["IAU"], dtype=float)
    gld_ret    = np.diff(np.log(gld))
    iau_ret    = np.diff(np.log(iau))
    spread_ret = gld_ret - beta * iau_ret
    tp         = np.asarray(target_position, dtype=float)
    return tp[:-1] * spread_ret * 10_000.0
