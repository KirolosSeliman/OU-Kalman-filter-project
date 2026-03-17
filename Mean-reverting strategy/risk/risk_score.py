"""
risk/risk_score.py
─────────────────────────────────────────────────────────────────────────────
Layer 2 Component: Composite Risk Score and Adaptive Weight Estimator.

Produces RiskScore ∈ [0, 1] every bar from five factors:

    RiskScore = w1*ZRisk + w2*RegimeRisk + w3*FilterRisk
              + w4*TimeRisk + w5*HLJumpRisk

where weights w1..w5 are produced by the Adaptive Weight Estimator.

Five risk factors:
    ZRisk       = clip((|z| - z_onset) / (z_full - z_onset), 0, 1)
                  0 when |z| <= z_onset, 1 when |z| >= z_full
    RegimeRisk  = clip((regime_onset - IMM_score) / regime_onset, 0, 1)
                  0 when IMM healthy, 1 when IMM near 0
    FilterRisk  = clip(P_k / P_max, 0, 1)
                  0 when filter converged, 1 at P_max
    TimeRisk    = clip((trade_age / (mult * HL_entry))^2, 0, 1)
                  quadratic, 1 at time_risk_hl_multiplier * HL_entry bars
    HLJumpRisk  = clip((HL_current - HL_entry) / (jump_mult * HL_entry), 0, 1)
                  0 when HL unchanged, 1 when HL doubles

Adaptive Weight Estimator (3 layers):
    Layer A: Spearman rank correlation of each factor vs next-bar |price move|
             over a rolling window. Most predictive factor → highest weight.
    Layer B: Regime-conditional multipliers based on current IMM probability,
             ADF p-value, and HL stability.
    Layer C: Stability dampening — blend toward prior weights when evidence
             is weak (low observation count, unstable regime, unstable OU).

RiskScore usage in TargetPosition:
    TargetPosition = SignalScore * SizeScalar * (1 - sqrt(RiskScore)) * TimeDecay
    RiskScore = 0.00 → scalar = 1.00 (full size)
    RiskScore = 0.25 → scalar = 0.50 (half)
    RiskScore = 0.64 → scalar = 0.20 (80% reduction)
    RiskScore = 1.00 → scalar = 0.00 (fully flat)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclass — per bar, assembled by pipeline_runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskInput:
    """All inputs needed to compute RiskScore for one bar."""
    # From signal layer
    z_score:      float   # current z-score
    imm_score:    float   # IMM p1 (mean-reverting model probability)
    P_k:          float   # Kalman error covariance
    adf_pvalue:   float   # current ADF p-value (for Layer B)

    # From trade state (provided by PositionManager)
    in_trade:     bool    # True if a position is currently open
    trade_age:    int     # bars since position was opened (0 if not in trade)
    hl_entry:     float   # HL at the moment the trade was opened (LOCKED)
    hl_current:   float   # current HL estimate from StationarityOUEngine


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass — per bar
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskBar:
    """All outputs of RiskScoreEngine.compute() for one bar."""
    risk_score:    float          # ∈ [0, 1] composite risk
    position_scalar: float        # (1 - sqrt(risk_score)) ∈ [0, 1]

    # Individual factors
    z_risk:        float
    regime_risk:   float
    filter_risk:   float
    time_risk:     float
    hl_jump_risk:  float

    # Weights used
    weights:       np.ndarray     # shape (5,), sum = 1.0

    # Diagnostics
    n_obs:         int            # observation count in rolling window
    weights_source: str           # "prior", "estimated", or "blended"


# ─────────────────────────────────────────────────────────────────────────────
# Observation record (for Layer A rolling window)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Observation:
    """One bar's factor values and the next bar's realized |price move|."""
    z_risk:       float
    regime_risk:  float
    filter_risk:  float
    time_risk:    float
    hl_jump_risk: float
    next_move:    float   # filled in by record_outcome() the following bar


# ─────────────────────────────────────────────────────────────────────────────
# RiskScoreEngine
# ─────────────────────────────────────────────────────────────────────────────

class RiskScoreEngine:
    """
    Computes Composite Risk Score and maintains Adaptive Weight Estimator.

    Correct usage pattern (PER BAR, in this order):
        1. risk_engine.record_outcome(abs(spread_k - spread_{k-1}))
              ← fills in next_move for the PREVIOUS bar's observation
        2. bar = risk_engine.compute(risk_input_k)
              ← computes factors and weights for the CURRENT bar

    This ordering ensures Layer A uses only past data for weight estimation.
    record_outcome() must be called BEFORE compute() on every bar.
    On the very first bar, call compute() without calling record_outcome().
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg      = cfg
        risk_cfg       = cfg.risk

        self._window   = risk_cfg.adaptive_window
        self._min_obs  = risk_cfg.stability_min_obs
        self._dampen   = risk_cfg.dampening

        # Rolling observation buffer (maxlen = adaptive_window)
        self._obs_buf: deque = deque(maxlen=self._window)

        # Pending observation (current bar's factors, awaiting next_move)
        self._pending: Optional[_Observation] = None

        # Prior weight vectors (regime-keyed)
        self._priors: Dict[str, np.ndarray] = {
            "mean_reverting": np.array(risk_cfg.prior_weights_mean_reverting),
            "transitional":   np.array(risk_cfg.prior_weights_transitional),
            "trending":       np.array(risk_cfg.prior_weights_trending),
        }

    # ─────────────────────────────────────────────────────────────────────
    # Outcome recorder (call at bar start, before compute)
    # ─────────────────────────────────────────────────────────────────────

    def record_outcome(self, abs_price_move: float) -> None:
        """
        Record the realized |spread move| for the previous bar's observation.
        Must be called before compute() on each bar (except the first).

        Parameters
        ----------
        abs_price_move : float
            |spread_k - spread_{k-1}|  — adverse move proxy.
        """
        if self._pending is not None:
            self._pending.next_move = float(abs_price_move)
            self._obs_buf.append(self._pending)
            self._pending = None

    # ─────────────────────────────────────────────────────────────────────
    # Main computation
    # ─────────────────────────────────────────────────────────────────────

    def compute(self, inp: RiskInput) -> RiskBar:
        """
        Compute RiskScore for current bar.

        Parameters
        ----------
        inp : RiskInput

        Returns
        -------
        RiskBar
        """
        risk_cfg = self._cfg.risk

        # ── Step 1: Compute five risk factors ─────────────────────────────
        z_risk       = self._compute_z_risk(inp)
        regime_risk  = self._compute_regime_risk(inp)
        filter_risk  = self._compute_filter_risk(inp)
        time_risk    = self._compute_time_risk(inp)
        hl_jump_risk = self._compute_hl_jump_risk(inp)

        factors = np.array([z_risk, regime_risk, filter_risk,
                             time_risk, hl_jump_risk], dtype=float)

        # ── Step 2: Adaptive Weight Estimator ─────────────────────────────
        weights, n_obs, source = self._compute_weights(inp)

        # ── Step 3: Composite RiskScore ───────────────────────────────────
        risk_score = float(np.dot(weights, factors))
        risk_score = float(np.clip(risk_score, 0.0, 1.0))

        # ── Step 4: Position scalar ───────────────────────────────────────
        # (1 - sqrt(RiskScore)) — nonlinear, aggressive reduction at
        # moderate risk. sqrt ensures factor is never > 1.
        position_scalar = float(1.0 - np.sqrt(risk_score))
        position_scalar = float(np.clip(position_scalar, 0.0, 1.0))

        # ── Step 5: Store pending observation for next bar ────────────────
        self._pending = _Observation(
            z_risk       = z_risk,
            regime_risk  = regime_risk,
            filter_risk  = filter_risk,
            time_risk    = time_risk,
            hl_jump_risk = hl_jump_risk,
            next_move    = float("nan"),  # filled in by next record_outcome()
        )

        return RiskBar(
            risk_score      = risk_score,
            position_scalar = position_scalar,
            z_risk          = z_risk,
            regime_risk     = regime_risk,
            filter_risk     = filter_risk,
            time_risk       = time_risk,
            hl_jump_risk    = hl_jump_risk,
            weights         = weights,
            n_obs           = n_obs,
            weights_source  = source,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Five risk factor computations
    # ─────────────────────────────────────────────────────────────────────

    def _compute_z_risk(self, inp: RiskInput) -> float:
        """
        ZRisk: 0 when |z| <= z_onset, ramps to 1 when |z| >= z_full.
        Non-zero only when in a trade — outside a trade z_risk is irrelevant
        to position management (there is no position to manage).
        """
        if not inp.in_trade:
            return 0.0
        cfg = self._cfg.risk
        onset = cfg.z_risk_onset
        full  = cfg.z_risk_full
        ramp  = full - onset
        if ramp <= 0:
            return 0.0
        return float(np.clip((abs(inp.z_score) - onset) / ramp, 0.0, 1.0))

    def _compute_regime_risk(self, inp: RiskInput) -> float:
        """
        RegimeRisk: 0 when IMM healthy (score >= onset), 1 when score = 0.
        Active regardless of trade state — regime deterioration is always
        a risk signal even before entry.
        """
        cfg   = self._cfg.risk
        onset = cfg.regime_risk_onset
        if onset <= 0:
            return 0.0
        return float(np.clip((onset - inp.imm_score) / onset, 0.0, 1.0))

    def _compute_filter_risk(self, inp: RiskInput) -> float:
        """
        FilterRisk: 0 when P_k = 0 (fully converged), 1 at P_max.
        Proxy for Kalman filter uncertainty.
        """
        P_max = self._cfg.signal.P_max
        if P_max <= 0:
            return 0.0
        return float(np.clip(inp.P_k / P_max, 0.0, 1.0))

    def _compute_time_risk(self, inp: RiskInput) -> float:
        """
        TimeRisk: quadratic ramp from 0 at entry to 1 at
        (time_risk_hl_multiplier * HL_entry) bars.

        Uses HL_ENTRY — locked at entry, never updated.
        This prevents HL jumps from extending the time clock.
        Only non-zero when in_trade=True.
        """
        if not inp.in_trade:
            return 0.0
        cfg  = self._cfg.risk
        mult = cfg.time_risk_hl_multiplier
        if not np.isfinite(inp.hl_entry) or inp.hl_entry <= 0:
            return 0.0
        horizon = mult * inp.hl_entry
        if horizon <= 0:
            return 0.0
        return float(np.clip((inp.trade_age / horizon) ** 2, 0.0, 1.0))

    def _compute_hl_jump_risk(self, inp: RiskInput) -> float:
        """
        HLJumpRisk: 0 when HL unchanged from entry, ramps to 1 when
        HL extends by (hl_jump_risk_multiplier * HL_entry).

        Only non-zero when in_trade=True and HL_current > HL_entry.
        HL compression (faster reversion) is not a risk.
        """
        if not inp.in_trade:
            return 0.0
        if not np.isfinite(inp.hl_entry)   or inp.hl_entry   <= 0:
            return 0.0
        if not np.isfinite(inp.hl_current) or inp.hl_current <= 0:
            return 0.0
        cfg      = self._cfg.risk
        denom    = cfg.hl_jump_risk_multiplier * inp.hl_entry
        if denom <= 0:
            return 0.0
        hl_delta = max(0.0, inp.hl_current - inp.hl_entry)
        return float(np.clip(hl_delta / denom, 0.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────
    # Adaptive Weight Estimator (3 layers)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_weights(
        self, inp: RiskInput
    ) -> tuple[np.ndarray, int, str]:
        """
        Returns (weights, n_obs, source).
        source ∈ {"prior", "estimated", "blended"}
        """
        regime     = self._current_regime(inp)
        prior      = self._priors[regime].copy()
        n_obs      = len(self._obs_buf)

        # Not enough observations — use prior only
        if n_obs < self._min_obs:
            return prior.copy(), n_obs, "prior"

        # ── Layer A: Spearman rank correlations ───────────────────────────
        layer_a = self._layer_a_weights()

        # ── Layer B: Regime-conditional multipliers ───────────────────────
        layer_b = self._layer_b_multipliers(inp, regime)

        # ── Combine A × B ─────────────────────────────────────────────────
        raw      = layer_a * layer_b
        raw_sum  = raw.sum()
        if raw_sum < 1e-12:
            estimated = prior.copy()
        else:
            estimated = raw / raw_sum

        # ── Layer C: Stability dampening ──────────────────────────────────
        stability = self._stability_factor(inp, n_obs)
        blend     = self._dampen * stability
        weights   = (1.0 - blend) * prior + blend * estimated
        w_sum     = weights.sum()
        if w_sum < 1e-12:
            weights = prior.copy()
        else:
            weights /= w_sum

        source = "blended" if blend > 1e-6 else "prior"
        return weights, n_obs, source

    def _layer_a_weights(self) -> np.ndarray:
        """
        Spearman rank correlations between each factor and next-bar |move|.
        Returns non-negative weight vector summing to 1.
        Falls back to uniform if correlations are all non-positive.
        """
        obs_list = list(self._obs_buf)
        valid    = [o for o in obs_list
                    if np.isfinite(o.next_move) and o.next_move >= 0]

        if len(valid) < 5:
            return np.ones(5) / 5.0

        factor_matrix = np.array([
            [o.z_risk, o.regime_risk, o.filter_risk,
             o.time_risk, o.hl_jump_risk]
            for o in valid
        ])
        outcomes = np.array([o.next_move for o in valid])

        corrs = np.zeros(5)
        for j in range(5):
            col = factor_matrix[:, j]
            if np.std(col) < 1e-12:
                corrs[j] = 0.0
            else:
                try:
                    r, _ = spearmanr(col, outcomes)
                    corrs[j] = max(0.0, float(r))
                except Exception:
                    corrs[j] = 0.0

        total = corrs.sum()
        if total < 1e-12:
            return np.ones(5) / 5.0
        return corrs / total

    def _layer_b_multipliers(
        self, inp: RiskInput, regime: str
    ) -> np.ndarray:
        """
        Regime-conditional scaling vector (5,).
        Amplifies factors that are theoretically important in current regime.
        Dampens factors that are theoretically inappropriate.
        Returns a non-negative vector (not normalized).
        """
        s = self._cfg.risk.regime_sensitivity
        mults = np.ones(5)

        if regime == "mean_reverting":
            # ZRisk is most important: spread displacement is the signal
            mults[0] *= (1.0 + 0.5 * s)   # ZRisk ↑
            mults[2] *= (1.0 + 0.3 * s)   # FilterRisk ↑ (convergence matters)
            mults[3] *= (1.0 + 0.2 * s)   # TimeRisk ↑  (HL is reliable)
            mults[1] *= (1.0 - 0.2 * s)   # RegimeRisk ↓ (regime is healthy)

        elif regime == "transitional":
            # Regime is uncertain — RegimeRisk and FilterRisk dominate
            mults[1] *= (1.0 + 0.5 * s)   # RegimeRisk ↑
            mults[2] *= (1.0 + 0.4 * s)   # FilterRisk ↑
            mults[0] *= (1.0 - 0.2 * s)   # ZRisk ↓
            mults[3] *= (1.0 - 0.2 * s)   # TimeRisk ↓ (HL unreliable)

        elif regime == "trending":
            # Trend in place — RegimeRisk and HLJumpRisk are most informative
            mults[1] *= (1.0 + 0.5 * s)   # RegimeRisk ↑
            mults[4] *= (1.0 + 0.4 * s)   # HLJumpRisk ↑
            mults[0] *= (1.0 - 0.2 * s)   # ZRisk ↓ (z may be large for wrong reasons)
            mults[3] *= (1.0 - 0.2 * s)   # TimeRisk ↓

        return np.clip(mults, 0.0, None)

    def _stability_factor(self, inp: RiskInput, n_obs: int) -> float:
        """
        Stability factor ∈ [0, 1].
        1.0 = fully trust estimated weights.
        0.0 = fully use prior.

        Combines:
          - Observation count (more obs → more trust)
          - IMM certainty (near 0.5 → transitional → less trust)
          - HL stability proxy (missing hl → less trust)
        """
        min_obs = self._min_obs
        window  = self._window

        # Count stability ∈ [0, 1]
        count_stability = float(np.clip(
            (n_obs - min_obs) / max(window - min_obs, 1),
            0.0, 1.0
        ))

        # Regime certainty: distance from 0.5 (maximum uncertainty)
        regime_certainty = float(2.0 * abs(inp.imm_score - 0.5))

        # HL stability: penalize if hl_current is missing or extreme
        if (inp.in_trade
                and np.isfinite(inp.hl_entry)
                and np.isfinite(inp.hl_current)
                and inp.hl_entry > 0):
            hl_ratio = inp.hl_current / inp.hl_entry
            # Stable: ratio near 1.0. Unstable: ratio >> 1 or << 0.5
            hl_stability = float(np.clip(
                1.0 - abs(hl_ratio - 1.0), 0.0, 1.0
            ))
        else:
            hl_stability = 0.5  # neutral when not in trade

        stability = (count_stability * 0.5
                     + regime_certainty * 0.3
                     + hl_stability     * 0.2)
        return float(np.clip(stability, 0.0, 1.0))

    def _current_regime(self, inp: RiskInput) -> str:
        """
        Classify current regime from IMM score.
        mean_reverting:  imm_score >= 0.60
        transitional:    0.30 <= imm_score < 0.60
        trending:        imm_score < 0.30
        """
        if inp.imm_score >= 0.60:
            return "mean_reverting"
        elif inp.imm_score >= 0.30:
            return "transitional"
        else:
            return "trending"

    # ─────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        """Serializable snapshot for StateManager."""
        obs_list = []
        for o in self._obs_buf:
            obs_list.append({
                "z_risk":      o.z_risk,
                "regime_risk": o.regime_risk,
                "filter_risk": o.filter_risk,
                "time_risk":   o.time_risk,
                "hl_jump_risk":o.hl_jump_risk,
                "next_move":   o.next_move,
            })
        pending = None
        if self._pending is not None:
            pending = {
                "z_risk":      self._pending.z_risk,
                "regime_risk": self._pending.regime_risk,
                "filter_risk": self._pending.filter_risk,
                "time_risk":   self._pending.time_risk,
                "hl_jump_risk":self._pending.hl_jump_risk,
                "next_move":   self._pending.next_move,
            }
        return {"obs_buf": obs_list, "pending": pending}

    def restore_state(self, snapshot: dict) -> None:
        """Restore from snapshot produced by get_state_snapshot()."""
        self._obs_buf.clear()
        self._pending = None
        for d in snapshot.get("obs_buf", []):
            self._obs_buf.append(_Observation(**d))
        p = snapshot.get("pending")
        if p is not None:
            self._pending = _Observation(**p)
        logger.info(
            f"RiskScoreEngine state restored | "
            f"n_obs={len(self._obs_buf)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m risk.risk_score
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("RiskScoreEngine smoke test")
    print("=" * 65)

    cfg    = SystemConfig.from_yaml("config/params.yaml")
    engine = RiskScoreEngine(cfg)

    def make_inp(**kwargs):
        defaults = dict(
            z_score    = 1.5,
            imm_score  = 0.75,
            P_k        = 0.0001,
            adf_pvalue = 0.01,
            in_trade   = True,
            trade_age  = 3,
            hl_entry   = 4.0,
            hl_current = 4.0,
        )
        defaults.update(kwargs)
        return RiskInput(**defaults)

    # ── Test 1: No-risk baseline ──────────────────────────────────────────
    print("\n[Test 1] No-risk baseline (z=1.5, healthy regime, no HL jump)")
    bar = engine.compute(make_inp())
    print(f"  risk_score      = {bar.risk_score:.4f}")
    print(f"  position_scalar = {bar.position_scalar:.4f}")
    print(f"  z_risk          = {bar.z_risk:.4f}")
    print(f"  regime_risk     = {bar.regime_risk:.4f}")
    print(f"  filter_risk     = {bar.filter_risk:.4f}")
    print(f"  time_risk       = {bar.time_risk:.4f}")
    print(f"  hl_jump_risk    = {bar.hl_jump_risk:.4f}")
    print(f"  weights_source  = {bar.weights_source}")
    assert 0.0 <= bar.risk_score     <= 1.0, "risk_score out of bounds"
    assert 0.0 <= bar.position_scalar <= 1.0, "position_scalar out of bounds"
    print("  ✓ All in bounds")

    # ── Test 2: ZRisk factor ──────────────────────────────────────────────
    print("\n[Test 2] ZRisk factor")
    # z_onset=2.0, z_full=3.5
    assert engine.compute(make_inp(z_score=1.0)).z_risk == 0.0,  "ZRisk should be 0 at z=1.0"
    assert engine.compute(make_inp(z_score=2.0)).z_risk == 0.0,  "ZRisk should be 0 at z=2.0"
    assert engine.compute(make_inp(z_score=3.5)).z_risk == 1.0,  "ZRisk should be 1 at z=3.5"
    mid = engine.compute(make_inp(z_score=2.75)).z_risk
    assert abs(mid - 0.5) < 1e-10, f"ZRisk at midpoint should be 0.5, got {mid:.4f}"
    print("  ✓ ZRisk factor: 0 at onset, 0.5 at midpoint, 1 at full")

    # ── Test 3: RegimeRisk factor ─────────────────────────────────────────
    print("\n[Test 3] RegimeRisk factor")
    # regime_risk_onset = 0.30
    assert engine.compute(make_inp(imm_score=0.30)).regime_risk == 0.0, \
        "RegimeRisk should be 0 at onset=0.30"
    assert engine.compute(make_inp(imm_score=0.0)).regime_risk  == 1.0, \
        "RegimeRisk should be 1 at imm_score=0"
    assert engine.compute(make_inp(imm_score=0.15)).regime_risk == 0.5, \
        "RegimeRisk should be 0.5 at midpoint"
    print("  ✓ RegimeRisk factor correct")

    # ── Test 4: FilterRisk factor ─────────────────────────────────────────
    print("\n[Test 4] FilterRisk factor")
    P_max = cfg.signal.P_max
    assert engine.compute(make_inp(P_k=0.0)).filter_risk     == 0.0, "FilterRisk at P=0"
    assert engine.compute(make_inp(P_k=P_max)).filter_risk   == 1.0, "FilterRisk at P_max"
    assert engine.compute(make_inp(P_k=P_max/2)).filter_risk == 0.5, "FilterRisk at P_max/2"
    print("  ✓ FilterRisk factor correct")

    # ── Test 5: TimeRisk factor ───────────────────────────────────────────
    print("\n[Test 5] TimeRisk factor")
    # mult=2.0, hl_entry=4.0 → TimeRisk=1 at trade_age=8
    assert engine.compute(make_inp(trade_age=0)).time_risk  == 0.0, "TimeRisk at age=0"
    assert engine.compute(make_inp(trade_age=8)).time_risk  == 1.0, "TimeRisk at age=8"
    mid_time = engine.compute(make_inp(trade_age=4)).time_risk
    assert abs(mid_time - 0.25) < 1e-10, \
        f"TimeRisk at half horizon (quadratic) should be 0.25, got {mid_time:.4f}"
    print("  ✓ TimeRisk factor correct (quadratic)")

    # ── Test 6: HLJumpRisk factor ─────────────────────────────────────────
    print("\n[Test 6] HLJumpRisk factor")
    # jump_mult=2.0, hl_entry=4.0 → HLJumpRisk=1 when hl_current=4+8=12
    assert engine.compute(make_inp(hl_current=4.0)).hl_jump_risk  == 0.0, \
        "HLJumpRisk at no change"
    assert engine.compute(make_inp(hl_current=12.0)).hl_jump_risk == 1.0, \
        "HLJumpRisk at full jump"
    assert engine.compute(make_inp(hl_current=2.0)).hl_jump_risk  == 0.0, \
        "HLJumpRisk should be 0 when HL compresses (not a risk)"
    print("  ✓ HLJumpRisk factor correct")

    # ── Test 7: Not in trade → time and HL factors = 0 ───────────────────
    print("\n[Test 7] Not in trade → trade-specific risks = 0")
    bar7 = engine.compute(make_inp(in_trade=False, trade_age=50, hl_current=20.0))
    assert bar7.time_risk    == 0.0, "TimeRisk should be 0 when not in trade"
    assert bar7.hl_jump_risk == 0.0, "HLJumpRisk should be 0 when not in trade"
    assert bar7.z_risk       == 0.0, "ZRisk should be 0 when not in trade"
    print("  ✓ Trade-specific risks = 0 when not in trade")

    # ── Test 8: Adaptive weights stabilise after window obs ───────────────
    print(f"\n[Test 8] Adaptive weights after {cfg.risk.adaptive_window} observations")
    engine8 = RiskScoreEngine(cfg)
    rng     = np.random.default_rng(42)
    prev_spread = 0.001
    for _ in range(cfg.risk.adaptive_window + 10):
        curr_spread = prev_spread + rng.normal(0, 0.0002)
        engine8.record_outcome(abs(curr_spread - prev_spread))
        inp8 = make_inp(
            z_score    = rng.uniform(0, 2.0),
            imm_score  = rng.uniform(0.4, 0.9),
            trade_age  = rng.integers(1, 10),
        )
        bar8 = engine8.compute(inp8)
        prev_spread = curr_spread

    assert bar8.weights_source in ("blended", "estimated"), \
        f"Expected blended weights after enough obs, got: {bar8.weights_source}"
    assert abs(bar8.weights.sum() - 1.0) < 1e-10, \
        f"Weights must sum to 1.0, got {bar8.weights.sum():.6f}"
    print(f"  weights_source = {bar8.weights_source}")
    print(f"  weights        = {np.round(bar8.weights, 4)}")
    print(f"  sum            = {bar8.weights.sum():.6f}")
    print("  ✓ Weights valid and sum to 1.0")

    # ── Test 9: RiskScore=1 → position_scalar=0 ──────────────────────────
    print("\n[Test 9] RiskScore=1 → position_scalar=0 (full exit)")
    # Force RiskScore toward 1: extreme z, dying regime, diverged filter,
    # old trade, large HL jump
    bar9 = engine.compute(make_inp(
        z_score    = 3.5,
        imm_score  = 0.0,
        P_k        = cfg.signal.P_max,
        trade_age  = 100,
        hl_current = 100.0,
    ))
    print(f"  risk_score      = {bar9.risk_score:.4f}")
    print(f"  position_scalar = {bar9.position_scalar:.4f}")
    assert bar9.risk_score      <= 1.0, "risk_score must be <= 1.0"
    assert bar9.position_scalar >= 0.0, "position_scalar must be >= 0.0"
    print("  ✓ Extreme risk handled correctly")

    # ── Test 10: State snapshot / restore ─────────────────────────────────
    print("\n[Test 10] State snapshot / restore")
    snap     = engine8.get_state_snapshot()
    engine10 = RiskScoreEngine(cfg)
    engine10.restore_state(snap)
    assert len(engine10._obs_buf) == len(engine8._obs_buf), \
        "Observation buffer length mismatch after restore"
    bar_orig     = engine8.compute(make_inp())
    bar_restored = engine10.compute(make_inp())
    assert abs(bar_orig.risk_score - bar_restored.risk_score) < 1e-10, \
        f"risk_score mismatch: {bar_orig.risk_score:.6f} vs {bar_restored.risk_score:.6f}"
    print("  ✓ State snapshot / restore produces identical output")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
