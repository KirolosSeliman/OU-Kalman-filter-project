"""
risk/target_position.py
─────────────────────────────────────────────────────────────────────────────
Layer 2: Target Position computation.

Produces every bar:
    TargetPosition ∈ [-1, 1]   — normalized desired exposure
    MaxShares      ∈ [0, ∞)    — risk-budgeted share ceiling for this bar
    TimeDecay      ∈ [0, 1]    — time-based position reduction factor

Formula:
    TargetPosition = SignalScore × (1 - sqrt(RiskScore)) × TimeDecay

    Note: (1 - sqrt(RiskScore)) = position_scalar from RiskScoreEngine.
    We accept it as an input rather than recomputing it here.

TimeDecay (quadratic concave decay, using LOCKED HL_entry):
    TimeDecay = max(0, 1 - (trade_age / (3 × HL_entry))^2)

    age = 0          → TimeDecay = 1.00
    age = 1×HL_entry → TimeDecay = 0.89  (slow initial decay)
    age = 2×HL_entry → TimeDecay = 0.56
    age = 3×HL_entry → TimeDecay = 0.00  (automatic full exit)

    CRITICAL: HL_entry is locked at the bar the trade was opened.
    It is NEVER updated mid-trade. This prevents an extending HL
    from resetting the time clock on a losing trade.

MaxShares (risk-budgeted):
    stop_distance = hard_exit_z × sigma_ou × leg1_price  [dollar risk per share]
    RiskPct (Mode A) = risk_pct_fixed
    RiskPct (Mode B) = lerp(min_risk, max_risk,
                           clip((|z| - 1.5) / (2.5 - 1.5), 0, 1))
    MaxShares = floor(account_size × RiskPct × imm_score / stop_distance)
    MaxShares = clip(MaxShares, 0, position_cap_shares)
    position_cap_shares = floor(account_size × position_cap_pct / leg1_price)

Design invariants:
    TargetPosition = 0 when SignalScore = 0    (gate closure exits automatically)
    TargetPosition = 0 when RiskScore  = 1    (time exit built in)
    TargetPosition = 0 when TimeDecay  = 0    (time exit built in)
    MaxShares = 0 when stop_distance is degenerate → no trade possible
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetPositionInput:
    """All inputs needed for Layer 2 computation."""
    # From Layer 1
    signal_score:     float   # ∈ [-1, 1]

    # From RiskScoreEngine
    position_scalar:  float   # (1 - sqrt(RiskScore)) ∈ [0, 1]

    # From StationarityOUEngine
    sigma_ou:         float   # OU residual std
    hl_current:       float   # current HL estimate

    # From KalmanIMMEngine
    imm_score:        float   # p1, mean-reverting probability ∈ [0, 1]
    z_score:          float   # current z-score

    # From SpreadEngine / market data
    leg1_price:       float   # GLD price (for dollar risk calculation)

    # From trade state (PositionManager)
    in_trade:         bool
    trade_age:        int     # bars since entry (0 on entry bar)
    hl_entry:         float   # HL at the moment of entry — LOCKED, never changes


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetPositionBar:
    """All outputs of TargetPositionEngine.compute() for one bar."""
    target_position:  float   # ∈ [-1, 1] normalized desired exposure
    max_shares:       int     # risk-budgeted share ceiling (Layer 3 uses this)
    time_decay:       float   # ∈ [0, 1]
    risk_pct_used:    float   # actual RiskPct applied (Mode A or B)
    stop_distance:    float   # dollar risk per share (diagnostic)
    valid:            bool    # False if inputs are degenerate


# ─────────────────────────────────────────────────────────────────────────────
# TargetPositionEngine
# ─────────────────────────────────────────────────────────────────────────────

class TargetPositionEngine:
    """
    Computes Layer 2 Target Position.

    Stateless between bars — no internal state.
    HL_entry locking is the responsibility of PositionManager, which
    records HL_entry at trade open and passes it unchanged on every bar.

    Correct usage pattern:
        engine = TargetPositionEngine(cfg)
        for each bar k:
            inp = TargetPositionInput(...)
            bar = engine.compute(inp)
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg = cfg

    # ─────────────────────────────────────────────────────────────────────
    # Main computation
    # ─────────────────────────────────────────────────────────────────────

    def compute(self, inp: TargetPositionInput) -> TargetPositionBar:
        """
        Compute TargetPosition, MaxShares, and TimeDecay for one bar.
        """
        _nan = float("nan")

        # ── Step 1: Input validity ────────────────────────────────────────
        if not self._inputs_valid(inp):
            return TargetPositionBar(
                target_position = 0.0,
                max_shares      = 0,
                time_decay      = 0.0,
                risk_pct_used   = 0.0,
                stop_distance   = 0.0,
                valid           = False,
            )

        sig_cfg  = self._cfg.signal
        siz_cfg  = self._cfg.sizing

        # ── Step 2: TimeDecay ─────────────────────────────────────────────
        time_decay = self._compute_time_decay(inp)

        # ── Step 3: RiskPct ───────────────────────────────────────────────
        risk_pct = self._compute_risk_pct(inp)

        # ── Step 4: MaxShares ─────────────────────────────────────────────
        # stop_distance = hard_exit_z × sigma_ou × leg1_price
        # = expected dollar loss per share if spread moves to hard stop
        stop_distance = (
            sig_cfg.hard_exit_z
            * abs(inp.sigma_ou)
            * abs(inp.leg1_price)
        )

        if stop_distance < 1e-8:
            logger.warning(
                f"TargetPositionEngine: stop_distance={stop_distance:.2e} "
                "is degenerate. MaxShares=0."
            )
            return TargetPositionBar(
                target_position = 0.0,
                max_shares      = 0,
                time_decay      = time_decay,
                risk_pct_used   = risk_pct,
                stop_distance   = stop_distance,
                valid           = False,
            )

        raw_max_shares = (
            siz_cfg.account_size
            * risk_pct
            * float(np.clip(inp.imm_score, 0.0, 1.0))
            / stop_distance
        )
        max_shares_uncapped = int(math.floor(raw_max_shares))

        # Position cap: never exceed position_cap_pct of account in one pair
        position_cap_shares = int(math.floor(
            siz_cfg.account_size
            * siz_cfg.position_cap_pct
            / max(inp.leg1_price, 1e-8)
        ))

        max_shares = max(0, min(max_shares_uncapped, position_cap_shares))

        # ── Step 5: TargetPosition ────────────────────────────────────────
        # TargetPosition = SignalScore × position_scalar × TimeDecay
        # All three factors ∈ [-1, 1] or [0, 1]
        # Result ∈ [-1, 1]
        target_position = float(
            inp.signal_score
            * float(np.clip(inp.position_scalar, 0.0, 1.0))
            * time_decay
        )
        target_position = float(np.clip(target_position, -1.0, 1.0))

        return TargetPositionBar(
            target_position = target_position,
            max_shares      = max_shares,
            time_decay      = time_decay,
            risk_pct_used   = risk_pct,
            stop_distance   = stop_distance,
            valid           = True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # TimeDecay
    # ─────────────────────────────────────────────────────────────────────

    def _compute_time_decay(self, inp: TargetPositionInput) -> float:
        """
        Quadratic concave time decay using locked HL_entry.

        TimeDecay = max(0, 1 - (trade_age / (3 × HL_entry))^2)

        Not-in-trade: TimeDecay = 1.0 (no decay before entry)
        Invalid HL_entry: TimeDecay = 1.0 (fail-safe: do not force exit)
        """
        if not inp.in_trade:
            return 1.0

        hl = inp.hl_entry
        if not np.isfinite(hl) or hl <= 0:
            logger.warning(
                f"TargetPositionEngine: hl_entry={hl} invalid. "
                "TimeDecay=1.0 (no decay applied)."
            )
            return 1.0

        horizon = 3.0 * hl
        ratio   = inp.trade_age / horizon
        decay   = max(0.0, 1.0 - ratio ** 2)
        return float(decay)

    # ─────────────────────────────────────────────────────────────────────
    # RiskPct (Mode A / Mode B)
    # ─────────────────────────────────────────────────────────────────────

    def _compute_risk_pct(self, inp: TargetPositionInput) -> float:
        """
        Mode A: fixed risk_pct_fixed regardless of z-score.
        Mode B: linearly interpolated between min_risk and max_risk
                as |z| moves from 1.5 to 2.5.
                Below 1.5: min_risk. Above 2.5: max_risk.
        """
        cfg = self._cfg.sizing

        if cfg.mode == "A":
            return float(cfg.risk_pct_fixed)

        # Mode B: z-score scaled
        z_abs    = abs(inp.z_score)
        z_onset  = 1.5
        z_full   = 2.5
        z_range  = z_full - z_onset

        t = float(np.clip((z_abs - z_onset) / z_range, 0.0, 1.0))
        risk_pct = cfg.risk_pct_min + t * (cfg.risk_pct_max - cfg.risk_pct_min)
        return float(np.clip(risk_pct, cfg.risk_pct_min, cfg.risk_pct_max))

    # ─────────────────────────────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────────────────────────────

    def _inputs_valid(self, inp: TargetPositionInput) -> bool:
        if not np.isfinite(inp.signal_score):
            logger.debug("TargetPosition: signal_score not finite")
            return False
        if not np.isfinite(inp.position_scalar) or inp.position_scalar < 0:
            logger.debug("TargetPosition: position_scalar invalid")
            return False
        if not np.isfinite(inp.sigma_ou) or inp.sigma_ou <= 0:
            logger.debug("TargetPosition: sigma_ou invalid")
            return False
        if not np.isfinite(inp.leg1_price) or inp.leg1_price <= 0:
            logger.debug("TargetPosition: leg1_price invalid")
            return False
        if not np.isfinite(inp.imm_score):
            logger.debug("TargetPosition: imm_score not finite")
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m risk.target_position
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("TargetPositionEngine smoke test")
    print("=" * 65)

    cfg    = SystemConfig.from_yaml("config/params.yaml")
    engine = TargetPositionEngine(cfg)

    def make_inp(**kwargs):
        defaults = dict(
            signal_score    = -0.70,
            position_scalar = 0.80,
            sigma_ou        = 0.001,
            hl_current      = 4.0,
            imm_score       = 0.75,
            z_score         = 2.0,
            leg1_price      = 190.0,
            in_trade        = True,
            trade_age       = 3,
            hl_entry        = 4.0,
        )
        defaults.update(kwargs)
        return TargetPositionInput(**defaults)

    # ── Test 1: Normal short signal ───────────────────────────────────────
    print("\n[Test 1] Normal short signal")
    bar = engine.compute(make_inp())
    print(f"  target_position = {bar.target_position:.4f}  (should be negative)")
    print(f"  max_shares      = {bar.max_shares}")
    print(f"  time_decay      = {bar.time_decay:.4f}")
    print(f"  risk_pct_used   = {bar.risk_pct_used:.4f}")
    print(f"  stop_distance   = {bar.stop_distance:.4f}")
    print(f"  valid           = {bar.valid}")
    assert bar.valid
    assert bar.target_position < 0,        "Short signal should give negative target"
    assert bar.max_shares > 0,             "MaxShares should be positive"
    assert 0.0 < bar.time_decay <= 1.0,   "TimeDecay should be in (0, 1]"
    print("  ✓ Normal short signal correct")

    # ── Test 2: TimeDecay values ──────────────────────────────────────────
    print("\n[Test 2] TimeDecay at key trade ages (HL_entry=4.0)")
    # Horizon = 3 × 4 = 12 bars
    # age=0:  1 - (0/12)^2 = 1.00
    # age=4:  1 - (4/12)^2 = 1 - (1/3)^2 = 1 - 0.111 = 0.889
    # age=8:  1 - (8/12)^2 = 1 - (2/3)^2 = 1 - 0.444 = 0.556
    # age=12: 1 - (12/12)^2 = 0.000
    tests = [(0, 1.000), (4, 0.889), (8, 0.556), (12, 0.000)]
    for age, expected in tests:
        b = engine.compute(make_inp(trade_age=age))
        assert abs(b.time_decay - expected) < 0.001, \
            f"TimeDecay at age={age}: expected {expected:.3f}, got {b.time_decay:.3f}"
        print(f"  age={age:>2} → TimeDecay={b.time_decay:.4f}  (expected {expected:.3f}) ✓")

    # ── Test 3: TimeDecay = 1 when not in trade ───────────────────────────
    print("\n[Test 3] TimeDecay = 1 when not in trade")
    bar3 = engine.compute(make_inp(in_trade=False, trade_age=100))
    assert bar3.time_decay == 1.0, "TimeDecay should be 1.0 when not in trade"
    print(f"  time_decay = {bar3.time_decay}  ✓")

    # ── Test 4: Mode B RiskPct interpolation ─────────────────────────────
    print("\n[Test 4] Mode B RiskPct (z-score scaled)")
    # min_risk=0.0075, max_risk=0.0200, onset=1.5, full=2.5
    cases = [
        (0.5,  0.0075),   # below onset → min_risk
        (1.5,  0.0075),   # at onset    → min_risk
        (2.0,  0.01375),  # midpoint    → (0.0075 + 0.0200) / 2
        (2.5,  0.0200),   # at full     → max_risk
        (3.0,  0.0200),   # above full  → max_risk
    ]
    for z, expected_pct in cases:
        b = engine.compute(make_inp(z_score=z))
        assert abs(b.risk_pct_used - expected_pct) < 1e-6, \
            f"RiskPct at z={z}: expected {expected_pct:.4f}, got {b.risk_pct_used:.4f}"
        print(f"  z={z:.1f} → risk_pct={b.risk_pct_used:.4f}  (expected {expected_pct:.4f}) ✓")

    # ── Test 5: SignalScore = 0 → TargetPosition = 0 ─────────────────────
    print("\n[Test 5] SignalScore=0 → TargetPosition=0")
    bar5 = engine.compute(make_inp(signal_score=0.0))
    assert bar5.target_position == 0.0
    print(f"  target_position = {bar5.target_position}  ✓")

    # ── Test 6: position_scalar=0 (RiskScore=1) → TargetPosition=0 ───────
    print("\n[Test 6] position_scalar=0 → TargetPosition=0 (auto exit)")
    bar6 = engine.compute(make_inp(position_scalar=0.0))
    assert bar6.target_position == 0.0
    print(f"  target_position = {bar6.target_position}  ✓")

    # ── Test 7: TimeDecay=0 (age=3×HL) → TargetPosition=0 ───────────────
    print("\n[Test 7] TimeDecay=0 at 3×HL_entry → TargetPosition=0 (auto exit)")
    bar7 = engine.compute(make_inp(trade_age=12))  # 3 × HL_entry=4
    assert bar7.time_decay == 0.0
    assert bar7.target_position == 0.0
    print(f"  trade_age=12, time_decay=0.0, target_position=0.0  ✓")

    # ── Test 8: Position cap applied ─────────────────────────────────────
    print("\n[Test 8] Position cap (sigma_ou very small → huge raw MaxShares)")
    bar8 = engine.compute(make_inp(sigma_ou=0.0000001))
    cap  = int(math.floor(
        cfg.sizing.account_size * cfg.sizing.position_cap_pct / 190.0
    ))
    assert bar8.max_shares <= cap, \
        f"MaxShares {bar8.max_shares} exceeds cap {cap}"
    print(f"  max_shares={bar8.max_shares} <= cap={cap}  ✓")

    # ── Test 9: Invalid leg1_price → valid=False ──────────────────────────
    print("\n[Test 9] Invalid inputs → valid=False")
    assert not engine.compute(make_inp(leg1_price=0.0)).valid
    assert not engine.compute(make_inp(sigma_ou=0.0)).valid
    assert not engine.compute(make_inp(signal_score=float("nan"))).valid
    print("  ✓ Invalid inputs return valid=False")

    # ── Test 10: Long signal is positive ─────────────────────────────────
    print("\n[Test 10] Long signal (signal_score > 0) → target_position > 0")
    bar10 = engine.compute(make_inp(signal_score=0.60))
    assert bar10.target_position > 0
    print(f"  target_position = {bar10.target_position:.4f}  ✓")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\nTarget position at key signal/risk combinations:")
    print(f"  {'signal':>8} {'scalar':>8} {'decay':>8} {'target':>10}")
    for s, sc, d in [(0.80, 1.00, 1.00), (0.80, 0.50, 1.00),
                     (0.80, 1.00, 0.56), (-0.80, 1.00, 1.00)]:
        b = engine.compute(make_inp(
            signal_score=s, position_scalar=sc,
            trade_age=int(8 if d < 0.57 else 0)
        ))
        print(f"  {s:>8.2f} {sc:>8.2f} {b.time_decay:>8.3f} {b.target_position:>10.4f}")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
