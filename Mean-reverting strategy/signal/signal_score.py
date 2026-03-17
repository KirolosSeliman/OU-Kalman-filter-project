"""
signal/signal_score.py
─────────────────────────────────────────────────────────────────────────────
Layer 1: Signal Score computation.

Produces a single value SignalScore ∈ [-1, 1] every bar by multiplying:

    SignalScore = RawSignal × GateFactor × FilterFactor
                           × RegimeFactor × CostFactor

Where:
    RawSignal    = -tanh(z / z_scale)
    GateFactor   = 1.0 if gate OPEN, 0.0 if CLOSED  [only binary factor]
    FilterFactor = max(0, 1 - P_k / P_max)
    RegimeFactor = IMM p1 (mean-reverting model probability)
    CostFactor   = max(0, expected_gross - roundtrip_cost) / expected_gross
                   0.0 when edge disappears after costs

Hard exits bypass Layer 1 entirely:
    |z| >= hard_exit_z   → SignalScore = 0  (caller must exit immediately)
    gate closed mid-trade → SignalScore = 0  (via GateFactor = 0)

Design invariants:
    ANY factor = 0  →  SignalScore = 0  →  no trade, no position
    Sign of SignalScore = direction:  > 0 → long spread, < 0 → short spread
    |SignalScore| = conviction strength for sizing

z-score definition:
    z = (x_hat - mu) / sigma_ou
    where x_hat = Kalman filtered spread, mu and sigma_ou = OU estimates.
    Sign convention: z > 0 → spread above mean → negative RawSignal → short.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclass — one per bar, assembled by pipeline_runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalInput:
    """
    All upstream outputs needed to compute SignalScore for one bar.
    Assembled by pipeline_runner.py from the three engines.
    """
    # From KalmanIMMEngine
    x_hat:       float   # Kalman filtered spread estimate
    P_k:         float   # Kalman error covariance
    imm_score:   float   # p1: mean-reverting model probability ∈ [0, 1]

    # From StationarityOUEngine
    gate_open:   bool    # ADF + Hurst gate
    mu:          float   # OU long-run mean
    sigma_ou:    float   # OU residual std
    ou_valid:    bool    # True if OU params are from current-bar estimation
    ou_ever_valid: bool  # True if at least one valid OU estimate exists

    # From SpreadEngine (for cost factor)
    spread:      float   # raw spread value S_k (used to compute dollar gross)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass — one per bar
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalBar:
    """All outputs of SignalScoreEngine.compute() for a single bar."""
    signal_score:   float   # ∈ [-1, 1], direction + conviction
    z_score:        float   # (x_hat - mu) / sigma_ou
    raw_signal:     float   # -tanh(z / z_scale) before multipliers
    gate_factor:    float   # 0.0 or 1.0
    filter_factor:  float   # ∈ [0, 1]
    regime_factor:  float   # ∈ [0, 1]
    cost_factor:    float   # ∈ [0, 1]
    hard_exit:      bool    # True if |z| >= hard_exit_z
    tradeable:      bool    # True if |signal_score| >= entry_threshold
    valid:          bool    # False if upstream inputs are unusable


# ─────────────────────────────────────────────────────────────────────────────
# SignalScoreEngine
# ─────────────────────────────────────────────────────────────────────────────

class SignalScoreEngine:
    """
    Computes Layer 1 Signal Score every bar.

    Stateless between bars — no internal state.
    All state lives in the upstream engines.
    Call compute() once per bar with fresh SignalInput.

    Correct usage pattern:
        engine = SignalScoreEngine(cfg)
        for each bar k:
            inp = SignalInput(x_hat=..., P_k=..., ...)
            bar = engine.compute(inp)
            if bar.hard_exit:
                # force-close position immediately
            elif bar.tradeable:
                # pass bar.signal_score to Layer 2
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg = cfg

    # ─────────────────────────────────────────────────────────────────────
    # Main computation
    # ─────────────────────────────────────────────────────────────────────

    def compute(self, inp: SignalInput) -> SignalBar:
        """
        Compute SignalScore for one bar.

        Parameters
        ----------
        inp : SignalInput
            All upstream values for this bar.

        Returns
        -------
        SignalBar
            valid=False if inputs are unusable (NaN, missing OU params, etc.)
        """
        sig_cfg = self._cfg.signal
        _nan    = float("nan")

        # ── Step 0: Input validity check ─────────────────────────────────
        # We need: finite x_hat, P_k, imm_score, and at least one valid
        # OU estimate (ou_ever_valid). Without these we cannot compute z.
        if not self._inputs_valid(inp):
            return SignalBar(
                signal_score  = 0.0,
                z_score       = _nan,
                raw_signal    = 0.0,
                gate_factor   = 0.0,
                filter_factor = 0.0,
                regime_factor = 0.0,
                cost_factor   = 0.0,
                hard_exit     = False,
                tradeable     = False,
                valid         = False,
            )

        # ── Step 1: z-score ───────────────────────────────────────────────
        # z = (x_hat - mu) / sigma_ou
        # sigma_ou guaranteed > 0 by _inputs_valid check
        z = float((inp.x_hat - inp.mu) / inp.sigma_ou)

        # ── Step 2: Hard exit check ───────────────────────────────────────
        # Bypasses all continuous logic. Pure binary.
        hard_exit = (abs(z) >= sig_cfg.hard_exit_z)
        if hard_exit:
            return SignalBar(
                signal_score  = 0.0,
                z_score       = z,
                raw_signal    = 0.0,
                gate_factor   = 0.0,
                filter_factor = 0.0,
                regime_factor = 0.0,
                cost_factor   = 0.0,
                hard_exit     = True,
                tradeable     = False,
                valid         = True,
            )

        # ── Step 3: Raw directional signal ───────────────────────────────
        # RawSignal = -tanh(z / z_scale)
        # z > 0 → spread above mean → expect reversion down → short signal
        # z < 0 → spread below mean → expect reversion up  → long signal
        raw_signal = float(-np.tanh(z / sig_cfg.z_scale))

        # ── Step 4: GateFactor ────────────────────────────────────────────
        # Only binary factor in Layer 1.
        # Gate closed → GateFactor = 0 → SignalScore = 0 automatically.
        gate_factor = 1.0 if inp.gate_open else 0.0

        # ── Step 5: FilterFactor ──────────────────────────────────────────
        # Measures Kalman filter convergence.
        # P_k → 0: filter fully converged, FilterFactor → 1.0
        # P_k → P_max: filter uncertain, FilterFactor → 0.0
        P_max         = sig_cfg.P_max
        filter_factor = float(max(0.0, 1.0 - inp.P_k / P_max))

        # ── Step 6: RegimeFactor ──────────────────────────────────────────
        # IMM probability of mean-reverting model.
        # imm_score = 1.0: clearly mean-reverting → full signal pass-through
        # imm_score = 0.2: likely trending → signal muted to 20%
        regime_factor = float(np.clip(inp.imm_score, 0.0, 1.0))

        # ── Step 7: CostFactor ────────────────────────────────────────────
        # Prevents trading when expected gross edge < transaction cost.
        # expected_gross = |z| * sigma_ou  (expected dollar move toward mean)
        # roundtrip_cost = tc_decimal (fraction of spread price)
        #
        # CostFactor = max(0, expected_gross - tc) / expected_gross
        # = 0.0 when expected edge is entirely consumed by costs
        # = 1.0 when costs are negligible relative to edge
        expected_gross = abs(z) * abs(inp.sigma_ou)
        tc             = self._cfg.tc_decimal  # round-trip cost as fraction

        if expected_gross < 1e-12:
            cost_factor = 0.0
        else:
            cost_factor = float(max(0.0,
                (expected_gross - tc) / expected_gross
            ))

        # ── Step 8: Composite SignalScore ─────────────────────────────────
        signal_score = (
            raw_signal
            * gate_factor
            * filter_factor
            * regime_factor
            * cost_factor
        )

        # Numerical safety: clip to [-1, 1]
        signal_score = float(np.clip(signal_score, -1.0, 1.0))

        tradeable = abs(signal_score) >= sig_cfg.entry_threshold

        return SignalBar(
            signal_score  = signal_score,
            z_score       = z,
            raw_signal    = raw_signal,
            gate_factor   = gate_factor,
            filter_factor = filter_factor,
            regime_factor = regime_factor,
            cost_factor   = cost_factor,
            hard_exit     = False,
            tradeable     = tradeable,
            valid         = True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Input validation
    # ─────────────────────────────────────────────────────────────────────

    def _inputs_valid(self, inp: SignalInput) -> bool:
        """
        Return False if any required input is missing or degenerate.
        Logs the specific reason at DEBUG level.
        """
        if not np.isfinite(inp.x_hat):
            logger.debug("SignalScore: x_hat is not finite")
            return False
        if not np.isfinite(inp.P_k) or inp.P_k < 0:
            logger.debug(f"SignalScore: P_k invalid ({inp.P_k})")
            return False
        if not np.isfinite(inp.imm_score):
            logger.debug("SignalScore: imm_score is not finite")
            return False
        if not inp.ou_ever_valid:
            logger.debug("SignalScore: no valid OU estimate yet")
            return False
        if not np.isfinite(inp.mu):
            logger.debug("SignalScore: mu is not finite")
            return False
        if not np.isfinite(inp.sigma_ou) or inp.sigma_ou <= 0:
            logger.debug(f"SignalScore: sigma_ou invalid ({inp.sigma_ou})")
            return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m signal.signal_score
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("SignalScoreEngine smoke test")
    print("=" * 65)

    cfg    = SystemConfig.from_yaml("config/params.yaml")
    engine = SignalScoreEngine(cfg)

    def make_input(**kwargs):
        defaults = dict(
            x_hat       = 0.002,
            P_k         = 0.0001,
            imm_score   = 0.80,
            gate_open   = True,
            mu          = 0.0,
            sigma_ou    = 0.001,
            ou_valid    = True,
            ou_ever_valid = True,
            spread      = 0.002,
        )
        defaults.update(kwargs)
        return SignalInput(**defaults)

    # ── Test 1: Normal signal, gate open, strong z ────────────────────────
    print("\n[Test 1] Normal entry signal (z=2.0, gate open)")
    inp = make_input(x_hat=0.002, mu=0.0, sigma_ou=0.001)
    bar = engine.compute(inp)
    # z = (0.002 - 0.0) / 0.001 = 2.0
    # RawSignal = -tanh(2.0 / 2.0) = -tanh(1.0) ≈ -0.762  (short signal)
    print(f"  z_score       = {bar.z_score:.4f}  (expected 2.0)")
    print(f"  raw_signal    = {bar.raw_signal:.4f}  (expected ≈ -0.762)")
    print(f"  gate_factor   = {bar.gate_factor:.4f}  (expected 1.0)")
    print(f"  filter_factor = {bar.filter_factor:.4f}")
    print(f"  regime_factor = {bar.regime_factor:.4f}  (expected 0.80)")
    print(f"  cost_factor   = {bar.cost_factor:.4f}")
    print(f"  signal_score  = {bar.signal_score:.4f}")
    print(f"  tradeable     = {bar.tradeable}")
    print(f"  hard_exit     = {bar.hard_exit}")

    assert bar.valid,                              "valid should be True"
    assert not bar.hard_exit,                      "hard_exit should be False"
    assert abs(bar.z_score - 2.0)      < 1e-10,   "z_score mismatch"
    assert abs(bar.raw_signal - (-math.tanh(1.0))) < 1e-10, "raw_signal mismatch"
    assert bar.gate_factor  == 1.0,                "gate_factor should be 1.0"
    assert bar.regime_factor == 0.80,              "regime_factor should be 0.80"
    assert bar.signal_score  < 0,                  "signal should be short (negative)"
    print("  ✓ All assertions passed")

    # ── Test 2: Gate closed → signal = 0 ─────────────────────────────────
    print("\n[Test 2] Gate closed → signal must be exactly 0")
    bar2 = engine.compute(make_input(gate_open=False))
    assert bar2.signal_score == 0.0,   "signal_score must be 0 when gate closed"
    assert bar2.gate_factor  == 0.0,   "gate_factor must be 0 when gate closed"
    assert not bar2.tradeable,          "tradeable must be False when gate closed"
    assert bar2.valid,                  "valid should be True even when gate closed"
    print(f"  signal_score = {bar2.signal_score}  ✓")

    # ── Test 3: Hard exit — |z| >= 3.5 ───────────────────────────────────
    print("\n[Test 3] Hard exit (z=4.0)")
    bar3 = engine.compute(make_input(x_hat=0.004, mu=0.0, sigma_ou=0.001))
    # z = 4.0 >= hard_exit_z=3.5
    assert bar3.hard_exit,              "hard_exit should be True for z=4.0"
    assert bar3.signal_score == 0.0,   "signal_score must be 0 on hard exit"
    assert not bar3.tradeable,          "tradeable must be False on hard exit"
    print(f"  z={bar3.z_score:.1f}, hard_exit=True  ✓")

    # ── Test 4: P_k at P_max → FilterFactor = 0 → signal = 0 ────────────
    print("\n[Test 4] P_k = P_max → FilterFactor = 0 → signal = 0")
    bar4 = engine.compute(make_input(P_k=cfg.signal.P_max))
    assert bar4.filter_factor == 0.0,  "filter_factor should be 0 at P_max"
    assert bar4.signal_score  == 0.0,  "signal_score must be 0 at P_max"
    print(f"  filter_factor={bar4.filter_factor}  ✓")

    # ── Test 5: IMM score = 0 → regime_factor = 0 → signal = 0 ──────────
    print("\n[Test 5] IMM score = 0 → signal = 0")
    bar5 = engine.compute(make_input(imm_score=0.0))
    assert bar5.regime_factor == 0.0,  "regime_factor should be 0"
    assert bar5.signal_score  == 0.0,  "signal_score must be 0"
    print(f"  regime_factor={bar5.regime_factor}  ✓")

    # ── Test 6: Invalid input — ou_ever_valid=False ───────────────────────
    print("\n[Test 6] ou_ever_valid=False → valid=False")
    bar6 = engine.compute(make_input(ou_ever_valid=False))
    assert not bar6.valid,             "valid should be False"
    assert bar6.signal_score == 0.0,   "signal_score must be 0"
    print(f"  valid={bar6.valid}  ✓")

    # ── Test 7: Cost factor suppresses tiny-z signals ────────────────────
    print("\n[Test 7] Cost factor — tiny z near 0 → cost_factor ≈ 0")
    bar7 = engine.compute(make_input(
        x_hat=0.0000001, mu=0.0, sigma_ou=0.001
    ))
    # z = 0.0001, expected_gross ≈ 0.0001 * 0.001 = 1e-7 << tc_decimal=0.0005
    print(f"  z={bar7.z_score:.6f}, cost_factor={bar7.cost_factor:.6f}")
    assert bar7.cost_factor == 0.0,    "cost_factor should be 0 for negligible edge"
    print(f"  cost_factor=0.0  ✓")

    # ── Test 8: Sign symmetry — long signal for z < 0 ─────────────────────
    print("\n[Test 8] Sign symmetry (z=-2.0 → long signal)")
    bar8 = engine.compute(make_input(x_hat=-0.002, mu=0.0, sigma_ou=0.001))
    assert bar8.z_score    < 0,        "z should be negative"
    assert bar8.raw_signal > 0,        "raw_signal should be positive (long)"
    assert bar8.signal_score > 0,      "signal_score should be positive (long)"
    print(f"  z={bar8.z_score:.2f}, signal_score={bar8.signal_score:.4f} (long)  ✓")

    # ── Test 9: NaN x_hat → valid=False ───────────────────────────────────
    print("\n[Test 9] NaN x_hat → valid=False")
    bar9 = engine.compute(make_input(x_hat=float("nan")))
    assert not bar9.valid
    assert bar9.signal_score == 0.0
    print(f"  valid={bar9.valid}  ✓")

    # ── Test 10: Signal scale verification ───────────────────────────────
    print("\n[Test 10] Signal scale at key z-scores")
    print(f"  {'z':>6} {'raw_signal':>12} {'|signal_score|':>16}")
    for z_test in [1.5, 2.0, 2.5, 3.0, 3.4]:
        b = engine.compute(make_input(
            x_hat   = z_test * 0.001,
            mu      = 0.0,
            sigma_ou = 0.001,
            P_k     = 0.0001,
            imm_score = 1.0,
        ))
        if not b.hard_exit:
            print(f"  {z_test:>6.1f} {b.raw_signal:>12.4f} {abs(b.signal_score):>16.4f}")
    print("  ✓ Scale table printed")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
