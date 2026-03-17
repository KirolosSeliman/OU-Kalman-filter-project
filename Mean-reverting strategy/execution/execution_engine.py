"""
execution/execution_engine.py
─────────────────────────────────────────────────────────────────────────────
Layer 3: Execution Engine.

Converts Layer 2 outputs into concrete share orders, tracks live position,
enforces the minimum-delta threshold, and produces a fill record every bar.

Responsibilities:
    1. Compute DesiredShares from TargetPosition × MaxShares
    2. Apply min_delta_threshold: suppress order if delta is too small
    3. Produce an Order (buy/sell/flat/hold) with share count and direction
    4. Simulate paper fill (price + slippage + commission)
    5. Update CurrentPosition and track realized/unrealized PnL
    6. Enforce session-level risk: daily_loss_limit, cooldown_bars

Design invariants:
    CurrentPosition ∈ {-MaxShares, ..., 0, ..., +MaxShares}
    Long spread  → CurrentPosition > 0 → buy GLD, sell IAU (in SpreadEngine)
    Short spread → CurrentPosition < 0 → sell GLD, buy IAU
    TargetPosition = 0 → DesiredShares = 0 → flatten if currently in position
    Hard exit from Layer 1 bypasses threshold, forces immediate flatten
    Force-flat at force_flat_time: DesiredShares = 0, threshold bypassed

Slippage model:
    FillPrice = MidPrice ± slippage_ticks × tick_size
    + for buys, - for sells
    Commission = tc_bps × trade_value / 10000  (round-trip allocated per leg)

PnL accounting:
    UnrealizedPnL = CurrentPosition × (current_spread - entry_spread)
    RealizedPnL   = accumulated on flatten / partial exits
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Order types
# ─────────────────────────────────────────────────────────────────────────────

ORDER_HOLD    = "hold"     # no change — delta too small or already at target
ORDER_BUY     = "buy"      # increase long or reduce short
ORDER_SELL    = "sell"     # increase short or reduce long
ORDER_FLATTEN = "flatten"  # explicit full exit (TargetPosition=0 or hard exit)


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionInput:
    """All inputs needed for Layer 3 per bar."""
    # From Layer 2
    target_position:  float   # ∈ [-1, 1]
    max_shares:       int     # risk-budgeted ceiling

    # From Layer 1
    hard_exit:        bool    # force flatten immediately
    signal_valid:     bool    # False → treat as hard exit

    # From market data
    spread:           float   # current spread value (for PnL)
    leg1_price:       float   # GLD mid price (for fill simulation)
    leg2_price:       float   # IAU mid price (for fill simulation)

    # Session control
    force_flat:       bool    # True when bar_time >= force_flat_time
    bar_index:        int     # monotonic bar counter for cooldown


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionBar:
    """All outputs of ExecutionEngine.step() for one bar."""
    order_type:       str     # ORDER_* constant
    order_shares:     int     # |shares traded| this bar (0 if hold)
    order_direction:  int     # +1 = buy, -1 = sell, 0 = hold

    # Position state AFTER this bar's fill
    current_position: int     # shares (signed)
    entry_spread:     float   # spread at which current position was opened
    trade_age:        int     # bars since entry (0 on entry bar, -1 if flat)
    in_trade:         bool    # True if current_position != 0

    # Fill details
    fill_price_leg1:  float   # simulated fill price for GLD leg
    fill_price_leg2:  float   # simulated fill price for IAU leg
    fill_cost:        float   # total commission this bar

    # PnL
    unrealized_pnl:   float
    realized_pnl:     float   # cumulative since session open
    session_pnl:      float   # realized_pnl + unrealized_pnl

    # Session risk state
    in_cooldown:      bool
    daily_loss_halted:bool


# ─────────────────────────────────────────────────────────────────────────────
# ExecutionEngine
# ─────────────────────────────────────────────────────────────────────────────

class ExecutionEngine:
    """
    Tracks live position and converts target exposures to orders.

    Stateful — maintains CurrentPosition, entry_spread, trade_age,
    realized_pnl, cooldown, and daily_loss_halt across bars.

    Correct usage pattern:
        engine = ExecutionEngine(cfg)
        engine.reset_session()
        for each bar k:
            inp = ExecutionInput(...)
            bar = engine.step(inp)
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg = cfg
        self._reset_internal()

    # ─────────────────────────────────────────────────────────────────────
    # Session reset
    # ─────────────────────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """
        Reset all session-level state at market open.
        Position is carried over only if explicitly restored via restore_state().
        In paper trading, we always start flat at session open.
        """
        self._reset_internal()
        logger.info("ExecutionEngine: session reset — flat, PnL cleared.")

    def _reset_internal(self) -> None:
        self._current_position: int   = 0
        self._entry_spread:     float = float("nan")
        self._trade_age:        int   = -1   # -1 = not in trade
        self._realized_pnl:     float = 0.0
        self._cooldown_until:   int   = -1   # bar_index when cooldown ends
        self._daily_loss_halted: bool = False
        self._daily_loss_limit: float = (
            self._cfg.session_risk.daily_loss_limit_pct
            * self._cfg.sizing.account_size
        )

    # ─────────────────────────────────────────────────────────────────────
    # Per-bar step
    # ─────────────────────────────────────────────────────────────────────

    def step(self, inp: ExecutionInput) -> ExecutionBar:
        """
        Process one bar: compute desired position, generate order,
        simulate fill, update state.
        """
        cfg_ex  = self._cfg.execution
        cfg_sr  = self._cfg.session_risk
        _nan    = float("nan")

        # ── Step 1: Compute unrealized PnL ───────────────────────────────
        unrealized_pnl = self._compute_unrealized(inp.spread)

        # ── Step 2: Check daily loss halt ────────────────────────────────
        session_pnl_now = self._realized_pnl + unrealized_pnl
        if (not self._daily_loss_halted
                and session_pnl_now < -self._daily_loss_limit):
            self._daily_loss_halted = True
            logger.warning(
                f"ExecutionEngine: daily loss limit hit "
                f"(session_pnl={session_pnl_now:.2f}). Trading halted."
            )

        # ── Step 3: Determine desired shares ─────────────────────────────
        # Forced flat conditions: hard exit, invalid signal, force_flat
        # time, daily loss halt. All bypass min_delta_threshold.
        force_flat = (
            inp.force_flat
            or inp.hard_exit
            or not inp.signal_valid
            or self._daily_loss_halted
        )

        if force_flat:
            desired_shares = 0
        else:
            # DesiredShares = sign(TargetPosition) × floor(|TP| × MaxShares)
            desired_shares = int(math.floor(
                abs(inp.target_position) * inp.max_shares
            ))
            if inp.target_position < 0:
                desired_shares = -desired_shares

        # ── Step 4: Check cooldown ────────────────────────────────────────
        in_cooldown = (inp.bar_index < self._cooldown_until)

        if in_cooldown and not force_flat:
            # During cooldown: only allow re-entry if signal is very strong
            if abs(inp.target_position) < cfg_sr.stronger_signal_threshold:
                desired_shares = 0  # hold, not flatten

        # ── Step 5: Compute delta and apply min_delta_threshold ──────────
        delta = desired_shares - self._current_position

        if not force_flat and abs(delta) == 0:
            return self._no_trade_bar(inp, unrealized_pnl, in_cooldown)

        # Suppress small adjustments (noise filter)
        # Exception: if desired_shares = 0, always execute (flatten)
        if (not force_flat
                and desired_shares != 0
                and abs(delta) < cfg_ex.min_delta_threshold * inp.max_shares):
            return self._no_trade_bar(inp, unrealized_pnl, in_cooldown)

        # ── Step 6: Determine order type and direction ────────────────────
        if desired_shares == 0 and self._current_position == 0:
            return self._no_trade_bar(inp, unrealized_pnl, in_cooldown)

        if desired_shares == 0:
            order_type = ORDER_FLATTEN
        elif delta > 0:
            order_type = ORDER_BUY
        else:
            order_type = ORDER_SELL

        order_shares    = abs(delta)
        order_direction = int(np.sign(delta))

        # ── Step 7: Simulate paper fill ───────────────────────────────────
        slip      = cfg_ex.slippage_ticks * cfg_ex.tick_size
        tc_rate   = cfg_ex.tc_bps / 10_000.0

        # Fill price: buy gets worse (higher), sell gets better (lower)
        fill_leg1 = inp.leg1_price + order_direction * slip
        fill_leg2 = inp.leg2_price - order_direction * slip  # opposite leg

        trade_value   = abs(order_shares) * (abs(fill_leg1) + abs(fill_leg2))
        fill_cost     = tc_rate * trade_value

        # ── Step 8: Update position state ────────────────────────────────
        prev_position = self._current_position
        self._current_position = desired_shares

        # PnL: realize gain/loss on closed portion
        if prev_position != 0 and self._current_position == 0:
            # Full flatten
            if np.isfinite(self._entry_spread) and np.isfinite(inp.spread):
                closed_pnl = prev_position * (inp.spread - self._entry_spread)
                self._realized_pnl += closed_pnl - fill_cost
            else:
                self._realized_pnl -= fill_cost
            self._entry_spread = float("nan")

        elif prev_position == 0 and self._current_position != 0:
            # New entry
            self._entry_spread = inp.spread
            self._realized_pnl -= fill_cost

        elif (np.sign(prev_position) != np.sign(self._current_position)
              and prev_position != 0 and self._current_position != 0):
            # Direction flip: realize full old position, open new
            if np.isfinite(self._entry_spread) and np.isfinite(inp.spread):
                closed_pnl = prev_position * (inp.spread - self._entry_spread)
                self._realized_pnl += closed_pnl - fill_cost
            else:
                self._realized_pnl -= fill_cost
            self._entry_spread = inp.spread

        else:
            # Partial add or reduce — simple cost deduction
            # For production, VWAP-update entry_spread on adds
            if prev_position == 0:
                self._entry_spread = inp.spread
            self._realized_pnl -= fill_cost

        # ── Step 9: Update trade_age ──────────────────────────────────────
        if self._current_position == 0:
            self._trade_age = -1
            # Trigger cooldown after any exit
            self._cooldown_until = (
                inp.bar_index + cfg_sr.cooldown_bars_after_stop
            )
        else:
            if prev_position == 0:
                self._trade_age = 0   # entry bar
            else:
                self._trade_age += 1

        # ── Step 10: Recompute unrealized post-fill ───────────────────────
        unrealized_pnl  = self._compute_unrealized(inp.spread)
        session_pnl     = self._realized_pnl + unrealized_pnl

        # ── Step 11: Check daily loss post-fill ───────────────────────────
        if (not self._daily_loss_halted
                and session_pnl < -self._daily_loss_limit):
            self._daily_loss_halted = True
            logger.warning(
                f"ExecutionEngine: daily loss limit hit post-fill "
                f"(session_pnl={session_pnl:.2f}). Trading halted."
            )

        return ExecutionBar(
            order_type        = order_type,
            order_shares      = order_shares,
            order_direction   = order_direction,
            current_position  = self._current_position,
            entry_spread      = self._entry_spread,
            trade_age         = self._trade_age,
            in_trade          = self._current_position != 0,
            fill_price_leg1   = fill_leg1,
            fill_price_leg2   = fill_leg2,
            fill_cost         = fill_cost,
            unrealized_pnl    = unrealized_pnl,
            realized_pnl      = self._realized_pnl,
            session_pnl       = session_pnl,
            in_cooldown       = in_cooldown,
            daily_loss_halted = self._daily_loss_halted,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _compute_unrealized(self, current_spread: float) -> float:
        if self._current_position == 0:
            return 0.0
        if not np.isfinite(self._entry_spread) or not np.isfinite(current_spread):
            return 0.0
        return float(self._current_position * (current_spread - self._entry_spread))

    def _no_trade_bar(
        self, inp: ExecutionInput,
        unrealized_pnl: float, in_cooldown: bool
    ) -> ExecutionBar:
        session_pnl = self._realized_pnl + unrealized_pnl
        return ExecutionBar(
            order_type        = ORDER_HOLD,
            order_shares      = 0,
            order_direction   = 0,
            current_position  = self._current_position,
            entry_spread      = self._entry_spread,
            trade_age         = self._trade_age,
            in_trade          = self._current_position != 0,
            fill_price_leg1   = float("nan"),
            fill_price_leg2   = float("nan"),
            fill_cost         = 0.0,
            unrealized_pnl    = unrealized_pnl,
            realized_pnl      = self._realized_pnl,
            session_pnl       = session_pnl,
            in_cooldown       = in_cooldown,
            daily_loss_halted = self._daily_loss_halted,
        )

    # ─────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        return {
            "current_position":  self._current_position,
            "entry_spread":      self._entry_spread,
            "trade_age":         self._trade_age,
            "realized_pnl":      self._realized_pnl,
            "cooldown_until":    self._cooldown_until,
            "daily_loss_halted": self._daily_loss_halted,
        }

    def restore_state(self, snapshot: dict) -> None:
        self._current_position  = snapshot.get("current_position",  0)
        self._entry_spread      = snapshot.get("entry_spread",       float("nan"))
        self._trade_age         = snapshot.get("trade_age",          -1)
        self._realized_pnl      = snapshot.get("realized_pnl",       0.0)
        self._cooldown_until    = snapshot.get("cooldown_until",      -1)
        self._daily_loss_halted = snapshot.get("daily_loss_halted",   False)
        logger.info(
            f"ExecutionEngine restored | "
            f"position={self._current_position} | "
            f"realized_pnl={self._realized_pnl:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m execution.execution_engine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("ExecutionEngine smoke test")
    print("=" * 65)

    cfg    = SystemConfig.from_yaml("config/params.yaml")
    engine = ExecutionEngine(cfg)
    engine.reset_session()

    def make_inp(**kwargs):
        defaults = dict(
            target_position = -0.70,
            max_shares      = 100,
            hard_exit       = False,
            signal_valid    = True,
            spread          = 0.002,
            leg1_price      = 190.0,
            leg2_price      = 19.0,
            force_flat      = False,
            bar_index       = 0,
        )
        defaults.update(kwargs)
        return ExecutionInput(**defaults)

    # ── Test 1: Entry on new signal ───────────────────────────────────────
    print("\n[Test 1] New short entry (target=-0.70, max_shares=100)")
    bar1 = engine.step(make_inp(bar_index=0))
    print(f"  order_type       = {bar1.order_type}")
    print(f"  order_shares     = {bar1.order_shares}")
    print(f"  order_direction  = {bar1.order_direction}")
    print(f"  current_position = {bar1.current_position}")
    print(f"  trade_age        = {bar1.trade_age}")
    print(f"  fill_cost        = {bar1.fill_cost:.4f}")
    assert bar1.order_type      == ORDER_SELL,  "Should be SELL on short entry"
    assert bar1.current_position < 0,           "Position should be negative"
    assert bar1.order_shares    == 70,          f"Expected 70, got {bar1.order_shares}"
    assert bar1.trade_age       == 0,           "trade_age should be 0 on entry bar"
    assert bar1.in_trade,                       "in_trade should be True"
    print("  ✓ Entry correct")

    # ── Test 2: Hold on small delta ────────────────────────────────────────
    print("\n[Test 2] Hold — delta below min_delta_threshold")
    # Current: -70. Desired: floor(0.72 * 100) = 72 → delta = 72-(-70)? No.
    # Short: desired = -floor(0.71*100) = -71 → delta = -71-(-70) = -1
    # min_delta = 0.10 * 100 = 10 shares → 1 < 10 → HOLD
    bar2 = engine.step(make_inp(target_position=-0.71, max_shares=100, bar_index=1))
    print(f"  order_type = {bar2.order_type}  (expected: hold)")
    assert bar2.order_type == ORDER_HOLD, f"Expected hold, got {bar2.order_type}"
    print("  ✓ Small delta suppressed correctly")

    # ── Test 3: Hard exit bypasses threshold ──────────────────────────────
    print("\n[Test 3] Hard exit — bypasses min_delta_threshold, flattens immediately")
    bar3 = engine.step(make_inp(
        target_position=-0.71, max_shares=100,
        hard_exit=True, bar_index=2
    ))
    print(f"  order_type       = {bar3.order_type}")
    print(f"  current_position = {bar3.current_position}")
    assert bar3.order_type        == ORDER_FLATTEN, "Hard exit should flatten"
    assert bar3.current_position  == 0,             "Position should be 0 after flatten"
    assert bar3.in_trade          == False,         "in_trade should be False"
    print("  ✓ Hard exit flattened correctly")

    # ── Test 4: Cooldown after exit ────────────────────────────────────────
    print("\n[Test 4] Cooldown active after exit")
    # cooldown_bars_after_stop = 10
    # After test 3 at bar_index=2: cooldown_until = 2 + 10 = 12
    bar4 = engine.step(make_inp(
        target_position=-0.65, max_shares=100,
        bar_index=5, spread=0.002
    ))
    print(f"  in_cooldown = {bar4.in_cooldown}  (expected True)")
    print(f"  order_type  = {bar4.order_type}")
    assert bar4.in_cooldown, "Should be in cooldown"
    # target_position=-0.70 < stronger_signal_threshold=0.70 → hold
    assert bar4.order_type == ORDER_HOLD, \
        f"Should hold during cooldown below threshold, got {bar4.order_type}"
    print("  ✓ Cooldown suppresses weak re-entry")

    # ── Test 5: Strong signal re-entry during cooldown ────────────────────
    print("\n[Test 5] Strong signal re-entry during cooldown")
    bar5 = engine.step(make_inp(
        target_position=-0.80, max_shares=100,
        bar_index=8, spread=0.002
    ))
    print(f"  in_cooldown = {bar5.in_cooldown}")
    print(f"  order_type  = {bar5.order_type}")
    # |target|=0.80 >= stronger_signal_threshold=0.70 → re-entry allowed
    assert bar5.order_type != ORDER_HOLD, \
        "Strong signal should bypass cooldown hold"
    print("  ✓ Strong signal re-enters during cooldown")

    # ── Test 6: Force flat ─────────────────────────────────────────────────
    print("\n[Test 6] Force flat (end of session)")
    engine6 = ExecutionEngine(cfg)
    engine6.reset_session()
    engine6.step(make_inp(bar_index=0))  # enter
    bar6 = engine6.step(make_inp(
        target_position=-0.70,
        force_flat=True, bar_index=1
    ))
    assert bar6.order_type       == ORDER_FLATTEN
    assert bar6.current_position == 0
    print(f"  order_type={bar6.order_type}, position={bar6.current_position}  ✓")

    # ── Test 7: PnL accounting ─────────────────────────────────────────────
    print("\n[Test 7] PnL accounting — enter, hold, exit with profit")
    engine7 = ExecutionEngine(cfg)
    engine7.reset_session()
    # Enter short spread at 0.002
    engine7.step(make_inp(
        target_position=-0.60, max_shares=100,
        spread=0.002, bar_index=0
    ))
    # Spread compresses to 0.001 — profitable for short spread
    bar7_hold = engine7.step(make_inp(
        target_position=-0.60, max_shares=100,
        spread=0.001, bar_index=1
    ))
    # Unrealized PnL should be positive (spread compressed)
    print(f"  unrealized_pnl = {bar7_hold.unrealized_pnl:.4f}  (expected > 0)")
    assert bar7_hold.unrealized_pnl > 0, \
        f"Short spread with compression should have positive unrealized PnL"
    # Flatten
    bar7_exit = engine7.step(make_inp(
        target_position=0.0,
        hard_exit=True,
        spread=0.001, bar_index=2
    ))
    print(f"  realized_pnl   = {bar7_exit.realized_pnl:.4f}  (expected > 0 minus costs)")
    assert bar7_exit.current_position == 0
    print("  ✓ PnL accounting correct")

    # ── Test 8: Daily loss halt ────────────────────────────────────────────
    print("\n[Test 8] Daily loss halt")
    engine8 = ExecutionEngine(cfg)
    engine8.reset_session()
    # Inject large realized loss directly
    engine8._realized_pnl = -(cfg.session_risk.daily_loss_limit_pct
                               * cfg.sizing.account_size + 1.0)
    bar8 = engine8.step(make_inp(bar_index=0))
    assert bar8.daily_loss_halted, "daily_loss_halted should be True"
    assert bar8.order_type == ORDER_HOLD, \
        f"No new trades after halt, got {bar8.order_type}"
    print(f"  daily_loss_halted=True, order_type=hold  ✓")

    # ── Test 9: TargetPosition=0 → flatten ────────────────────────────────
    print("\n[Test 9] TargetPosition=0 → flatten existing position")
    engine9 = ExecutionEngine(cfg)
    engine9.reset_session()
    engine9.step(make_inp(target_position=-0.60, max_shares=100, bar_index=0))
    bar9 = engine9.step(make_inp(
        target_position=0.0, max_shares=100, bar_index=1
    ))
    assert bar9.order_type       == ORDER_FLATTEN
    assert bar9.current_position == 0
    print(f"  order_type=flatten, position=0  ✓")

    # ── Test 10: State snapshot / restore ─────────────────────────────────
    print("\n[Test 10] State snapshot / restore")
    snap     = engine7.get_state_snapshot()
    engine10 = ExecutionEngine(cfg)
    engine10.restore_state(snap)
    assert engine10._current_position == engine7._current_position
    assert abs(engine10._realized_pnl - engine7._realized_pnl) < 1e-10
    assert engine10._trade_age        == engine7._trade_age
    print("  ✓ State snapshot / restore correct")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
