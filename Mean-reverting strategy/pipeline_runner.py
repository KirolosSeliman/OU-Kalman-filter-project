"""
pipeline_runner.py
─────────────────────────────────────────────────────────────────────────────
Top-level orchestrator. Wires all seven modules into a single per-bar call.

Call order per bar (strictly causal):
    1. SpreadEngine.compute_spread()         → spread, beta
    2. KalmanIMMEngine.step()                → x_hat, P, imm_score
    3. RiskScoreEngine.record_outcome()      ← fills previous bar's obs
    4. StationarityOUEngine.step()           → gate, phi, mu, sigma_ou, hl
    5. SignalScoreEngine.compute()           → signal_score, z, hard_exit
    6. RiskScoreEngine.compute()             → risk_score, position_scalar
    7. TargetPositionEngine.compute()        → target_position, max_shares
    8. ExecutionEngine.step()                → order, fill, pnl

Session lifecycle:
    on_session_open(init_data, state_snapshot)
    on_bar(bar_data) → PipelineBar
    on_session_close() → state_snapshot
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.config_loader import SystemConfig
from model.spread_engine          import SpreadEngine,          SpreadBar
from model.kalman_imm_engine      import KalmanIMMEngine,      KalmanIMMBar
from model.stationarity_ou_engine import StationarityOUEngine, StationarityOUBar
from signal.signal_score          import SignalScoreEngine,     SignalInput,          SignalBar
from risk.risk_score              import RiskScoreEngine,       RiskInput,            RiskBar
from risk.target_position         import TargetPositionEngine,  TargetPositionInput,  TargetPositionBar
from execution.execution_engine   import ExecutionEngine,       ExecutionInput,       ExecutionBar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Session initialization data (passed to on_session_open)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionInitData:
    """
    Prior-session data required to initialize SpreadEngine and KalmanIMMEngine.
    Must contain ONLY data from sessions BEFORE the current session.
    """
    prior_ohlc:      pd.DataFrame   # columns: [leg1, leg2], DatetimeIndex
    prior_spread:    np.ndarray     # spread values from prior session
    first_spread:    float          # first spread value of the new session
                                    # (bar 0 price, used for Kalman init)


# ─────────────────────────────────────────────────────────────────────────────
# Per-bar input
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BarData:
    """Raw market data for one bar. Provided by data feed."""
    bar_index:   int
    timestamp:   str
    leg1_price:  float   # GLD mid price
    leg2_price:  float   # IAU mid price
    force_flat:  bool    # True when bar_time >= force_flat_time


# ─────────────────────────────────────────────────────────────────────────────
# Per-bar output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineBar:
    """Complete record of one bar's computation and execution."""
    bar_index:        int
    timestamp:        str
    spread:           SpreadBar
    kalman:           KalmanIMMBar
    station:          StationarityOUBar
    signal:           SignalBar
    risk:             RiskBar
    target:           TargetPositionBar
    execution:        ExecutionBar
    # Convenience fields
    in_trade:         bool
    z_score:          float
    signal_score:     float
    risk_score:       float
    target_position:  float
    current_position: int
    session_pnl:      float


# ─────────────────────────────────────────────────────────────────────────────
# PipelineRunner
# ─────────────────────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Orchestrates all seven engines in strict causal order every bar.

    Usage:
        runner = PipelineRunner(cfg)
        runner.on_session_open(init_data, state_snapshot=None)
        for bar_data in session_bars:
            pb = runner.on_bar(bar_data)
        snap = runner.on_session_close()
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg     = cfg
        self._spread  = SpreadEngine(cfg)
        self._kalman  = KalmanIMMEngine(cfg)
        self._station = StationarityOUEngine(cfg)
        self._signal  = SignalScoreEngine(cfg)
        self._risk    = RiskScoreEngine(cfg)
        self._target  = TargetPositionEngine(cfg)
        self._exec    = ExecutionEngine(cfg)

        self._hl_entry:    float = float("nan")
        self._prev_spread: float = float("nan")
        self._bar_count:   int   = 0

    # ─────────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def on_session_open(
        self,
        init_data:       SessionInitData,
        state_snapshot:  Optional[dict] = None,
    ) -> None:
        """
        Call once at 09:30 ET before the first bar of the session.

        Parameters
        ----------
        init_data : SessionInitData
            Prior-session price data and spread series.
            SpreadEngine and KalmanIMMEngine are re-initialized from this
            every session — they do NOT carry intra-session state across
            the overnight gap.
        state_snapshot : dict or None
            Output of on_session_close() from the prior session.
            Pass None on the first ever session.
        """
        # ── SpreadEngine: re-initialize from prior OHLC data ─────────────
        # Beta is re-estimated fresh every session from prior sessions.
        self._spread.initialize_session(init_data.prior_ohlc)

        # ── KalmanIMMEngine: re-initialize from prior spread + first bar ──
        self._kalman.initialize_session(
            prior_spread       = init_data.prior_spread,
            sigma_spread_prior = self._spread.sigma_spread_prior or 1e-4,
            first_spread       = init_data.first_spread,
        )

        # ── StationarityOUEngine: clear buffer, preserve OU params ────────
        self._station.reset_session()

        # ── ExecutionEngine: reset to flat ────────────────────────────────
        self._exec.reset_session()

        # ── Restore persisted params from prior session ───────────────────
        if state_snapshot is not None:
            if "station" in state_snapshot:
                self._station.restore_state(state_snapshot["station"])
            if "risk" in state_snapshot:
                self._risk.restore_state(state_snapshot["risk"])
            # Never restore exec state — always start flat at session open
            self._hl_entry = state_snapshot.get("hl_entry", float("nan"))
        else:
            self._hl_entry = float("nan")

        self._prev_spread = float("nan")
        self._bar_count   = 0
        logger.info("PipelineRunner: session opened.")

    def on_session_close(self) -> dict:
        """
        Call after the last bar of the session.
        Returns state snapshot to persist to disk.
        """
        snap = {
            "spread":   self._spread.get_state_snapshot(),
            "kalman":   self._kalman.get_state_snapshot(),
            "station":  self._station.get_state_snapshot(),
            "risk":     self._risk.get_state_snapshot(),
            "exec":     self._exec.get_state_snapshot(),
            "hl_entry": self._hl_entry,
        }
        logger.info(
            f"PipelineRunner: session closed | "
            f"bars={self._bar_count} | hl_entry={self._hl_entry}"
        )
        return snap

    # ─────────────────────────────────────────────────────────────────────
    # Per-bar processing
    # ─────────────────────────────────────────────────────────────────────

    def on_bar(self, bar_data: BarData) -> PipelineBar:
        """Process one bar through all seven engines in causal order."""
        self._bar_count += 1

        # ── 1. SpreadEngine ───────────────────────────────────────────────
        spread_bar = self._spread.compute_spread(
            bar_data.leg1_price, bar_data.leg2_price
        )
        raw_spread = spread_bar.spread

        # ── 2. KalmanIMMEngine ────────────────────────────────────────────
        kalman_bar = self._kalman.step(
            raw_spread if np.isfinite(raw_spread) else 0.0
        )

        # ── 3. RiskScoreEngine.record_outcome ─────────────────────────────
        if np.isfinite(self._prev_spread) and np.isfinite(raw_spread):
            self._risk.record_outcome(abs(raw_spread - self._prev_spread))
        self._prev_spread = raw_spread

        # ── 4. StationarityOUEngine ───────────────────────────────────────
        station_bar = self._station.step(kalman_bar.x_hat)

        # ── 5. Retrieve current trade state ───────────────────────────────
        exec_snap  = self._exec.get_state_snapshot()
        in_trade   = exec_snap["current_position"] != 0
        trade_age  = exec_snap["trade_age"] if in_trade else 0
        hl_current = station_bar.hl_bars

        # HL_entry locking: capture at entry, release at flatten
        if in_trade and not np.isfinite(self._hl_entry):
            self._hl_entry = hl_current if np.isfinite(hl_current) else float("nan")
        elif not in_trade:
            self._hl_entry = float("nan")

        # ── 6. SignalScoreEngine ──────────────────────────────────────────
        signal_inp = SignalInput(
            x_hat         = kalman_bar.x_hat,
            P_k           = kalman_bar.P,           # field is .P not .P_k
            imm_score     = kalman_bar.imm_score,
            gate_open     = station_bar.gate_open,
            mu            = station_bar.mu,
            sigma_ou      = station_bar.sigma_ou,
            ou_valid      = station_bar.ou_valid,
            ou_ever_valid = station_bar.ou_ever_valid,
            spread        = raw_spread,
        )
        signal_bar = self._signal.compute(signal_inp)

        # ── 7. RiskScoreEngine.compute ────────────────────────────────────
        risk_inp = RiskInput(
            z_score    = signal_bar.z_score if np.isfinite(signal_bar.z_score) else 0.0,
            imm_score  = kalman_bar.imm_score,
            P_k        = kalman_bar.P,
            adf_pvalue = station_bar.adf_pvalue,
            in_trade   = in_trade,
            trade_age  = trade_age,
            hl_entry   = self._hl_entry,
            hl_current = hl_current,
        )
        risk_bar = self._risk.compute(risk_inp)

        # ── 8. TargetPositionEngine ───────────────────────────────────────
        sigma_ou_safe = (
            station_bar.sigma_ou
            if np.isfinite(station_bar.sigma_ou) and station_bar.sigma_ou > 0
            else 0.001
        )
        target_inp = TargetPositionInput(
            signal_score    = signal_bar.signal_score,
            position_scalar = risk_bar.position_scalar,
            sigma_ou        = sigma_ou_safe,
            hl_current      = hl_current,
            imm_score       = kalman_bar.imm_score,
            z_score         = signal_bar.z_score if np.isfinite(signal_bar.z_score) else 0.0,
            leg1_price      = bar_data.leg1_price,
            in_trade        = in_trade,
            trade_age       = trade_age,
            hl_entry        = self._hl_entry,
        )
        target_bar = self._target.compute(target_inp)

        # ── 9. ExecutionEngine ────────────────────────────────────────────
        exec_inp = ExecutionInput(
            target_position = target_bar.target_position,
            max_shares      = target_bar.max_shares,
            hard_exit       = signal_bar.hard_exit,
            signal_valid    = signal_bar.valid,
            spread          = raw_spread if np.isfinite(raw_spread) else 0.0,
            leg1_price      = bar_data.leg1_price,
            leg2_price      = bar_data.leg2_price,
            force_flat      = bar_data.force_flat,
            bar_index       = bar_data.bar_index,
        )
        exec_bar = self._exec.step(exec_inp)

        # ── 10. Assemble PipelineBar ──────────────────────────────────────
        return PipelineBar(
            bar_index        = bar_data.bar_index,
            timestamp        = bar_data.timestamp,
            spread           = spread_bar,
            kalman           = kalman_bar,
            station          = station_bar,
            signal           = signal_bar,
            risk             = risk_bar,
            target           = target_bar,
            execution        = exec_bar,
            in_trade         = exec_bar.in_trade,
            z_score          = signal_bar.z_score,
            signal_score     = signal_bar.signal_score,
            risk_score       = risk_bar.risk_score,
            target_position  = target_bar.target_position,
            current_position = exec_bar.current_position,
            session_pnl      = exec_bar.session_pnl,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m pipeline_runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("PipelineRunner smoke test — full end-to-end")
    print("=" * 65)

    cfg = SystemConfig.from_yaml("config/params.yaml")
    rng = np.random.default_rng(42)

    # ── Build synthetic prior data ────────────────────────────────────────
    # 35 prior sessions × 390 bars, true relationship:
    # log(GLD) = 0.12 + 1.35 * log(IAU) + noise
    TRUE_BETA  = 1.35
    TRUE_ALPHA = 0.12
    N_PRIOR    = 35 * 390
    N_SESSION  = 390
    PHI        = 0.85
    SIGMA_OU   = 0.0005

    log_iau_p = np.log(19.0) + np.cumsum(rng.normal(0, 0.0003, N_PRIOR))
    log_gld_p = TRUE_ALPHA + TRUE_BETA * log_iau_p + rng.normal(0, 0.0005, N_PRIOR)

    timestamps_prior = pd.date_range(
        start="2025-01-02 09:30", periods=N_PRIOR, freq="1min"
    )
    prior_ohlc = pd.DataFrame({
        "GLD": np.exp(log_gld_p),
        "IAU": np.exp(log_iau_p),
    }, index=timestamps_prior)

    # Build prior spread series (last 390 bars of prior data)
    prior_spread_series = (
        log_gld_p[-390:] - TRUE_ALPHA - TRUE_BETA * log_iau_p[-390:]
    )

    # ── Build synthetic current session ───────────────────────────────────
    # Continue IAU/GLD paths from where prior ended
    log_iau_s = log_iau_p[-1] + np.cumsum(rng.normal(0, 0.0003, N_SESSION))
    log_gld_s = TRUE_ALPHA + TRUE_BETA * log_iau_s + rng.normal(0, 0.0005, N_SESSION)
    gld_prices = np.exp(log_gld_s)
    iau_prices = np.exp(log_iau_s)

    # First spread of new session (for Kalman init)
    first_spread = float(
        log_gld_s[0] - TRUE_ALPHA - TRUE_BETA * log_iau_s[0]
    )

    # ── SessionInitData ────────────────────────────────────────────────────
    init_data = SessionInitData(
        prior_ohlc   = prior_ohlc,
        prior_spread = prior_spread_series,
        first_spread = first_spread,
    )

    # ── Run session ───────────────────────────────────────────────────────
    runner = PipelineRunner(cfg)
    runner.on_session_open(init_data, state_snapshot=None)

    results = []
    for t in range(N_SESSION):
        bar_data = BarData(
            bar_index  = t,
            timestamp  = f"2026-03-14T09:30:00+{t:04d}",
            leg1_price = float(gld_prices[t]),
            leg2_price = float(iau_prices[t]),
            force_flat = (t >= N_SESSION - 5),
        )
        results.append(runner.on_bar(bar_data))

    snap = runner.on_session_close()

    # ── Assertions ────────────────────────────────────────────────────────
    print(f"\n  Bars processed : {len(results)}")
    print(f"  Final position : {results[-1].current_position}")
    print(f"  Session PnL    : {results[-1].session_pnl:.4f}")
    print(f"  Bars in trade  : {sum(r.in_trade for r in results)}")
    print(f"  Bars gate open : {sum(r.signal.gate_factor == 1.0 for r in results)}")

    assert len(results) == N_SESSION
    assert results[-1].current_position == 0, \
        f"Should be flat after force_flat. Got {results[-1].current_position}"

    for i, r in enumerate(results):
        assert np.isfinite(r.signal_score),          f"signal_score NaN at bar {i}"
        assert np.isfinite(r.risk_score),            f"risk_score NaN at bar {i}"
        assert np.isfinite(r.session_pnl),           f"session_pnl NaN at bar {i}"
        assert -1.0 <= r.signal_score <= 1.0,        f"signal_score OOB at bar {i}"
        assert  0.0 <= r.risk_score   <= 1.0,        f"risk_score OOB at bar {i}"

    required_keys = {"spread", "kalman", "station", "risk", "exec", "hl_entry"}
    assert required_keys.issubset(snap.keys()), \
        f"Missing snapshot keys: {required_keys - snap.keys()}"

    print("\n  ✓ All 390 bars processed without error")
    print("  ✓ Final position is flat (force_flat applied)")
    print("  ✓ No NaN in signal_score, risk_score, session_pnl")
    print("  ✓ All values within declared bounds")
    print("  ✓ State snapshot complete")

    # ── Test 2: Session restore ────────────────────────────────────────────
    print("\n[Test 2] Second session inherits prior state")
    runner2 = PipelineRunner(cfg)

    # Build init data for session 2 (continue from session 1 endpoint)
    log_iau_s2 = log_iau_s[-1] + np.cumsum(rng.normal(0, 0.0003, 390))
    log_gld_s2 = TRUE_ALPHA + TRUE_BETA * log_iau_s2 + rng.normal(0, 0.0005, 390)
    prior_ohlc_2 = pd.DataFrame({
        "GLD": np.concatenate([prior_ohlc["GLD"].values[-390*5:], gld_prices]),
        "IAU": np.concatenate([prior_ohlc["IAU"].values[-390*5:], iau_prices]),
    }, index=pd.date_range("2026-03-13 09:30", periods=390*6, freq="1min"))

    init_data_2 = SessionInitData(
        prior_ohlc   = prior_ohlc_2,
        prior_spread = prior_spread_series,  # reuse — acceptable for smoke test
        first_spread = float(log_gld_s2[0] - TRUE_ALPHA - TRUE_BETA * log_iau_s2[0]),
    )
    runner2.on_session_open(init_data_2, state_snapshot=snap)
    pb2 = runner2.on_bar(BarData(
        bar_index  = 0,
        timestamp  = "2026-03-15T09:30:00+0000",
        leg1_price = float(np.exp(log_gld_s2[0])),
        leg2_price = float(np.exp(log_iau_s2[0])),
        force_flat = False,
    ))
    assert np.isfinite(pb2.signal_score), "signal_score not finite after restore"
    assert np.isfinite(pb2.risk_score),   "risk_score not finite after restore"
    print("  ✓ Second session starts correctly from restored state")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n  Sample bars (every 50):")
    print(f"  {'bar':>4} {'z':>7} {'signal':>8} {'risk':>7} "
          f"{'target':>8} {'pos':>6} {'pnl':>10}")
    for r in results[::50]:
        z = r.z_score if np.isfinite(r.z_score) else float("nan")
        print(f"  {r.bar_index:>4} {z:>7.3f} {r.signal_score:>8.4f} "
              f"{r.risk_score:>7.4f} {r.target_position:>8.4f} "
              f"{r.current_position:>6} {r.session_pnl:>10.4f}")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
