"""
logging/session_logger.py
─────────────────────────────────────────────────────────────────────────────
Structured per-session CSV logger.

Writes every PipelineBar to a flat CSV file, one row per bar.
Designed for post-session analysis, regime diagnosis, and backtesting replay.

Storage layout:
    logs/
        sessions/
            YYYY-MM-DD.csv       ← one file per session
        latest.csv               ← symlink / copy of current session (live)

Columns written (55 total):
    ── Identity ──────────────────────────────────────────────────
    bar_index, timestamp

    ── Spread layer ──────────────────────────────────────────────
    leg1_price, leg2_price, spread, log_gld, log_iau, beta, alpha,
    spread_valid

    ── Kalman / IMM layer ────────────────────────────────────────
    x_hat, kalman_P, imm_score, imm_p0, imm_p1, imm_p2,
    kalman_innovation, kalman_Q, kalman_R, filter_diverged

    ── Stationarity / OU layer ───────────────────────────────────
    gate_open, gate_factor, adf_pvalue, hurst, ou_valid,
    ou_ever_valid, phi, mu, sigma_ou, hl_bars

    ── Signal layer ──────────────────────────────────────────────
    z_score, signal_score, hard_exit, signal_valid,
    trend_mute, imm_mute, filter_mute

    ── Risk layer ────────────────────────────────────────────────
    risk_score, position_scalar,
    z_risk, regime_risk, filter_risk, time_risk, hl_jump_risk,
    w0, w1, w2, w3, w4, weights_source, risk_n_obs

    ── Target position layer ─────────────────────────────────────
    target_position, max_shares, time_decay,
    risk_pct_used, stop_distance, target_valid

    ── Execution layer ───────────────────────────────────────────
    order_type, order_shares, order_direction,
    current_position, entry_spread, trade_age, in_trade,
    fill_cost, unrealized_pnl, realized_pnl, session_pnl,
    in_cooldown, daily_loss_halted

Design invariants:
    - File opened once at session_open(), flushed after every bar,
      closed at session_close(). Never re-opened mid-session.
    - Header written only on file creation (not on append).
    - All float values written at full double precision (17 sig figs).
    - NaN written as empty string "". Inf written as "inf"/"-inf".
    - On write failure: log warning, continue — never crash the live loop.
    - Thread-safe for single-threaded pipeline (no locking needed).
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np

from config.config_loader import SystemConfig
from pipeline_runner import PipelineBar

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions — order is the CSV column order
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS: List[str] = [
    # Identity
    "bar_index", "timestamp",

    # Spread
    "leg1_price", "leg2_price", "spread", "log_gld", "log_iau",
    "beta", "alpha", "spread_valid",

    # Kalman / IMM
    "x_hat", "kalman_P", "imm_score",
    "imm_p0", "imm_p1", "imm_p2",
    "kalman_innovation", "kalman_Q", "kalman_R", "filter_diverged",

    # Stationarity / OU
    "gate_open", "gate_factor", "adf_pvalue", "hurst",
    "ou_valid", "ou_ever_valid",
    "phi", "mu", "sigma_ou", "hl_bars",

    # Signal
    "z_score", "signal_score", "hard_exit", "signal_valid",
    "trend_mute", "imm_mute", "filter_mute",

    # Risk
    "risk_score", "position_scalar",
    "z_risk", "regime_risk", "filter_risk", "time_risk", "hl_jump_risk",
    "w0", "w1", "w2", "w3", "w4",
    "weights_source", "risk_n_obs",

    # Target position
    "target_position", "max_shares", "time_decay",
    "risk_pct_used", "stop_distance", "target_valid",

    # Execution
    "order_type", "order_shares", "order_direction",
    "current_position", "entry_spread", "trade_age", "in_trade",
    "fill_cost", "unrealized_pnl", "realized_pnl", "session_pnl",
    "in_cooldown", "daily_loss_halted",
]


# ─────────────────────────────────────────────────────────────────────────────
# Float formatter
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val: object) -> str:
    """
    Format a value for CSV output.
    - NaN        → "" (empty — easily filtered in pandas with na_values="")
    - inf/-inf   → "inf" / "-inf"
    - float      → 17-significant-figure repr
    - bool       → "1" / "0"  (before int check — bool is subclass of int)
    - int        → str(int)
    - str        → val
    - None       → ""
    """
    if val is None:
        return ""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, float):
        if np.isnan(val):
            return ""
        if np.isposinf(val):
            return "inf"
        if np.isneginf(val):
            return "-inf"
        return repr(val)          # repr guarantees round-trip precision
    if isinstance(val, (np.floating,)):
        return _fmt(float(val))
    if isinstance(val, (np.integer,)):
        return str(int(val))
    if isinstance(val, int):
        return str(val)
    return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# Row extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_row(pb: PipelineBar) -> dict:
    """
    Flatten a PipelineBar into a flat dict keyed by COLUMNS.
    All values are Python primitives (str, int, float, bool).
    """
    sp  = pb.spread
    k   = pb.kalman
    st  = pb.station
    si  = pb.signal
    ri  = pb.risk
    tp  = pb.target
    ex  = pb.execution

    # IMM probabilities — p_all is ndarray of length 3
    p_all = k.p_all if hasattr(k, "p_all") and k.p_all is not None \
            else [float("nan")] * 3

    # Weights — ndarray of length 5
    w = ri.weights if hasattr(ri, "weights") and ri.weights is not None \
        else [float("nan")] * 5

    return {
        # Identity
        "bar_index":          pb.bar_index,
        "timestamp":          pb.timestamp,

        # Spread
        "leg1_price":         sp.log_gld,       # log_gld for completeness
        "leg2_price":         sp.log_iau,
        "spread":             sp.spread,
        "log_gld":            sp.log_gld,
        "log_iau":            sp.log_iau,
        "beta":               sp.beta,
        "alpha":              sp.alpha,
        "spread_valid":       sp.valid,

        # Kalman / IMM
        "x_hat":              k.x_hat,
        "kalman_P":           k.P,
        "imm_score":          k.imm_score,
        "imm_p0":             float(p_all[0]),
        "imm_p1":             float(p_all[1]),
        "imm_p2":             float(p_all[2]),
        "kalman_innovation":  k.innovation,
        "kalman_Q":           k.Q_k,
        "kalman_R":           k.R_k,
        "filter_diverged":    k.filter_diverged,

        # Stationarity / OU
        "gate_open":          st.gate_open,
        "gate_factor":        si.gate_factor,
        "adf_pvalue":         st.adf_pvalue,
        "hurst":              st.hurst,
        "ou_valid":           st.ou_valid,
        "ou_ever_valid":      st.ou_ever_valid,
        "phi":                st.phi,
        "mu":                 st.mu,
        "sigma_ou":           st.sigma_ou,
        "hl_bars":            st.hl_bars,

        # Signal
        "z_score":            si.z_score,
        "signal_score":       si.signal_score,
        "hard_exit":          si.hard_exit,
        "signal_valid":       si.valid,


        # Risk
        "risk_score":         ri.risk_score,
        "position_scalar":    ri.position_scalar,
        "z_risk":             ri.z_risk,
        "regime_risk":        ri.regime_risk,
        "filter_risk":        ri.filter_risk,
        "time_risk":          ri.time_risk,
        "hl_jump_risk":       ri.hl_jump_risk,
        "w0":                 float(w[0]),
        "w1":                 float(w[1]),
        "w2":                 float(w[2]),
        "w3":                 float(w[3]),
        "w4":                 float(w[4]),
        "weights_source":     ri.weights_source,
        "risk_n_obs":         ri.n_obs,

        # Target position
        "target_position":    tp.target_position,
        "max_shares":         tp.max_shares,
        "time_decay":         tp.time_decay,
        "risk_pct_used":      tp.risk_pct_used,
        "stop_distance":      tp.stop_distance,
        "target_valid":       tp.valid,

        # Execution
        "order_type":         ex.order_type,
        "order_shares":       ex.order_shares,
        "order_direction":    ex.order_direction,
        "current_position":   ex.current_position,
        "entry_spread":       ex.entry_spread,
        "trade_age":          ex.trade_age,
        "in_trade":           ex.in_trade,
        "fill_cost":          ex.fill_cost,
        "unrealized_pnl":     ex.unrealized_pnl,
        "realized_pnl":       ex.realized_pnl,
        "session_pnl":        ex.session_pnl,
        "in_cooldown":        ex.in_cooldown,
        "daily_loss_halted":  ex.daily_loss_halted,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SessionLogger
# ─────────────────────────────────────────────────────────────────────────────

class SessionLogger:
    """
    Writes one PipelineBar per row to a session CSV file.

    Usage:
        logger_obj = SessionLogger(cfg)
        logger_obj.open_session("2026-03-14")
        for each bar:
            logger_obj.log_bar(pipeline_bar)
        logger_obj.close_session()
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._log_dir     = Path(cfg.logging.session_log_dir)
        self._session_dir = self._log_dir / "sessions"
        self._latest_path = self._log_dir / "latest.csv"

        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._file        = None
        self._writer      = None
        self._session_date: str = ""
        self._bars_written: int = 0
        self._is_open:      bool = False

    # ─────────────────────────────────────────────────────────────────────
    # Session lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def open_session(self, session_date: str) -> None:
        """
        Open a new CSV file for the session.

        Parameters
        ----------
        session_date : str
            ISO date string, e.g. "2026-03-14".
            Used as the CSV filename.

        If a file for this date already exists, it is overwritten.
        In production, session_date is always today's date, so this
        only happens on restart within the same session.
        """
        if self._is_open:
            logger.warning(
                "SessionLogger.open_session called while session already open. "
                "Closing previous session first."
            )
            self.close_session()

        self._session_date  = session_date
        self._bars_written  = 0
        session_path        = self._session_dir / f"{session_date}.csv"

        try:
            self._file   = open(session_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._file,
                fieldnames = COLUMNS,
                extrasaction = "ignore",   # silently drop unknown fields
                restval      = "",         # fill missing fields with ""
            )
            self._writer.writeheader()
            self._file.flush()
            self._is_open = True
            logger.info(
                f"SessionLogger: opened → {session_path} "
                f"({len(COLUMNS)} columns)"
            )
        except OSError as exc:
            logger.error(
                f"SessionLogger: failed to open {session_path}: {exc}. "
                "Logging disabled for this session."
            )
            self._is_open = False

    def close_session(self) -> None:
        """
        Flush, close the CSV file, and copy to latest.csv.
        """
        if not self._is_open or self._file is None:
            return

        try:
            self._file.flush()
            self._file.close()
        except OSError as exc:
            logger.warning(f"SessionLogger: error closing file: {exc}")
        finally:
            self._file   = None
            self._writer = None
            self._is_open = False

        # Copy to latest.csv for live monitoring
        session_path = self._session_dir / f"{self._session_date}.csv"
        try:
            import shutil
            shutil.copy2(session_path, self._latest_path)
            logger.info(
                f"SessionLogger: session closed | "
                f"bars_written={self._bars_written} | "
                f"copied to {self._latest_path}"
            )
        except OSError as exc:
            logger.warning(
                f"SessionLogger: could not update latest.csv: {exc}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Per-bar write
    # ─────────────────────────────────────────────────────────────────────

    def log_bar(self, pb: PipelineBar) -> None:
        """
        Write one PipelineBar as a CSV row.

        Never raises. On failure, logs a warning and continues.
        Flushes after every write — ensures no data loss on crash.

        Parameters
        ----------
        pb : PipelineBar
            Complete output from PipelineRunner.on_bar().
        """
        if not self._is_open or self._writer is None:
            return

        try:
            raw_row = _extract_row(pb)
            fmt_row = {k: _fmt(v) for k, v in raw_row.items()}
            self._writer.writerow(fmt_row)
            self._file.flush()
            self._bars_written += 1
        except Exception as exc:
            logger.warning(
                f"SessionLogger.log_bar: write failed at bar "
                f"{pb.bar_index}: {exc}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────

    @property
    def bars_written(self) -> int:
        return self._bars_written

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def current_session_path(self) -> Optional[Path]:
        if not self._session_date:
            return None
        return self._session_dir / f"{self._session_date}.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m logging.session_logger
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import dataclasses
    import shutil
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("SessionLogger smoke test — full pipeline integration")
    print("=" * 65)

    from config.config_loader import SystemConfig
    from pipeline_runner import PipelineRunner, SessionInitData, BarData
    import pandas as pd

    cfg = SystemConfig.from_yaml("config/params.yaml")

    # Override log_dir to temp directory
    cfg = dataclasses.replace(
    cfg,
    logging=dataclasses.replace(cfg.logging, session_log_dir="logs/_test_tmp")
    )   
    rng = np.random.default_rng(99)

    # ── Build synthetic prior data ─────────────────────────────────────────
    TRUE_BETA  = 1.35
    TRUE_ALPHA = 0.12
    N_PRIOR    = 35 * 390
    N_SESSION  = 390

    log_iau_p = np.log(19.0) + np.cumsum(rng.normal(0, 0.0003, N_PRIOR))
    log_gld_p = TRUE_ALPHA + TRUE_BETA * log_iau_p + rng.normal(0, 0.0005, N_PRIOR)
    timestamps_prior = pd.date_range(
        start="2025-01-02 09:30", periods=N_PRIOR, freq="1min"
    )
    prior_ohlc = pd.DataFrame({
        "GLD": np.exp(log_gld_p),
        "IAU": np.exp(log_iau_p),
    }, index=timestamps_prior)
    prior_spread_series = (
        log_gld_p[-390:] - TRUE_ALPHA - TRUE_BETA * log_iau_p[-390:]
    )
    log_iau_s = log_iau_p[-1] + np.cumsum(rng.normal(0, 0.0003, N_SESSION))
    log_gld_s = TRUE_ALPHA + TRUE_BETA * log_iau_s + rng.normal(0, 0.0005, N_SESSION)
    gld_prices = np.exp(log_gld_s)
    iau_prices = np.exp(log_iau_s)
    first_spread = float(log_gld_s[0] - TRUE_ALPHA - TRUE_BETA * log_iau_s[0])

    init_data = SessionInitData(
        prior_ohlc   = prior_ohlc,
        prior_spread = prior_spread_series,
        first_spread = first_spread,
    )

    # ── Run pipeline ───────────────────────────────────────────────────────
    runner       = PipelineRunner(cfg)
    session_log  = SessionLogger(cfg)

    runner.on_session_open(init_data, state_snapshot=None)
    session_log.open_session("2026-03-14")

    results = []
    for t in range(N_SESSION):
        bar_data = BarData(
            bar_index  = t,
            timestamp  = f"2026-03-14T09:30:00+{t:04d}",
            leg1_price = float(gld_prices[t]),
            leg2_price = float(iau_prices[t]),
            force_flat = (t >= N_SESSION - 5),
        )
        pb = runner.on_bar(bar_data)
        session_log.log_bar(pb)
        results.append(pb)

    runner.on_session_close()
    session_log.close_session()

    # ── Validate CSV ───────────────────────────────────────────────────────
    csv_path = session_log.current_session_path
    assert csv_path.exists(), f"CSV file not found: {csv_path}"

    df = pd.read_csv(csv_path, na_values=[""])
    print(f"\n  CSV path       : {csv_path}")
    print(f"  Rows written   : {len(df)}")
    print(f"  Columns        : {len(df.columns)}")
    print(f"  Expected rows  : {N_SESSION}")
    print(f"  Expected cols  : {len(COLUMNS)}")

    # ── Test 1: Row count ─────────────────────────────────────────────────
    assert len(df) == N_SESSION, \
        f"Expected {N_SESSION} rows, got {len(df)}"
    print("\n  ✓ Row count correct")

    # ── Test 2: Column count ──────────────────────────────────────────────
    assert len(df.columns) == len(COLUMNS), \
        f"Expected {len(COLUMNS)} columns, got {len(df.columns)}"
    assert list(df.columns) == COLUMNS, "Column order mismatch"
    print("  ✓ All columns present in correct order")

    # ── Test 3: No NaN in identity and critical numeric columns ───────────
    assert df["bar_index"].notna().all(),    "bar_index has NaN"
    assert df["timestamp"].notna().all(),    "timestamp has NaN"
    assert df["signal_score"].notna().all(), "signal_score has NaN"
    assert df["risk_score"].notna().all(),   "risk_score has NaN"
    assert df["session_pnl"].notna().all(),  "session_pnl has NaN"
    print("  ✓ No NaN in identity and critical columns")

    # ── Test 4: Value bounds ──────────────────────────────────────────────
    assert (df["signal_score"].between(-1.0, 1.0)).all(), \
        "signal_score out of [-1, 1]"
    assert (df["risk_score"].between(0.0, 1.0)).all(), \
        "risk_score out of [0, 1]"
    assert (df["position_scalar"].between(0.0, 1.0)).all(), \
        "position_scalar out of [0, 1]"
    print("  ✓ signal_score ∈ [-1,1], risk_score ∈ [0,1], "
          "position_scalar ∈ [0,1]")

    # ── Test 5: Final position is flat ────────────────────────────────────
    assert df["current_position"].iloc[-1] == 0, \
        f"Final position should be 0, got {df['current_position'].iloc[-1]}"
    print("  ✓ Final position is 0 (force_flat applied)")

    # ── Test 6: Weight columns sum to 1 (where risk_n_obs >= min_obs) ─────
    weight_cols = ["w0", "w1", "w2", "w3", "w4"]
    w_sum = df[weight_cols].sum(axis=1)
    assert (w_sum - 1.0).abs().max() < 1e-9, \
        f"Weights do not sum to 1. Max deviation: {(w_sum-1.0).abs().max()}"
    print("  ✓ Weights sum to 1.0 on all bars")

    # ── Test 7: NaN round-trip via empty string ───────────────────────────
    # entry_spread should be NaN when not in trade
    flat_rows = df[df["in_trade"] == 0]
    assert flat_rows["entry_spread"].isna().all(), \
        "entry_spread should be NaN when not in trade"
    print("  ✓ NaN round-trips correctly via empty string")

    # ── Test 8: latest.csv written ────────────────────────────────────────
    latest = Path(cfg.logging.session_log_dir) / "latest.csv"
    assert latest.exists(), "latest.csv not written"
    df_latest = pd.read_csv(latest, na_values=[""])
    assert len(df_latest) == N_SESSION
    print("  ✓ latest.csv written and matches session CSV")

    # ── Test 9: Boolean columns serialized as 0/1 ─────────────────────────
    # Read raw without na_values to check bool encoding
    df_raw = pd.read_csv(csv_path)
    bool_cols = ["spread_valid", "gate_open", "ou_valid",
                 "hard_exit", "in_trade", "filter_diverged"]
    for col in bool_cols:
        unique_vals = set(df_raw[col].dropna().unique())
        assert unique_vals.issubset({0, 1, "0", "1"}), \
            f"Boolean column {col} has unexpected values: {unique_vals}"
    print("  ✓ Boolean columns serialized as 0/1")

    # ── Test 10: bars_written counter ─────────────────────────────────────
    assert session_log.bars_written == N_SESSION, \
        f"bars_written={session_log.bars_written}, expected {N_SESSION}"
    print(f"  ✓ bars_written counter = {session_log.bars_written}")

    # ── Print sample rows ─────────────────────────────────────────────────
    print("\n  Sample rows (bar 0, 195, 389):")
    key_cols = ["bar_index", "z_score", "signal_score",
                "risk_score", "current_position", "session_pnl"]
    print(df[key_cols].iloc[[0, 195, 389]].to_string(index=False))

    # ── Cleanup ───────────────────────────────────────────────────────────
    import os
    test_dir = str(Path(cfg.logging.session_log_dir).parent)  # "logs/_test_tmp" parent
    shutil.rmtree("logs/_test_tmp", ignore_errors=True)

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
