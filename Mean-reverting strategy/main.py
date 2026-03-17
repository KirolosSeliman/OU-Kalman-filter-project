"""
main.py
─────────────────────────────────────────────────────────────────────────────
Live trading entry point.

Orchestrates the full session lifecycle:
    1. Load config
    2. Load prior state snapshot (StateManager)
    3. Fetch prior OHLC data (data feed)
    4. Initialize pipeline (PipelineRunner.on_session_open)
    5. Open session logger (SessionLogger.open_session)
    6. Wait for session open (09:30 ET)
    7. Process bars in real time (PipelineRunner.on_bar)
    8. Submit orders to broker (BrokerAdapter)
    9. Detect force_flat time (15:55 ET default)
   10. Close session (PipelineRunner.on_session_close)
   11. Save state (StateManager.save)
   12. Close logger (SessionLogger.close_session)

BrokerAdapter (stub):
    Defined here as a minimal interface. Replace the body of
    BrokerAdapter.submit_order() with your actual broker API calls
    (IB TWS / Alpaca / etc.). The rest of main.py never changes.

Data feed (stub):
    DataFeed.get_latest_bar() is a stub that returns the most recent
    1-minute bar. Replace with your actual market data source.

Session timing (all times US/Eastern):
    PRE_MARKET_FETCH  = 09:15  — fetch prior OHLC, initialize pipeline
    SESSION_OPEN      = 09:30  — first bar processed
    FORCE_FLAT_TIME   = 15:55  — all positions flattened
    SESSION_CLOSE     = 16:00  — logger and state closed

Design invariants:
    - main.py never imports numpy directly — all math is in pipeline modules
    - All exceptions caught in the bar loop — one bad bar never kills the session
    - KeyboardInterrupt caught cleanly — session closed and state saved on Ctrl+C
    - dry_run=True → orders logged but never submitted to broker
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config.config_loader    import SystemConfig
from pipeline_runner         import PipelineRunner, SessionInitData, BarData, PipelineBar
from state.state_manager     import StateManager
from trade_log.session_logger import SessionLogger
from execution.execution_engine import ORDER_FLATTEN, ORDER_BUY, ORDER_SELL, ORDER_HOLD

import os
from dotenv import load_dotenv
load_dotenv()

from alpaca.data.historical  import StockHistoricalDataClient
from alpaca.data.requests    import StockBarsRequest
from alpaca.data.timeframe   import TimeFrame, TimeFrameUnit
from alpaca.trading.client   import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums    import OrderSide, TimeInForce

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt   = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level   = level,
        format  = fmt,
        handlers= [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("main.log", mode="a", encoding="utf-8"),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Broker adapter — Alpaca
# ─────────────────────────────────────────────────────────────────────────────

class BrokerAdapter:
    def __init__(self, cfg: SystemConfig, dry_run: bool = True) -> None:
        self._cfg     = cfg
        self._dry_run = dry_run
        self._leg1    = cfg.spread.leg1
        self._leg2    = cfg.spread.leg2
        paper         = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        self._client  = TradingClient(
            api_key    = os.getenv("ALPACA_API_KEY"),
            secret_key = os.getenv("ALPACA_SECRET_KEY"),
            paper      = paper,
        )

    def submit_spread_order(self, pb: PipelineBar, tag: str = "") -> None:
        ex = pb.execution
        if ex.order_type == ORDER_HOLD or ex.order_shares == 0:
            return
        direction = ex.order_direction
        shares    = ex.order_shares
        leg1_side = OrderSide.BUY  if direction > 0 else OrderSide.SELL
        leg2_side = OrderSide.SELL if direction > 0 else OrderSide.BUY
        self._submit(self._leg1, leg1_side, shares, tag)
        self._submit(self._leg2, leg2_side, shares, tag)

    def _submit(self, symbol: str, side: OrderSide, shares: int, tag: str) -> None:
        if self._dry_run:
            logger.info(
                f"[DRY RUN] ORDER | {symbol} {side.value.upper()} "
                f"{shares} shares | tag={tag}"
            )
            return
        try:
            order = self._client.submit_order(
                MarketOrderRequest(
                    symbol        = symbol,
                    qty           = shares,
                    side          = side,
                    time_in_force = TimeInForce.DAY,
                )
            )
            logger.info(
                f"ORDER SUBMITTED | {symbol} {side.value.upper()} "
                f"{shares} shares | id={order.id} | tag={tag}"
            )
        except Exception as exc:
            logger.error(
                f"BrokerAdapter: order failed | {symbol} "
                f"{side.value.upper()} {shares} shares | {exc}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Data feed (stub — replace with real market data source)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Data feed — Alpaca
# ─────────────────────────────────────────────────────────────────────────────

class DataFeed:
    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg         = cfg
        self._leg1        = cfg.spread.leg1
        self._leg2        = cfg.spread.leg2
        self._hist_client = StockHistoricalDataClient(
            api_key    = os.getenv("ALPACA_API_KEY"),
            secret_key = os.getenv("ALPACA_SECRET_KEY"),
        )

    def get_latest_bar(self) -> Optional[tuple[float, float]]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols = [self._leg1, self._leg2],
                timeframe         = TimeFrame(1, TimeFrameUnit.Minute),
                limit             = 1,
            )
            bars = self._hist_client.get_stock_bars(request).df.reset_index()
            leg1_bar = bars[bars["symbol"] == self._leg1].iloc[-1]
            leg2_bar = bars[bars["symbol"] == self._leg2].iloc[-1]
            return float(leg1_bar["close"]), float(leg2_bar["close"])
        except Exception as exc:
            logger.error(f"DataFeed.get_latest_bar failed: {exc}")
            return None

    def fetch_prior_ohlc(self, n_sessions: int = 20) -> pd.DataFrame:
        from datetime import timedelta, timezone
        calendar_days = int(n_sessions * 1.5) + 10
        start_dt = datetime.now(tz=timezone.utc) - timedelta(days=calendar_days)
        try:
            request = StockBarsRequest(
                symbol_or_symbols = [self._leg1, self._leg2],
                timeframe         = TimeFrame(1, TimeFrameUnit.Minute),
                start             = start_dt,
            )
            bars = self._hist_client.get_stock_bars(request).df.reset_index()
            df   = bars.pivot(index="timestamp", columns="symbol", values="close")
            df   = df[[self._leg1, self._leg2]].dropna()
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
            else:
                df.index = df.index.tz_convert("America/New_York")
            df = df.between_time("09:30", "16:00").dropna()
            logger.info(
                f"DataFeed: fetched {len(df)} bars of prior OHLC "
                f"({self._leg1}/{self._leg2}) via Alpaca"
            )
            return df
        except Exception as exc:
            logger.error(f"DataFeed.fetch_prior_ohlc failed: {exc}")
            return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# Session timing helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_et() -> datetime:
    return datetime.now(tz=ET)


def wait_until(target: datetime) -> None:
    """Block until target datetime (ET). Logs every 60 seconds."""
    while True:
        remaining = (target - now_et()).total_seconds()
        if remaining <= 0:
            return
        if remaining > 60:
            logger.info(
                f"Waiting for {target.strftime('%H:%M:%S ET')} — "
                f"{remaining/60:.1f} min remaining"
            )
            time.sleep(60)
        else:
            time.sleep(max(0.1, remaining - 0.05))
            return


def parse_time_et(time_str: str, session_date: date) -> datetime:
    """Parse 'HH:MM' string into a timezone-aware ET datetime for session_date."""
    h, m = map(int, time_str.split(":"))
    return datetime(
        session_date.year, session_date.month, session_date.day,
        h, m, 0, tzinfo=ET
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main session loop
# ─────────────────────────────────────────────────────────────────────────────

def run_session(cfg: SystemConfig, dry_run: bool = True) -> None:
    """
    Run one complete trading session.

    Parameters
    ----------
    cfg     : SystemConfig
    dry_run : bool
        True  → orders logged but never submitted.
        False → live order submission via BrokerAdapter._submit().
    """
    session_date     = now_et().date()
    session_date_str = session_date.strftime("%Y-%m-%d")

    logger.info(f"{'='*65}")
    logger.info(f"SESSION START | {session_date_str} | dry_run={dry_run}")
    logger.info(f"{'='*65}")

    # ── 1. Instantiate components ─────────────────────────────────────────
    state_mgr    = StateManager(cfg)
    session_log  = SessionLogger(cfg)
    data_feed    = DataFeed(cfg)
    broker       = BrokerAdapter(cfg, dry_run=dry_run)
    runner       = PipelineRunner(cfg)

    # ── 2. Pre-market: fetch prior data ──────────────────────────────────
    pre_market_time = parse_time_et("09:15", session_date)
    session_open    = parse_time_et(cfg.session.open_time,  session_date)
    force_flat_time = parse_time_et(cfg.session.force_flat_time, session_date)
    session_close   = parse_time_et(cfg.session.close_time, session_date)

    logger.info(f"Waiting for pre-market fetch at 09:15 ET ...")
    wait_until(pre_market_time)
    logger.info("Fetching prior OHLC data ...")

    prior_ohlc = data_feed.fetch_prior_ohlc(n_sessions=cfg.spread.beta_window_sessions)

    if prior_ohlc.empty:
        logger.error(
            "DataFeed returned empty prior_ohlc. Cannot initialize SpreadEngine. "
            "Aborting session."
        )
        return

    # Build prior spread series for Kalman init
    # This requires SpreadEngine to be initialized first with prior data
    # so we can compute the spread on historical bars.
    # Use a temporary SpreadEngine instance just for this.
    from model.spread_engine import SpreadEngine as _SE
    _tmp_spread = _SE(cfg)
    _tmp_spread.initialize_session(prior_ohlc)

    if not _tmp_spread.is_initialized or _tmp_spread.beta is None:
        logger.error("SpreadEngine initialization failed on prior data. Aborting.")
        return

    leg1_col = cfg.spread.leg1
    leg2_col = cfg.spread.leg2
    prior_spreads = []
    for _, row in prior_ohlc.tail(cfg.spread.sigma_spread_window_sessions * 390).iterrows():
        sb = _tmp_spread.compute_spread(row[leg1_col], row[leg2_col])
        if sb.valid:
            prior_spreads.append(sb.spread)

    if len(prior_spreads) < cfg.kalman.mle_warmup_bars:
        logger.error(
            f"Insufficient prior spread bars ({len(prior_spreads)}) "
            f"for Kalman MLE init. Aborting."
        )
        return

    prior_spread_arr = np.array(prior_spreads)

    # ── 3. Load prior state ───────────────────────────────────────────────
    state_snapshot = state_mgr.load_latest()
    if state_snapshot is None:
        logger.info("No prior state found — cold start.")
    else:
        logger.info("Prior state loaded successfully.")

    # ── 4. Initialize pipeline ────────────────────────────────────────────
    # First spread: use last row of prior_ohlc as proxy for bar 0
    last_row = prior_ohlc.iloc[-1]
    first_sb = _tmp_spread.compute_spread(
        last_row[leg1_col], last_row[leg2_col]
    )
    first_spread = first_sb.spread if first_sb.valid else float(prior_spreads[-1])

    init_data = SessionInitData(
        prior_ohlc   = prior_ohlc,
        prior_spread = prior_spread_arr,
        first_spread = first_spread,
    )
    runner.on_session_open(init_data, state_snapshot=state_snapshot)

    # ── 5. Open session logger ────────────────────────────────────────────
    session_log.open_session(session_date_str)

    # ── 6. Wait for session open ──────────────────────────────────────────
    logger.info(f"Waiting for session open at {cfg.session.open_time} ET ...")
    wait_until(session_open)
    logger.info("Session open. Starting bar loop.")

    # ── 7. Bar loop ───────────────────────────────────────────────────────
    bar_index   = 0
    last_pb: Optional[PipelineBar] = None

    try:
        while True:
            bar_time = now_et()

            # Session close check
            if bar_time >= session_close:
                logger.info("Session close time reached. Exiting bar loop.")
                break

            # Determine force_flat
            force_flat = bar_time >= force_flat_time

            # Fetch current bar prices
            prices = data_feed.get_latest_bar()
            if prices is None:
                logger.warning(
                    f"Bar {bar_index}: DataFeed returned None. "
                    "Skipping bar."
                )
                # Wait for next minute
                time.sleep(60)
                continue

            leg1_price, leg2_price = prices

            # Build BarData
            bar_data = BarData(
                bar_index  = bar_index,
                timestamp  = bar_time.isoformat(),
                leg1_price = leg1_price,
                leg2_price = leg2_price,
                force_flat = force_flat,
            )

            # Process bar through pipeline
            try:
                pb = runner.on_bar(bar_data)
            except Exception as exc:
                logger.error(
                    f"Bar {bar_index}: pipeline error: {exc}. "
                    "Skipping bar — position unchanged.",
                    exc_info=True,
                )
                time.sleep(60)
                bar_index += 1
                continue

            # Log bar
            session_log.log_bar(pb)
            last_pb = pb

            # Determine order tag
            if pb.execution.order_type == ORDER_FLATTEN:
                tag = "force_flat" if force_flat else "exit"
            elif pb.execution.order_type in (ORDER_BUY, ORDER_SELL):
                tag = "entry" if not pb.execution.in_trade else "adjust"
            else:
                tag = ""

            # Submit order
            if pb.execution.order_type != ORDER_HOLD:
                broker.submit_spread_order(pb, tag=tag)

            # Per-bar log summary
            logger.info(
                f"Bar {bar_index:>4} | "
                f"z={pb.z_score:+.3f} | "
                f"sig={pb.signal_score:+.4f} | "
                f"risk={pb.risk_score:.3f} | "
                f"pos={pb.current_position:>5} | "
                f"pnl={pb.session_pnl:>8.2f} | "
                f"order={pb.execution.order_type}"
            )

            # Daily loss halt check
            if pb.execution.daily_loss_halted:
                logger.warning(
                    "Daily loss limit hit. No further entries this session."
                )

            bar_index += 1

            # Sleep until next bar
            # Align to next minute boundary
            sleep_secs = 60 - now_et().second - now_et().microsecond / 1e6
            time.sleep(max(1.0, sleep_secs))

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Closing session gracefully.")

    # ── 8. Session close ──────────────────────────────────────────────────
    logger.info(f"Session close: bars_processed={bar_index}")

    # Save state
    snap = runner.on_session_close()
    state_mgr.save(snap, session_date=session_date_str)
    state_mgr.purge_old_archives(retain_days=30)

    # Close logger
    session_log.close_session()

    # Final summary
    if last_pb is not None:
        logger.info(
            f"{'='*65}\n"
            f"SESSION COMPLETE | {session_date_str}\n"
            f"  Bars processed  : {bar_index}\n"
            f"  Final position  : {last_pb.current_position}\n"
            f"  Session PnL     : {last_pb.session_pnl:.2f}\n"
            f"  Realized PnL    : {last_pb.execution.realized_pnl:.2f}\n"
            f"  Loss halted     : {last_pb.execution.daily_loss_halted}\n"
            f"{'='*65}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GLD/IAU Mean-Reversion Live Trading System"
    )
    parser.add_argument(
        "--config",
        type    = str,
        default = "config/params.yaml",
        help    = "Path to params.yaml (default: config/params.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        default = True,
        help    = "Log orders without submitting to broker (default: True)",
    )
    parser.add_argument(
        "--live",
        action  = "store_true",
        default = False,
        help    = "Submit real orders to broker. Overrides --dry-run.",
    )
    parser.add_argument(
        "--log-level",
        type    = str,
        default = "INFO",
        choices = ["DEBUG", "INFO", "WARNING", "ERROR"],
        help    = "Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args     = parse_args()
    dry_run  = not args.live   # --live sets dry_run=False

    setup_logging(args.log_level)

    logger.info(f"Loading config from {args.config}")
    cfg = SystemConfig.from_yaml(args.config)

    logger.info(
        f"System config loaded | "
        f"account_size={cfg.sizing.account_size:,.0f} | "
        f"mode={cfg.sizing.mode} | "
        f"dry_run={dry_run}"
    )

    if dry_run:
        logger.info("DRY RUN MODE — no orders will be submitted to broker.")
    else:
        logger.warning(
            "LIVE MODE — real orders WILL be submitted. "
            "Ensure broker connection is active."
        )

    run_session(cfg, dry_run=dry_run)
