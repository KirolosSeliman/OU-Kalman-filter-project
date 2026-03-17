"""
analytics/backtest.py
─────────────────────────────────────────────────────────────────────────────
Walk-forward backtest for the GLD/IAU OU mean-reversion strategy.

Pipeline per session:
  SpreadEngine → KalmanIMMEngine → StationarityOUEngine
  → Layer1 SignalScore → Layer2 TargetPosition → PnL

Causality contract:
  Session i is initialized using ONLY sessions 0..i-1.
  No current-session data can contaminate initialization.

Usage:
  python -m analytics.backtest --start 2024-01-01 --end 2026-03-14

Output:
  analytics/backtest_results.csv  — per-session metrics
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# CRITICAL: Add current directory FIRST so signal_layer_1.py is found
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical  import StockHistoricalDataClient
from alpaca.data.requests import *
from alpaca.data.timeframe   import TimeFrame, TimeFrameUnit

from config.config_loader             import SystemConfig
from model.spread_engine              import SpreadEngine
from model.kalman_imm_engine          import KalmanIMMEngine
from model.stationarity_ou_engine     import StationarityOUEngine



from signal_layer_1 import (
    layer1_complete,
    layer2_target_position,
    build_entry_state,
    proper_spread_pnl,
    AdaptiveWeightEstimator,
)


logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
ET     = ZoneInfo("America/New_York")


# ─────────────────────────────────────────────────────────────────────────────
# Data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_bars(leg1: str, leg2: str, start: str, end: str) -> pd.DataFrame:
    """Fetch 1-minute bars for both legs from Alpaca. Returns wide DataFrame."""
    client = StockHistoricalDataClient(
        api_key    = os.getenv("ALPACA_API_KEY"),
        secret_key = os.getenv("ALPACA_SECRET_KEY"),
    )
    request = StockBarsRequest(
        symbol_or_symbols = [leg1, leg2],
        timeframe         = TimeFrame(1, TimeFrameUnit.Minute),
        start             = datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        end               = datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )
    logger.info(f"Fetching {leg1}/{leg2} bars {start} → {end} ...")
    bars = client.get_stock_bars(request).df
    if bars.empty:
        logger.error("No bars returned. Check API keys and date range.")
        sys.exit(1)

    bars  = bars.reset_index()
    df    = bars.pivot(index="timestamp", columns="symbol", values="close")
    df    = df[[leg1, leg2]].dropna()
    df.index = pd.to_datetime(df.index)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")

    df = df.between_time("09:30", "16:00").dropna()
    logger.info(f"Fetched {len(df):,} bars total.")
    return df


def split_sessions(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split into daily sessions, discarding days with fewer than 30 bars."""
    sessions = [
        group.sort_index()
        for _, group in df.groupby(df.index.date)
        if len(group) >= 30
    ]
    logger.info(f"Split into {len(sessions)} sessions.")
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Single-session pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_session(
    cfg:       SystemConfig,
    session:   pd.DataFrame,
    prior_df:  pd.DataFrame,
    estimator: AdaptiveWeightEstimator,
) -> dict | None:
    """
    Run one trading session through the full pipeline.

    Parameters
    ----------
    session   : current-session bars (never used for initialization)
    prior_df  : prior-session bars (used for SpreadEngine + Kalman init only)
    estimator : AdaptiveWeightEstimator, updated in-place across sessions

    Returns dict of session metrics, or None if initialization fails.
    """
    leg1 = cfg.spread.leg1
    leg2 = cfg.spread.leg2

    # ── 1. SpreadEngine initialization ───────────────────────────────────
    spread_engine = SpreadEngine(cfg)
    spread_engine.initialize_session(prior_df)
    if not spread_engine.is_initialized or not spread_engine.beta_valid:
        return None

    # ── 2. Build prior spread array for Kalman MLE ───────────────────────
    tail = prior_df.tail(cfg.spread.sigma_spread_window_sessions * 390)
    prior_spreads = []
    for _, row in tail.iterrows():
        sb = spread_engine.compute_spread(float(row[leg1]), float(row[leg2]))
        if sb.valid:
            prior_spreads.append(sb.spread)

    if len(prior_spreads) < cfg.kalman.mle_warmup_bars:
        return None

    # ── 3. KalmanIMMEngine initialization ────────────────────────────────
    first_row = session.iloc[0]
    first_sb  = spread_engine.compute_spread(
        float(first_row[leg1]), float(first_row[leg2])
    )
    if not first_sb.valid:
        return None

    kalman_engine = KalmanIMMEngine(cfg)
    kalman_engine.initialize_session(
        prior_spread       = np.array(prior_spreads),
        sigma_spread_prior = spread_engine.sigma_spread_prior,
        first_spread       = float(first_sb.spread),
    )

    # ── 4. StationarityOUEngine initialization ───────────────────────────
    stat_engine = StationarityOUEngine(cfg)
    stat_engine.reset_session()

    # ── 5. Bar loop: collect session arrays ──────────────────────────────
    N         = len(session)
    gld_arr   = session[leg1].values.astype(float)
    iau_arr   = session[leg2].values.astype(float)
    spreads   = np.full(N, np.nan)
    x_hats    = np.full(N, np.nan)
    Ps        = np.full(N, np.nan)
    p1s       = np.full(N, np.nan)
    z_scores  = np.full(N, np.nan)
    gate_open = np.zeros(N, dtype=bool)
    hl_bars   = np.full(N, np.nan)

    for t in range(N):
        sb = spread_engine.compute_spread(gld_arr[t], iau_arr[t])
        if not sb.valid:
            continue

        kb    = kalman_engine.step(sb.spread)
        statb = stat_engine.step(kb.x_hat)

        spreads[t]    = sb.spread
        x_hats[t]     = kb.x_hat
        Ps[t]         = kb.P
        p1s[t]        = kb.imm_score
        gate_open[t]  = statb.gate_open

        if statb.ou_ever_valid and np.isfinite(statb.hl_bars):
            hl_bars[t] = statb.hl_bars

        # z_score sign convention: positive → spread below Kalman mean → BUY signal
        # (consistent with signal_layer_1 raw_signal = +tanh(z / z_scale))
        p_safe       = max(float(kb.P), 1e-12)
        z_scores[t]  = (kb.x_hat - sb.spread) / np.sqrt(p_safe)

    if np.isfinite(spreads).sum() < 30:
        return None

    # ── 6. Layer 1: SignalScore ───────────────────────────────────────────
    hl_valid  = hl_bars[np.isfinite(hl_bars)]
    hl_median = float(np.median(hl_valid)) if len(hl_valid) > 0 else 20.0
    kappa     = np.log(2.0) / max(hl_median, 1.0)

    ou_params = {
        "kappa":         kappa,
        "half_life_min": hl_median,
        "sigma":         float(np.nanstd(spreads - x_hats)),
    }
    kalman_result = {
        "spread":  spreads,
        "x_hat":   x_hats,
        "P":       Ps,
        "p1":      p1s,
        "z_score": z_scores,
        "beta":    float(spread_engine.beta),
        "GLD":     gld_arr,
        "IAU":     iau_arr,
    }

    raw_signal, _ = layer1_complete(kalman_result, ou_params, p_max_factor=3.0, gate_open=True)

    # Apply stationarity gate and force-flat last 5 bars
    signal_scores          = raw_signal * gate_open.astype(float)
    signal_scores[-5:]     = 0.0

    # ── 7. Layer 2: TargetPosition ────────────────────────────────────────
    hl_entry_arr, trade_age_arr, _ = build_entry_state(
        signal_scores, hl_bars,
        entry_threshold = cfg.signal.entry_threshold,
    )

    target_pos, risk_scores, factors = layer2_target_position(
        signal_score   = signal_scores,
        kalman_result  = kalman_result,
        ou_params      = ou_params,
        mode           = cfg.sizing.mode,
        risk_pct_fixed = cfg.sizing.risk_pct_fixed,
        risk_pct_min   = cfg.sizing.risk_pct_min,
        risk_pct_max   = cfg.sizing.risk_pct_max,
        p_max_factor   = 3.0,
        hl_entry       = hl_entry_arr,
        trade_age      = trade_age_arr,
        weights        = estimator.weights,
    )

    # ── 8. PnL accounting ─────────────────────────────────────────────────
    # proper_spread_pnl returns bps for bars 1..N-1 (length N-1)
    beta        = float(spread_engine.beta)
    gross_bps   = proper_spread_pnl(kalman_result, target_pos, beta)

    # Transaction cost drag: tc_bps applied per unit of position change
    delta_pos   = np.abs(np.diff(np.concatenate([[0.0], target_pos])))
    tc_drag     = delta_pos[:-1] * cfg.execution.tc_bps   # length N-1
    net_bps     = gross_bps - tc_drag

    session_pnl = float(net_bps.sum())
    bar_sharpe  = (
        float(np.mean(net_bps) / np.std(net_bps) * np.sqrt(390))
        if np.std(net_bps) > 1e-12 else 0.0
    )

    in_trade_arr   = (hl_entry_arr > 0).astype(int)
    trades_entered = int(np.sum(np.diff(np.concatenate([[0], in_trade_arr])) == 1))
    bars_in_trade  = int(in_trade_arr.sum())

    # ── 9. Update adaptive weight estimator ──────────────────────────────
    for t in range(1, N):
        if np.isfinite(factors[t]).all() and np.isfinite(spreads[t]) and np.isfinite(spreads[t - 1]):
            estimator.record_outcome(factors[t], abs(spreads[t] - spreads[t - 1]))
    estimator.update_weights(regime_score=float(np.nanmean(p1s)))

    return {
        "bars":          N,
        "bars_in_trade": bars_in_trade,
        "trades_entered":trades_entered,
        "gate_open_pct": round(float(gate_open.mean()), 4),
        "session_pnl":   round(session_pnl, 4),
        "sharpe_intra":  round(bar_sharpe, 4),
        "beta":          round(beta, 6),
        "hl_median":     round(hl_median, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward loop
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    cfg:             SystemConfig,
    all_bars:        pd.DataFrame,
    warmup_sessions: int = 20,
) -> pd.DataFrame:

    sessions  = split_sessions(all_bars)
    results   = []
    estimator = AdaptiveWeightEstimator(
        window            = cfg.risk.adaptive_window,
        stability_min_obs = cfg.risk.stability_min_obs,
        dampening         = cfg.risk.dampening,
    )

    logger.info(
        f"Walk-forward backtest | sessions={len(sessions)} | warmup={warmup_sessions}"
    )

    for i, session in enumerate(sessions):
        session_date = str(session.index[0].date())

        if i < warmup_sessions:
            logger.info(f"[{i:>3}] {session_date} — warmup skip")
            continue

        prior_df = pd.concat(sessions[max(0, i - warmup_sessions): i])
        metrics  = run_session(cfg, session, prior_df, estimator)

        if metrics is None:
            logger.warning(f"[{i:>3}] {session_date} — init failed, skip")
            continue

        metrics["session"] = session_date
        results.append(metrics)

        logger.info(
            f"[{i:>3}] {session_date} | "
            f"gate={metrics['gate_open_pct']:.1%} | "
            f"trades={metrics['trades_entered']:>2} | "
            f"bars_in_trade={metrics['bars_in_trade']:>4} | "
            f"pnl={metrics['session_pnl']:>8.2f} bps | "
            f"sharpe={metrics['sharpe_intra']:>6.3f} | "
            f"beta={metrics['beta']:.4f}"
        )

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        cols       = ["session"] + [c for c in results_df.columns if c != "session"]
        results_df = results_df[cols]
        out_path   = Path("analytics") / "backtest_results.csv"
        results_df.to_csv(out_path, index=False)
        logger.info(f"Results saved → {out_path}")
        _print_summary(results_df)

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame) -> None:
    total_pnl  = df["session_pnl"].sum()
    avg_pnl    = df["session_pnl"].mean()
    std_pnl    = df["session_pnl"].std()
    win_rate   = (df["session_pnl"] > 0).mean()
    sharpe     = avg_pnl / std_pnl * np.sqrt(252) if std_pnl > 0 else 0.0
    cum_pnl    = df["session_pnl"].cumsum()
    max_dd     = (cum_pnl - cum_pnl.cummax()).min()
    avg_trades = df["trades_entered"].mean()
    avg_gate   = df["gate_open_pct"].mean()
    avg_hl     = df["hl_median"].mean()

    print("\n" + "=" * 65)
    print("BACKTEST SUMMARY")
    print("=" * 65)
    print(f"  Sessions traded    : {len(df)}")
    print(f"  Total PnL (bps)    : {total_pnl:>12.2f}")
    print(f"  Avg session PnL    : {avg_pnl:>12.4f} bps")
    print(f"  Std session PnL    : {std_pnl:>12.4f} bps")
    print(f"  Win rate           : {win_rate:>12.1%}")
    print(f"  Ann. Sharpe        : {sharpe:>12.3f}")
    print(f"  Max drawdown (bps) : {max_dd:>12.2f}")
    print(f"  Avg trades/session : {avg_trades:>12.2f}")
    print(f"  Avg gate open      : {avg_gate:>12.1%}")
    print(f"  Avg HL (bars)      : {avg_hl:>12.2f}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward backtest")
    parser.add_argument("--config",  default="config/params.yaml")
    parser.add_argument("--start",   default="2024-01-01",
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default="2026-03-14",
                        help="End date YYYY-MM-DD")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="Number of prior sessions for initialization")
    args = parser.parse_args()

    Path("analytics").mkdir(exist_ok=True)

    cfg      = SystemConfig.from_yaml(args.config)
    all_bars = fetch_all_bars(cfg.spread.leg1, cfg.spread.leg2, args.start, args.end)
    run_backtest(cfg, all_bars, warmup_sessions=args.warmup)
