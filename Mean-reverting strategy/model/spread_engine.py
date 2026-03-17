"""
model/spread_engine.py
─────────────────────────────────────────────────────────────────────────────
Causal spread construction for the GLD/IAU cointegrated pair.

Responsibilities:
  1. Estimate hedge ratio (beta, alpha) via OLS on strictly prior-session data
  2. Freeze beta for the entire current session — no intra-session update
  3. Compute log-price spread: S_k = log(GLD_k) - alpha - beta * log(IAU_k)
  4. Compute sigma_spread_prior used by IMM engine for Q/R scaling
  5. Expose state snapshot / restore for StateManager

Causality guarantee:
  initialize_session() receives only prior-session data.
  After that call, beta and alpha are immutable for the session duration.
  compute_spread() uses only current bar prices and the frozen params.
  No current-session information can ever contaminate the hedge ratio.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpreadBar:
    """All outputs of SpreadEngine.compute_spread() for a single bar."""
    spread:  float    # S_k = log(GLD_k) - alpha - beta * log(IAU_k)
    log_gld: float    # log(GLD_k)
    log_iau: float    # log(IAU_k)
    beta:    float    # frozen hedge ratio for this session
    alpha:   float    # frozen intercept for this session
    valid:   bool     # False if price invalid, beta uninitialized, or beta out of bounds


# ─────────────────────────────────────────────────────────────────────────────
# SpreadEngine
# ─────────────────────────────────────────────────────────────────────────────

class SpreadEngine:
    """
    Causal hedge ratio estimator and per-bar spread calculator.

    Correct usage pattern — enforced via runtime guards:

        engine = SpreadEngine(cfg)
        engine.initialize_session(prior_data)   # once, before session open
        for each bar k:
            bar = engine.compute_spread(gld_k, iau_k)
            if bar.valid:
                # use bar.spread downstream
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg              = cfg
        self._beta:             Optional[float] = None
        self._alpha:            Optional[float] = None
        self._sigma_spread_prior: Optional[float] = None
        self._beta_valid:       bool = False
        self._initialized:      bool = False
        self._n_sessions_used:  int  = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Session initialization
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_session(self, prior_data: pd.DataFrame) -> None:
        """
        Estimate beta and alpha from prior-session data. Freeze for session.

        Parameters
        ----------
        prior_data : pd.DataFrame
            Columns must include cfg.spread.leg1 and cfg.spread.leg2
            (default: 'GLD', 'IAU'). Index must be a DatetimeIndex.
            Must contain ONLY data from prior sessions.
            The caller is responsible for this constraint.
            This method trusts its input unconditionally.

        Side effects
        ------------
        Sets self._beta, self._alpha, self._sigma_spread_prior,
        self._beta_valid, self._initialized.
        After this call, beta is immutable until the next
        initialize_session() call (i.e. next session open).
        """
        # Reset state before attempting estimation
        self._initialized       = False
        self._beta_valid        = False
        self._beta              = None
        self._alpha             = None
        self._sigma_spread_prior = None
        self._n_sessions_used   = 0

        leg1 = self._cfg.spread.leg1
        leg2 = self._cfg.spread.leg2

        # ── Step 1: Input validation ──────────────────────────────────────────
        if prior_data is None or prior_data.empty:
            logger.warning(
                "SpreadEngine.initialize_session: prior_data is empty. "
                "Beta cannot be estimated. All bars will return valid=False."
            )
            self._initialized = True
            return

        if leg1 not in prior_data.columns or leg2 not in prior_data.columns:
            logger.error(
                f"SpreadEngine.initialize_session: prior_data missing required "
                f"columns '{leg1}' or '{leg2}'. Got: {list(prior_data.columns)}"
            )
            self._initialized = True
            return

        if not isinstance(prior_data.index, pd.DatetimeIndex):
            logger.error(
                "SpreadEngine.initialize_session: prior_data must have a "
                "DatetimeIndex. Cannot identify session boundaries."
            )
            self._initialized = True
            return

        df = prior_data[[leg1, leg2]].dropna().copy()
        if len(df) < 30:
            logger.warning(
                f"SpreadEngine.initialize_session: only {len(df)} valid rows "
                "after dropping NaN. Need >= 30. Trading disabled."
            )
            self._initialized = True
            return

        # ── Step 2: Identify sessions, select last N ──────────────────────────
        df["_date"] = df.index.date
        sessions_available = df["_date"].nunique()

        if sessions_available < 2:
            logger.warning(
                "SpreadEngine.initialize_session: fewer than 2 prior sessions. "
                "Cannot estimate beta reliably. Trading disabled."
            )
            self._initialized = True
            return

        n_ols   = min(self._cfg.spread.beta_window_sessions, sessions_available)
        n_sigma = min(self._cfg.spread.sigma_spread_window_sessions, sessions_available)

        all_dates    = sorted(df["_date"].unique())
        ols_dates    = set(all_dates[-n_ols:])
        sigma_dates  = set(all_dates[-n_sigma:])

        df_ols = df[df["_date"].isin(ols_dates)].drop(columns="_date")
        self._n_sessions_used = n_ols

        # ── Step 3: Log-price OLS ─────────────────────────────────────────────
        # Model: log(GLD) = alpha + beta * log(IAU) + epsilon
        # Estimated by ordinary least squares.
        gld_vals = df_ols[leg1].values.astype(float)
        iau_vals = df_ols[leg2].values.astype(float)

        if np.any(gld_vals <= 0) or np.any(iau_vals <= 0):
            n_bad = np.sum(gld_vals <= 0) + np.sum(iau_vals <= 0)
            logger.warning(
                f"SpreadEngine: {n_bad} non-positive prices found in OLS window. "
                "Dropping affected rows."
            )
            mask     = (gld_vals > 0) & (iau_vals > 0)
            gld_vals = gld_vals[mask]
            iau_vals = iau_vals[mask]

        if len(gld_vals) < 30:
            logger.warning(
                "SpreadEngine: insufficient valid bars after filtering. "
                "Trading disabled."
            )
            self._initialized = True
            return

        log_gld = np.log(gld_vals)
        log_iau = np.log(iau_vals)

        # Design matrix: [intercept, log(IAU)]
        X      = np.column_stack([np.ones(len(log_iau)), log_iau])
        coeffs = np.linalg.lstsq(X, log_gld, rcond=None)[0]

        alpha_hat = float(coeffs[0])
        beta_hat  = float(coeffs[1])

        # ── Step 4: Sanity bounds on beta ─────────────────────────────────────
        beta_min = self._cfg.spread.beta_min
        beta_max = self._cfg.spread.beta_max
        beta_in_bounds = (beta_min <= beta_hat <= beta_max)

        if not beta_in_bounds:
            logger.warning(
                f"SpreadEngine: beta={beta_hat:.4f} is outside sanity bounds "
                f"[{beta_min}, {beta_max}]. This is a warning, not a halt. "
                "Review whether GLD/IAU remain cointegrated."
            )

        # ── Step 5: sigma_spread_prior ────────────────────────────────────────
        # Computed from the last sigma_sessions sessions using the OLS params
        # just estimated. This is the in-sample spread residual std.
        # Used by KalmanIMMEngine to scale its Q/R parameters.
        # Must be computed here, before session open, from prior data only.
        df_sigma  = df[df["_date"].isin(sigma_dates)].drop(columns="_date")
        log_gld_s = np.log(np.maximum(df_sigma[leg1].values.astype(float), 1e-12))
        log_iau_s = np.log(np.maximum(df_sigma[leg2].values.astype(float), 1e-12))
        spread_s  = log_gld_s - alpha_hat - beta_hat * log_iau_s
        spread_s  = spread_s[np.isfinite(spread_s)]

        if len(spread_s) < 10:
            sigma_spread = 1e-4
            logger.warning(
                "SpreadEngine: insufficient data for sigma_spread_prior. "
                "Using emergency fallback 1e-4. IMM Q/R scaling will be inaccurate."
            )
        else:
            sigma_spread = float(np.std(spread_s, ddof=1))
            if sigma_spread < 1e-8:
                logger.warning(
                    "SpreadEngine: sigma_spread_prior is effectively zero. "
                    "Possible data issue or perfect cointegration artifact."
                )
                sigma_spread = 1e-4

        # ── Step 6: Commit all estimated values ───────────────────────────────
        self._alpha              = alpha_hat
        self._beta               = beta_hat
        self._sigma_spread_prior = sigma_spread
        self._beta_valid         = beta_in_bounds
        self._initialized        = True

        logger.info(
            f"SpreadEngine initialized | "
            f"sessions_used={n_ols} | "
            f"beta={self._beta:.5f} | "
            f"alpha={self._alpha:.5f} | "
            f"sigma_spread_prior={self._sigma_spread_prior:.6f} | "
            f"beta_valid={self._beta_valid}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Per-bar computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_spread(self, gld_price: float, iau_price: float) -> SpreadBar:
        """
        Compute spread for bar k.

        Parameters
        ----------
        gld_price : float — GLD close at bar k
        iau_price : float — IAU close at bar k

        Returns
        -------
        SpreadBar
          valid=False when:
            - initialize_session() has not been called
            - beta could not be estimated (insufficient prior data)
            - either price is non-positive, NaN, or infinite
          valid=True and beta_valid=False when:
            - beta is outside sanity bounds (trade at your own risk;
              the pipeline will suppress entries via this flag)
        """
        _nan = float("nan")

        if not self._initialized:
            logger.error(
                "SpreadEngine.compute_spread called before initialize_session. "
                "This is a pipeline sequencing error."
            )
            return SpreadBar(_nan, _nan, _nan, _nan, _nan, valid=False)

        if self._beta is None or self._alpha is None:
            return SpreadBar(_nan, _nan, _nan, _nan, _nan, valid=False)

        # Price sanity
        gld_ok = np.isfinite(gld_price) and gld_price > 0.0
        iau_ok = np.isfinite(iau_price) and iau_price > 0.0

        if not gld_ok:
            logger.warning(f"SpreadEngine.compute_spread: invalid GLD price={gld_price}")
            return SpreadBar(_nan, _nan, _nan, self._beta, self._alpha, valid=False)

        if not iau_ok:
            logger.warning(f"SpreadEngine.compute_spread: invalid IAU price={iau_price}")
            return SpreadBar(_nan, _nan, _nan, self._beta, self._alpha, valid=False)

        log_gld = float(np.log(gld_price))
        log_iau = float(np.log(iau_price))
        spread  = log_gld - self._alpha - self._beta * log_iau

        return SpreadBar(
            spread  = spread,
            log_gld = log_gld,
            log_iau = log_iau,
            beta    = self._beta,
            alpha   = self._alpha,
            valid   = self._beta_valid and np.isfinite(spread),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def beta(self) -> Optional[float]:
        return self._beta

    @property
    def alpha(self) -> Optional[float]:
        return self._alpha

    @property
    def sigma_spread_prior(self) -> Optional[float]:
        return self._sigma_spread_prior

    @property
    def beta_valid(self) -> bool:
        return self._beta_valid

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        """
        Serializable state snapshot for StateManager.
        Call at session close.
        """
        return {
            "beta":                 self._beta,
            "alpha":                self._alpha,
            "sigma_spread_prior":   self._sigma_spread_prior,
            "beta_valid":           self._beta_valid,
            "initialized":          self._initialized,
            "n_sessions_used":      self._n_sessions_used,
        }

    def restore_state(self, snapshot: dict) -> None:
        """
        Restore from a snapshot produced by get_state_snapshot().
        Used after a crash-recovery within an active session.
        In normal operation, initialize_session() is called instead.
        """
        self._beta               = snapshot.get("beta")
        self._alpha              = snapshot.get("alpha")
        self._sigma_spread_prior = snapshot.get("sigma_spread_prior")
        self._beta_valid         = snapshot.get("beta_valid", False)
        self._initialized        = snapshot.get("initialized", False)
        self._n_sessions_used    = snapshot.get("n_sessions_used", 0)
        logger.info(
            f"SpreadEngine state restored | "
            f"beta={self._beta} | beta_valid={self._beta_valid}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m model.spread_engine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 60)
    print("SpreadEngine smoke test")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = SystemConfig.from_yaml("config/params.yaml")
    print(f"Config loaded | leg1={cfg.spread.leg1} | leg2={cfg.spread.leg2}")

    # ── Build synthetic prior-session data ────────────────────────────────────
    # Simulate 35 sessions * 390 bars of GLD/IAU prices.
    # True relationship: log(GLD) = 0.12 + 1.35 * log(IAU) + noise
    TRUE_BETA  = 1.35
    TRUE_ALPHA = 0.12
    rng        = np.random.default_rng(42)
    n_sessions = 35
    bars_per   = 390
    n_total    = n_sessions * bars_per

    # IAU starts at ~40, GLD starts at ~190
    log_iau_base = np.log(40.0) + np.cumsum(rng.normal(0, 0.0003, n_total))
    log_gld_base = TRUE_ALPHA + TRUE_BETA * log_iau_base + rng.normal(0, 0.0004, n_total)

    timestamps = pd.date_range(
        start="2025-01-02 09:30",
        periods=n_total,
        freq="1min"
    )

    prior_data = pd.DataFrame({
        "GLD": np.exp(log_gld_base),
        "IAU": np.exp(log_iau_base),
    }, index=timestamps)

    print(f"\nSynthetic prior data | sessions={n_sessions} | bars={len(prior_data)}")
    print(f"GLD range: {prior_data['GLD'].min():.2f} – {prior_data['GLD'].max():.2f}")
    print(f"IAU range: {prior_data['IAU'].min():.2f} – {prior_data['IAU'].max():.2f}")

    # ── Initialize engine with prior data ─────────────────────────────────────
    engine = SpreadEngine(cfg)
    engine.initialize_session(prior_data)

    print(f"\nEstimated beta  = {engine.beta:.5f}  (true={TRUE_BETA})")
    print(f"Estimated alpha = {engine.alpha:.5f}  (true={TRUE_ALPHA})")
    print(f"sigma_spread_prior = {engine.sigma_spread_prior:.6f}")
    print(f"beta_valid = {engine.beta_valid}")

    beta_error  = abs(engine.beta - TRUE_BETA)
    alpha_error = abs(engine.alpha - TRUE_ALPHA)
    print(f"\nBeta estimation error:  {beta_error:.5f}")
    print(f"Alpha estimation error: {alpha_error:.5f}")

    assert beta_error  < 0.05, f"Beta error too large: {beta_error:.5f}"
    assert alpha_error < 0.10, f"Alpha error too large: {alpha_error:.5f}"
    print("✓ Estimation accuracy within tolerance")

    # ── Simulate a few current-session bars ───────────────────────────────────
    print("\nSimulating 5 current-session bars:")
    print(f"{'Bar':>4} {'GLD':>8} {'IAU':>8} {'Spread':>12} {'Valid':>6}")
    print("-" * 44)

    for i in range(5):
        # Slightly different prices from prior sessions
        gld_bar = 191.0 + rng.normal(0, 0.05)
        iau_bar =  40.2 + rng.normal(0, 0.01)
        bar = engine.compute_spread(gld_bar, iau_bar)
        print(f"{i:>4} {gld_bar:>8.3f} {iau_bar:>8.3f} {bar.spread:>12.6f} {str(bar.valid):>6}")
        assert bar.valid, f"Bar {i} returned valid=False unexpectedly"

    # ── Test invalid price handling ───────────────────────────────────────────
    print("\nTesting invalid price handling:")
    bad_bar = engine.compute_spread(float("nan"), 40.0)
    assert not bad_bar.valid, "NaN price should return valid=False"
    print("✓ NaN GLD price → valid=False")

    bad_bar = engine.compute_spread(191.0, -5.0)
    assert not bad_bar.valid, "Negative price should return valid=False"
    print("✓ Negative IAU price → valid=False")

    # ── Test state snapshot / restore ─────────────────────────────────────────
    snapshot = engine.get_state_snapshot()
    engine2  = SpreadEngine(cfg)
    engine2.restore_state(snapshot)
    bar_orig    = engine.compute_spread(191.0, 40.2)
    bar_restored = engine2.compute_spread(191.0, 40.2)
    assert abs(bar_orig.spread - bar_restored.spread) < 1e-12, \
        "Spread mismatch after state restore"
    print("✓ State snapshot / restore produces identical spread")

    # ── Test uninitialized guard ──────────────────────────────────────────────
    engine3 = SpreadEngine(cfg)
    bad_bar = engine3.compute_spread(191.0, 40.2)
    assert not bad_bar.valid, "Uninitialized engine should return valid=False"
    print("✓ Uninitialized engine → valid=False")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
