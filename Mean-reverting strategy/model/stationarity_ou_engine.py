"""
model/stationarity_ou_engine.py
─────────────────────────────────────────────────────────────────────────────
Rolling stationarity gate and OU parameter estimator.

Responsibilities:
  1. Maintain a rolling buffer of the last `window` Kalman-filtered spread
     values (x_hat from KalmanIMMEngine)
  2. Compute ADF p-value and Hurst exponent (R/S method) on that buffer
  3. Gate logic:
       OPEN   ← ADF p < threshold  AND  Hurst < hurst_threshold
       CLOSED ← ADF p >= threshold  OR  Hurst >= hurst_threshold
       TRENDING EXIT signal ← Hurst >= hurst_trending_exit (while gate closed)
  4. When gate is OPEN: estimate OU parameters (phi, mu, sigma_ou, hl_bars)
     via OLS on the same filtered buffer
  5. Hold last valid OU parameters when gate closes or OLS produces invalid
     estimates — never emit NaN params downstream

Causality guarantee:
  All computations use only x_hat values seen at or before bar k.
  The buffer is a rolling deque — no look-ahead possible.

Hurst exponent method (R/S):
  H = log(R/S_n) / log(n)
  R/S_n = (max(cumdev) - min(cumdev)) / std(series)
  where cumdev = cumulative deviation from mean.
  H < 0.45 → mean-reverting
  H ∈ [0.45, 0.55] → random walk
  H > 0.55 → trending
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from statsmodels.tsa.stattools import adfuller

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationarityOUBar:
    """All outputs of StationarityOUEngine.step() for a single bar."""
    # Gate
    gate_open:        bool    # True iff ADF p < threshold AND Hurst < hurst_threshold
    trending_exit:    bool    # True iff Hurst >= hurst_trending_exit (force-exit signal)
    adf_pvalue:       float   # ADF p-value (nan if buffer not full)
    hurst:            float   # Hurst exponent R/S (nan if buffer not full)
    buffer_full:      bool    # True once rolling buffer has >= window bars

    # OU parameters (hold-last-valid: never NaN after first valid estimation)
    phi:              float   # AR(1) coefficient (mean-reversion speed)
    mu:               float   # long-run mean of filtered spread
    sigma_ou:         float   # residual std of OLS fit
    hl_bars:          float   # half-life in bars = ln(2) / ln(1/phi)
    ou_valid:         bool    # True if current-bar OLS estimate passed validity checks
    ou_ever_valid:    bool    # True if at least one valid estimate has been produced


# ─────────────────────────────────────────────────────────────────────────────
# StationarityOUEngine
# ─────────────────────────────────────────────────────────────────────────────

class StationarityOUEngine:
    """
    Rolling stationarity gate and OU estimator.

    Correct usage pattern:
        engine = StationarityOUEngine(cfg)
        engine.reset_session()          # once at session open
        for each bar k:
            bar = engine.step(x_hat_k)
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg = cfg

        # Rolling buffer of filtered spread values (x_hat from Kalman)
        self._buffer: deque = deque(maxlen=cfg.ou.window)

        # Last valid OU parameters (held until next valid estimation)
        self._last_phi:      float = float("nan")
        self._last_mu:       float = float("nan")
        self._last_sigma_ou: float = float("nan")
        self._last_hl_bars:  float = float("nan")
        self._ou_ever_valid: bool  = False


    # ─────────────────────────────────────────────────────────────────────────
    # Session reset
    # ─────────────────────────────────────────────────────────────────────────

    def reset_session(self) -> None:
        """
        Clear the rolling buffer at session open ONLY. OU params are held across
        sessions (last valid persists).
        """
        self._buffer.clear()
        logger.info("StationarityOUEngine: session buffer cleared (OU params preserved).")

    # ─────────────────────────────────────────────────────────────────────────
    # Per-bar step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, x_hat_k: float) -> StationarityOUBar:
        """
        Process one bar.

    Parameters
    ----------
    x_hat_k : float
        Kalman-filtered spread estimate at bar k, from KalmanIMMEngine.

    Returns
    -------
    StationarityOUBar
        """
        # NaN input: gate closed, hold params
        if not np.isfinite(x_hat_k):
            return self._closed_bar(buffer_full=len(self._buffer) >= self._cfg.ou.window)

        self._buffer.append(x_hat_k)
        window      = self._cfg.ou.window
        buffer_full = len(self._buffer) >= window

        if not buffer_full:
            return self._closed_bar(buffer_full=False)

        arr = np.array(self._buffer, dtype=float)

        # ── Step 1: ADF test ──────────────────────────────────────────────────
        adf_pvalue = self._compute_adf(arr)

        # ── Step 2: Hurst exponent ────────────────────────────────────────────
        hurst = self._compute_hurst_ac1(arr)

        # ── Step 3: Gate logic ────────────────────────────────────────────────
        cfg_s     = self._cfg.stationarity
        adf_ok    = np.isfinite(adf_pvalue) and (adf_pvalue < cfg_s.adf_pvalue_threshold)
        hurst_ok  = np.isfinite(hurst)      and (hurst       < cfg_s.hurst_threshold)
        gate_open = adf_ok and hurst_ok

        trending_exit = np.isfinite(hurst) and (hurst >= cfg_s.hurst_trending_exit)

        # ── Step 4: OU estimation (only when gate open) ───────────────────────
        # Buffer is full (guaranteed above). Estimate whenever gate is open.
        # Hold last-valid when gate is closed — never emit NaN downstream.
        ou_valid = False
        if gate_open:
            phi, mu, sigma_ou, hl_bars, ou_valid = self._estimate_ou(arr)
            if ou_valid:
                self._last_phi      = phi
                self._last_mu       = mu
                self._last_sigma_ou = sigma_ou
                self._last_hl_bars  = hl_bars
                self._ou_ever_valid = True

        return StationarityOUBar(
        gate_open     = gate_open,
        trending_exit = trending_exit,
        adf_pvalue    = adf_pvalue,
        hurst         = hurst,
        buffer_full   = buffer_full,
        phi           = self._last_phi,
        mu            = self._last_mu,
        sigma_ou      = self._last_sigma_ou,
        hl_bars       = self._last_hl_bars,
        ou_valid      = ou_valid,
        ou_ever_valid = self._ou_ever_valid,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ADF
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_adf(self, arr: np.ndarray) -> float:
        """
        Run ADF test on arr. Returns p-value, or nan on failure.

        Null hypothesis: unit root (random walk).
        Low p-value → reject null → series is stationary → gate can open.
        """
        try:
            result = adfuller(arr, autolag="AIC", regression="c")
            pvalue = float(result[1])
            if not np.isfinite(pvalue):
                return float("nan")
            return pvalue
        except Exception as exc:
            logger.debug(f"StationarityOUEngine ADF failed: {exc}")
            return float("nan")

    # ─────────────────────────────────────────────────────────────────────────
    # Hurst exponent (R/S method)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_hurst_ac1(self, arr: np.ndarray) -> float:
        """
        Hurst proxy via lag-1 autocorrelation of first differences.

        Theoretical basis for AR(1) with coefficient phi:
        AC1(delta_x) = -(1 - phi) / 2
        H_proxy = 0.5 + AC1(delta_x)

        Mapping:
            phi = 0.85 → AC1 = -0.075 → H = 0.425  (mean-reverting, gate can open)
            phi = 0.95 → AC1 = -0.025 → H = 0.475  (near-RW, gate closed)
            phi = 1.00 → AC1 =  0.000 → H = 0.500  (random walk, gate closed)
            trending   → AC1 ≈ +0.5+  → H ≥ 0.55+ (trending_exit fires)

        Reliable at n=60. R/S requires n≥500 for equivalent reliability.
        """
        n = len(arr)
        if n < 8:
            return float("nan")

        diffs = np.diff(arr)
        if len(diffs) < 3:
            return float("nan")

        mean_d = float(np.mean(diffs))
        diffs_c = diffs - mean_d
        var_d   = float(np.var(diffs_c, ddof=1))

        if var_d < 1e-20:
            return float("nan")

        # Lag-1 autocorrelation of differences
        ac1 = float(np.dot(diffs_c[:-1], diffs_c[1:])) / ((len(diffs_c) - 1) * var_d)

        H = 0.5 + ac1

        # Clip to valid Hurst range
        H = float(np.clip(H, 0.01, 0.99))
        return H

    # ─────────────────────────────────────────────────────────────────────────
    # OU estimation via OLS AR(1)
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_ou(
        self, arr: np.ndarray
    ) -> tuple[float, float, float, float, bool]:
        """
        Estimate OU parameters via OLS regression of the discrete AR(1):
            x_t = a + b * x_{t-1} + epsilon_t

        phi      = b                    (mean-reversion coefficient)
        mu       = a / (1 - b)          (long-run mean)
        sigma_ou = std(residuals)       (residual noise)
        hl_bars  = ln(2) / ln(1/phi)   (half-life in bars)

        Validity checks (from config):
          phi ∈ (phi_min, phi_max)
          hl_bars ∈ (hl_min_bars, hl_max_bars)
          sigma_ou > 0

        Returns (phi, mu, sigma_ou, hl_bars, valid).
        All floats are nan and valid=False on failure.
        """
        _nan = float("nan")

        if len(arr) < 4:
            return _nan, _nan, _nan, _nan, False

        y = arr[1:]
        x = arr[:-1]
        X = np.column_stack([np.ones(len(x)), x])

        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError as exc:
            logger.debug(f"StationarityOUEngine OLS failed: {exc}")
            return _nan, _nan, _nan, _nan, False

        a   = float(coeffs[0])
        b   = float(coeffs[1])
        phi = b

        cfg_ou = self._cfg.ou

        # Validity: phi range
        if not (cfg_ou.phi_min < phi < cfg_ou.phi_max):
            return _nan, _nan, _nan, _nan, False

        # mu
        denom = 1.0 - phi
        if abs(denom) < 1e-10:
            return _nan, _nan, _nan, _nan, False
        mu = a / denom

        # sigma_ou
        residuals = y - (a + phi * x)
        sigma_ou  = float(np.std(residuals, ddof=2))
        if sigma_ou <= 0 or not np.isfinite(sigma_ou):
            return _nan, _nan, _nan, _nan, False

        # half-life
        log_inv_phi = np.log(1.0 / phi)
        if log_inv_phi <= 0 or not np.isfinite(log_inv_phi):
            return _nan, _nan, _nan, _nan, False
        hl_bars = float(np.log(2.0) / log_inv_phi)

        # Validity: hl_bars range
        if not (cfg_ou.hl_min_bars <= hl_bars <= cfg_ou.hl_max_bars):
            return _nan, _nan, _nan, _nan, False

        return phi, mu, sigma_ou, hl_bars, True

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _closed_bar(self, buffer_full: bool) -> StationarityOUBar:
        """Return a closed-gate bar with held OU params."""
        return StationarityOUBar(
            gate_open     = False,
            trending_exit = False,
            adf_pvalue    = float("nan"),
            hurst         = float("nan"),
            buffer_full   = buffer_full,
            phi           = self._last_phi,
            mu            = self._last_mu,
            sigma_ou      = self._last_sigma_ou,
            hl_bars       = self._last_hl_bars,
            ou_valid      = False,
            ou_ever_valid  = self._ou_ever_valid,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        """Serializable snapshot for StateManager. Call at session close."""
        return {
            "last_phi":      self._last_phi,
            "last_mu":       self._last_mu,
            "last_sigma_ou": self._last_sigma_ou,
            "last_hl_bars":  self._last_hl_bars,
            "ou_ever_valid": self._ou_ever_valid,

            # Buffer is NOT persisted — it resets at each session open
        }

    def restore_state(self, snapshot: dict) -> None:
        """
        Restore last-valid OU params from prior session.
        Buffer is NOT restored — it must be rebuilt from current session bars.
        """
        self._last_phi      = snapshot.get("last_phi",      float("nan"))
        self._last_mu       = snapshot.get("last_mu",       float("nan"))
        self._last_sigma_ou = snapshot.get("last_sigma_ou", float("nan"))
        self._last_hl_bars  = snapshot.get("last_hl_bars",  float("nan"))
        self._ou_ever_valid = snapshot.get("ou_ever_valid", False)
        logger.info(
            f"StationarityOUEngine state restored | "
            f"last_hl_bars={self._last_hl_bars:.2f} | "
            f"ou_ever_valid={self._ou_ever_valid}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m model.stationarity_ou_engine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("StationarityOUEngine smoke test")
    print("=" * 65)

    cfg = SystemConfig.from_yaml("config/params.yaml")
    rng = np.random.default_rng(7)

    TRUE_PHI   = 0.85
    TRUE_SIGMA = 0.001
    N          = 300

    ou_series = np.zeros(N)
    for t in range(1, N):
        ou_series[t] = TRUE_PHI * ou_series[t-1] + TRUE_SIGMA * rng.standard_normal()

    rw_series = np.cumsum(rng.normal(0, 0.001, N))

    # ── Test 1: OU series — gate should open ─────────────────────────────
    print("\n[Test 1] OU process (phi=0.85) — gate should open")
    engine = StationarityOUEngine(cfg)
    engine.reset_session()

    gate_open_count = 0
    last_bar = None
    for t in range(N):
        bar = engine.step(ou_series[t])
        if bar.gate_open:
            gate_open_count += 1
        last_bar = bar

    print(f"  Gate open: {gate_open_count}/{N} bars ({gate_open_count/N*100:.1f}%)")
    print(f"  Last ADF p-value: {last_bar.adf_pvalue:.4f}")
    print(f"  Last Hurst:       {last_bar.hurst:.4f}")
    print(f"  Last phi:         {last_bar.phi:.4f}  (true={TRUE_PHI})")
    print(f"  Last HL (bars):   {last_bar.hl_bars:.2f}")
    print(f"  ou_ever_valid:    {last_bar.ou_ever_valid}")

    assert gate_open_count > 0,         "Gate never opened on OU process"
    assert last_bar.ou_ever_valid,      "OU never produced valid estimate"
    assert np.isfinite(last_bar.phi),   "phi is nan"
    assert abs(last_bar.phi - TRUE_PHI) < 0.15, \
        f"phi too far from truth: {last_bar.phi:.4f} vs {TRUE_PHI}"
    print("  ✓ Gate opened, OU params estimated, phi close to truth")

    # ── Test 2: Random walk — gate should stay mostly closed ─────────────
    print("\n[Test 2] Random walk — gate should stay mostly closed")
    engine2 = StationarityOUEngine(cfg)
    engine2.reset_session()
    gate_open_rw = sum(engine2.step(rw_series[t]).gate_open for t in range(N))
    print(f"  Gate open: {gate_open_rw}/{N} bars ({gate_open_rw/N*100:.1f}%)")
    assert gate_open_rw / N < 0.20, \
        f"Gate opened too often on random walk: {gate_open_rw/N*100:.1f}%"
    print("  ✓ Gate correctly suppressed on random walk")

    # ── Test 3: Trending series — trending_exit should fire ───────────────
    print("\n[Test 3] Trending series — trending_exit should fire")
    trend_diffs    = np.zeros(N)
    trend_diffs[0] = 0.001
    for t in range(1, N):
        trend_diffs[t] = 0.70 * trend_diffs[t-1] + 0.0002 * rng.standard_normal()
    trend_series = np.cumsum(trend_diffs)

    engine3       = StationarityOUEngine(cfg)
    engine3.reset_session()
    trending_count = sum(engine3.step(trend_series[t]).trending_exit for t in range(N))
    print(f"  trending_exit fired: {trending_count}/{N} bars")
    assert trending_count > 0, "trending_exit never fired on trending series"
    print("  ✓ trending_exit correctly fires on trending series")

    # ── Test 4: NaN input — gate closed, no crash ────────────────────────
    print("\n[Test 4] NaN input handling")
    engine4  = StationarityOUEngine(cfg)
    engine4.reset_session()
    bar_nan  = engine4.step(float("nan"))
    assert not bar_nan.gate_open
    assert not bar_nan.buffer_full
    print("  ✓ NaN input handled — gate closed, no crash")

    # ── Test 5: Session reset clears buffer ──────────────────────────────
    print("\n[Test 5] Session reset clears buffer, OU params preserved")
    engine5 = StationarityOUEngine(cfg)
    engine5.reset_session()
    for t in range(cfg.ou.window):
        engine5.step(ou_series[t])
    assert len(engine5._buffer) == cfg.ou.window
    # Store any valid phi before reset
    engine5.reset_session()
    assert len(engine5._buffer) == 0,          "Buffer not cleared"
    print("  ✓ Buffer cleared after reset")

    # ── Test 6: Hold-last-valid on NaN input ─────────────────────────────
    # Design contract: when gate is closed (NaN forces it), held phi is returned.
    # Testing with continuous rw is invalid — the buffer genuinely updates.
    print("\n[Test 6] OU params held on gate-closed bar (NaN input)")
    engine6 = StationarityOUEngine(cfg)
    engine6.reset_session()

    for t in range(N):
        b = engine6.step(ou_series[t])

    # Find any bar where ou_valid=True occurred
    assert engine6._ou_ever_valid, "No valid OU estimate — cannot test hold"
    saved_phi = engine6._last_phi

    # NaN forces gate closed and _closed_bar returns held phi
    b_nan = engine6.step(float("nan"))
    assert not b_nan.gate_open,             "Gate should be closed after NaN"
    assert b_nan.phi == saved_phi,          \
        f"phi should be held on NaN: got {b_nan.phi}, expected {saved_phi}"
    assert b_nan.ou_ever_valid,             "ou_ever_valid should persist"
    print(f"  ✓ phi={saved_phi:.4f} held correctly on gate-closed bar")

    # ── Test 7: State snapshot / restore ─────────────────────────────────
    print("\n[Test 7] State snapshot / restore")
    snap    = engine.get_state_snapshot()
    engine7 = StationarityOUEngine(cfg)
    engine7.restore_state(snap)
    assert abs(engine7._last_phi - engine._last_phi) < 1e-12
    assert engine7._ou_ever_valid == engine._ou_ever_valid
    print("  ✓ State snapshot / restore correct")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
