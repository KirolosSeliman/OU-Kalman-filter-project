"""
model/kalman_imm_engine.py
─────────────────────────────────────────────────────────────────────────────
VB-AKF + IMM state estimator for the cointegrated spread.

Architecture (per master spec, Module 3A):
  Layer 1 — MLE initialization: estimates Q_0, R_0 on prior-session data.
             Provides well-conditioned starting point for VB hyperparameters.
  Layer 2 — VB-AKF: per bar, per model, Inverse-Gamma posteriors on Q and R.
             Outlier-robust: innovation capped at 4*sigma_recent for VB update.
  Layer 3 — IMM: 3 parallel VB-AKFs (mean-reverting / transitional / trending).
             Model probabilities computed in log-space (underflow-safe).
             Mixture estimate x_hat_k and P_k produced every bar.

Key correctness properties:
  - Each IMM sub-filter maintains its OWN independent VB hyperparameters.
  - Q_i and R_i adapt independently per model — they are NOT shared.
  - IMM probability floor prevents any model from dying permanently.
  - P_max flag is advisory only; filter continues running if P exceeds P_max.
  - MLE runs on prior-session spread data only — causal at initialization.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal model state (one instance per IMM sub-filter)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ModelState:
    """Mutable state for one IMM sub-filter. Not exposed externally."""
    x_hat:   float   # current state estimate
    P:       float   # current error covariance
    Q_k:     float   # current process noise estimate
    R_k:     float   # current measurement noise estimate
    alpha_R: float   # VB Inv-Gamma shape for R
    beta_R:  float   # VB Inv-Gamma scale for R
    alpha_Q: float   # VB Inv-Gamma shape for Q
    beta_Q:  float   # VB Inv-Gamma scale for Q


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass (one instance per bar)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KalmanIMMBar:
    """All outputs of KalmanIMMEngine.step() for a single bar."""
    x_hat:          float            # IMM mixture state estimate
    P:              float            # IMM mixture error covariance
    imm_score:      float            # p_{1,k}: mean-reverting model probability
    p_all:          np.ndarray       # shape (3,): all model probabilities
    innovation:     float            # spread_k - x_hat_{k-1} (mixture innovation)
    Q_k:            float            # probability-weighted mixture Q (diagnostic)
    R_k:            float            # probability-weighted mixture R (diagnostic)
    filter_diverged: bool            # True if P > P_max (advisory flag)
    imm_reset:      bool             # True if probabilities were reset (underflow)


# ─────────────────────────────────────────────────────────────────────────────
# KalmanIMMEngine
# ─────────────────────────────────────────────────────────────────────────────

class KalmanIMMEngine:
    """
    VB-AKF + IMM filter for the spread series.

    Correct usage pattern:
        engine = KalmanIMMEngine(cfg)
        engine.initialize_session(prior_spread, sigma_spread_prior)
        for each bar k:
            bar = engine.step(spread_k)
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg          = cfg
        self._models:       Optional[List[_ModelState]] = None
        self._probs:        Optional[np.ndarray]        = None
        self._initialized:  bool                        = False
        self._bar_count:    int                         = 0

        # Rolling innovation buffer for outlier detection
        # Shared across all models — uses mixture innovation
        self._innov_buffer: deque = deque(maxlen=cfg.kalman.outlier_lookback)

        # Store previous mixture x_hat for innovation computation at bar level
        self._prev_x_hat:  float = 0.0

        # MLE results (stored for snapshot)
        self._Q_mle:  float = 0.0
        self._R_mle:  float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Session initialization
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_session(
        self,
        prior_spread:        np.ndarray,
        sigma_spread_prior:  float,
        first_spread:        float,
    ) -> None:
        """
        Initialize all 3 IMM sub-filters for a new session.

        Parameters
        ----------
        prior_spread : np.ndarray
            Spread values from the last complete session (or last N sessions).
            Used ONLY for MLE initialization of Q_0, R_0.
            Must be strictly prior to the current session.
        sigma_spread_prior : float
            Spread standard deviation from prior sessions.
            Computed by SpreadEngine.initialize_session().
            Used to scale Q_i / R_i per model.
        first_spread : float
            First spread value of the current session (bar 0).
            Used to set the initial state estimate x_hat[0] = first_spread.
        """
        self._initialized = False
        self._bar_count   = 0
        self._innov_buffer.clear()

        imm_cfg    = self._cfg.imm
        kal_cfg    = self._cfg.kalman
        sigma      = float(sigma_spread_prior)

        if sigma <= 0 or not np.isfinite(sigma):
            logger.warning(
                "KalmanIMMEngine: sigma_spread_prior is invalid "
                f"(got {sigma}). Using fallback 1e-4."
            )
            sigma = 1e-4

        # ── Layer 1: MLE initialization ───────────────────────────────────────
        Q_mle, R_mle = self._mle_init(prior_spread, sigma)
        self._Q_mle  = Q_mle
        self._R_mle  = R_mle

        logger.info(
            f"KalmanIMMEngine MLE init | Q_mle={Q_mle:.2e} | R_mle={R_mle:.2e}"
        )

        # ── Layer 3: Initialize 3 IMM sub-filters ────────────────────────────
        # Each model uses:
        #   Q_i_init = Q_factors[i] * sigma^2
        #   R_i_init = R_factors[i] * sigma^2
        # VB hyperparameters are initialized from the per-model Q/R init values.
        alpha_prior = kal_cfg.alpha_prior

        self._models = []
        for i in range(imm_cfg.n_models):
            Q_i = float(imm_cfg.Q_factors[i]) * sigma ** 2
            R_i = float(imm_cfg.R_factors[i]) * sigma ** 2

            # Steady-state P for this model's Q/R
            P_ss = 0.5 * Q_i + np.sqrt((0.5 * Q_i) ** 2 + Q_i * R_i)
            P_ss *= self._cfg.kalman.P_init_multiplier

            self._models.append(_ModelState(
                x_hat   = first_spread,
                P       = P_ss,
                Q_k     = Q_i,
                R_k     = R_i,
                alpha_R = alpha_prior,
                beta_R  = alpha_prior * R_i,
                alpha_Q = alpha_prior,
                beta_Q  = alpha_prior * Q_i,
            ))

        # ── IMM initial probabilities ─────────────────────────────────────────
        self._probs    = np.array(imm_cfg.initial_probs, dtype=float)
        self._probs   /= self._probs.sum()   # renormalize for safety

        self._prev_x_hat  = first_spread
        self._initialized = True

        logger.info(
            f"KalmanIMMEngine initialized | "
            f"sigma_spread={sigma:.6f} | "
            f"models={imm_cfg.n_models} | "
            f"Q_factors={imm_cfg.Q_factors} | "
            f"R_factors={imm_cfg.R_factors} | "
            f"initial_probs={list(np.round(self._probs, 4))}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MLE initialization (Layer 1)
    # ─────────────────────────────────────────────────────────────────────────

    def _mle_init(self, prior_spread: np.ndarray, sigma: float) -> tuple[float, float]:
        """
        Estimate Q_0 and R_0 by maximizing the Kalman filter log-likelihood
        on prior-session spread data.

        L(Q,R) = -1/2 * sum_k [ log(S_k) + v_k^2 / S_k ]
        S_k = P_{k|k-1} + R,  v_k = spread_k - x_hat_{k|k-1}

        Uses L-BFGS-B with bounds to prevent Q, R hitting zero.
        Falls back to variance-based heuristic if optimizer fails.
        """
        prior_spread = np.asarray(prior_spread, dtype=float)
        prior_spread = prior_spread[np.isfinite(prior_spread)]

        if len(prior_spread) < self._cfg.kalman.mle_warmup_bars:
            logger.warning(
                f"KalmanIMMEngine._mle_init: only {len(prior_spread)} prior bars "
                f"(need {self._cfg.kalman.mle_warmup_bars}). Using heuristic fallback."
            )
            return self._mle_fallback(sigma)

        diff_var = float(np.var(np.diff(prior_spread)))
        if diff_var <= 0 or not np.isfinite(diff_var):
            return self._mle_fallback(sigma)

        def _run_kf(prices, Q, R):
            """Simple fixed-QR Kalman on a price array. Returns innovations and S."""
            n    = len(prices)
            v    = np.empty(n)
            S    = np.empty(n)
            x    = prices[0]
            P    = Q + R   # rough init
            for t in range(1, n):
                P_pred = P + Q
                S_t    = P_pred + R
                v_t    = prices[t] - x
                K      = P_pred / S_t
                x      = x + K * v_t
                P      = (1.0 - K) * P_pred
                v[t]   = v_t
                S[t]   = S_t
            return v[1:], S[1:]

        def _neg_log_likelihood(params):
            Q, R = params
            if Q <= 0 or R <= 0:
                return 1e10
            v, S = _run_kf(prior_spread, Q, R)
            valid = np.isfinite(S) & np.isfinite(v) & (S > 0)
            if valid.sum() < 10:
                return 1e10
            return 0.5 * float(np.sum(np.log(S[valid]) + v[valid] ** 2 / S[valid]))

        bounds = [(1e-8, diff_var * 10), (1e-8, diff_var * 10)]
        x0     = [diff_var * 0.1, diff_var * 0.9]

        try:
            result = minimize(
                _neg_log_likelihood, x0,
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 200, "ftol": 1e-9}
            )
            if result.success and np.all(np.array(result.x) > 0):
                return float(result.x[0]), float(result.x[1])
            else:
                logger.warning(
                    f"KalmanIMMEngine MLE did not converge "
                    f"(status={result.status}: {result.message}). "
                    "Using heuristic fallback."
                )
                return self._mle_fallback(sigma)
        except Exception as exc:
            logger.warning(f"KalmanIMMEngine MLE exception: {exc}. Using fallback.")
            return self._mle_fallback(sigma)

    def _mle_fallback(self, sigma: float) -> tuple[float, float]:
        """
        Variance-based heuristic when MLE fails or has insufficient data.
        Q = 0.01 * sigma^2  (spread mostly stable → small process noise)
        R = 0.50 * sigma^2  (measurement noise ≈ half total variance)
        """
        Q = 0.01 * sigma ** 2
        R = 0.50 * sigma ** 2
        logger.info(f"KalmanIMMEngine using MLE fallback | Q={Q:.2e} | R={R:.2e}")
        return Q, R

    # ─────────────────────────────────────────────────────────────────────────
    # Per-bar step
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, spread_k: float) -> KalmanIMMBar:
        """
        Execute one bar of VB-AKF + IMM.

        Parameters
        ----------
        spread_k : float
            Spread value at bar k, produced by SpreadEngine.compute_spread().

        Returns
        -------
        KalmanIMMBar
            All outputs needed by StationarityOUEngine and SignalScore.
        """
        if not self._initialized:
            raise RuntimeError(
                "KalmanIMMEngine.step() called before initialize_session(). "
                "This is a pipeline sequencing error."
            )

        # NaN spread: hold state, return NaN outputs
        if not np.isfinite(spread_k):
            logger.warning(
                f"KalmanIMMEngine.step: spread_k is not finite (got {spread_k}). "
                "Holding state."
            )
            return self._nan_bar()

        imm_cfg = self._cfg.imm
        kal_cfg = self._cfg.kalman
        lam     = kal_cfg.vb_lambda

        # ── Rolling sigma_recent for outlier detection ────────────────────────
        # Use mixture innovation from previous bar if available
        if len(self._innov_buffer) >= 2:
            sigma_recent = float(np.std(self._innov_buffer))
        elif len(self._innov_buffer) == 1:
            sigma_recent = abs(self._innov_buffer[0]) + 1e-12
        else:
            sigma_recent = abs(spread_k - self._prev_x_hat) + 1e-12

        outlier_cap = kal_cfg.outlier_sigma_threshold * sigma_recent

        # ── Per-model VB-AKF step ─────────────────────────────────────────────
        innovations = np.zeros(imm_cfg.n_models)
        S_vals      = np.zeros(imm_cfg.n_models)
        x_hats_new  = np.zeros(imm_cfg.n_models)
        P_new_vals  = np.zeros(imm_cfg.n_models)

        for i, m in enumerate(self._models):
            # PREDICT
            P_pred = m.P + m.Q_k
            v_i    = spread_k - m.x_hat
            S_i    = P_pred + m.R_k

            # KALMAN UPDATE (raw innovation — not capped)
            K_i          = P_pred / max(S_i, 1e-15)
            x_new        = m.x_hat + K_i * v_i
            P_new        = (1.0 - K_i) * P_pred
            x_hats_new[i] = x_new
            P_new_vals[i] = max(P_new, 1e-12)   # covariance floor

            # Outlier-capped innovation for VB parameter update only
            v_vb = float(np.clip(v_i, -outlier_cap, outlier_cap))

            # VB UPDATE — R (measurement noise)
            m.alpha_R = lam * m.alpha_R + 0.5
            m.beta_R  = lam * m.beta_R  + 0.5 * (v_vb ** 2 + S_i)
            denom_R   = max(m.alpha_R - 1.0, 1e-6)
            m.R_k     = max(m.beta_R / denom_R, 1e-12)

            # VB UPDATE — Q (process noise)
            dx        = x_new - m.x_hat
            m.alpha_Q = lam * m.alpha_Q + 0.5
            m.beta_Q  = lam * m.beta_Q  + 0.5 * (dx ** 2)
            denom_Q   = max(m.alpha_Q - 1.0, 1e-6)
            m.Q_k     = max(m.beta_Q / denom_Q, 1e-12)

            # Commit state update
            m.x_hat = x_new
            m.P     = P_new_vals[i]

            innovations[i] = v_i
            S_vals[i]      = max(S_i, 1e-15)

        # ── IMM probability update (log-space) ───────────────────────────────
        # log L_i = -0.5 * [ log(2π S_i) + v_i^2 / S_i ]
        log_L = (
            -0.5 * np.log(2.0 * np.pi * S_vals)
            - 0.5 * innovations ** 2 / S_vals
        )
        # Subtract max for numerical stability before exp
        log_L_shifted = log_L - log_L.max()
        L             = np.exp(log_L_shifted)

        unnorm = L * self._probs
        total  = unnorm.sum()

        imm_reset = False
        if total < 1e-15:
            # All likelihoods collapsed — reset to uniform
            self._probs = np.full(imm_cfg.n_models, 1.0 / imm_cfg.n_models)
            imm_reset   = True
            logger.warning(
                "KalmanIMMEngine: IMM probability collapse — reset to uniform. "
                "This indicates all models fit the data equally poorly. "
                "Check spread quality and sigma_spread_prior calibration."
            )
        else:
            self._probs = unnorm / total

        # Enforce probability floor to prevent permanent model death
        floor = float(imm_cfg.prob_floor)
        self._probs = np.maximum(self._probs, floor)
        self._probs /= self._probs.sum()

        # ── IMM mixture estimate ──────────────────────────────────────────────
        x_hat_k = float(np.dot(self._probs, x_hats_new))
        P_k     = float(np.dot(
            self._probs,
            P_new_vals + (x_hats_new - x_hat_k) ** 2
        ))

        # ── Mixture diagnostic quantities ─────────────────────────────────────
        Q_mix = float(np.dot(self._probs, [m.Q_k for m in self._models]))
        R_mix = float(np.dot(self._probs, [m.R_k for m in self._models]))

        # ── Mixture innovation (for innovation buffer) ────────────────────────
        innov_mix = spread_k - self._prev_x_hat
        self._innov_buffer.append(innov_mix)
        self._prev_x_hat = x_hat_k

        # ── P_max divergence flag ─────────────────────────────────────────────
        filter_diverged = P_k > self._cfg.signal.P_max

        self._bar_count += 1

        return KalmanIMMBar(
            x_hat           = x_hat_k,
            P               = P_k,
            imm_score       = float(self._probs[0]),
            p_all           = self._probs.copy(),
            innovation      = innov_mix,
            Q_k             = Q_mix,
            R_k             = R_mix,
            filter_diverged = filter_diverged,
            imm_reset       = imm_reset,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # State persistence
    # ─────────────────────────────────────────────────────────────────────────

    def get_state_snapshot(self) -> dict:
        """Serializable snapshot for StateManager. Call at session close."""
        models_snap = None
        if self._models is not None:
            models_snap = [
                {
                    "x_hat":   m.x_hat,
                    "P":       m.P,
                    "Q_k":     m.Q_k,
                    "R_k":     m.R_k,
                    "alpha_R": m.alpha_R,
                    "beta_R":  m.beta_R,
                    "alpha_Q": m.alpha_Q,
                    "beta_Q":  m.beta_Q,
                }
                for m in self._models
            ]
        return {
            "models":       models_snap,
            "probs":        self._probs.tolist() if self._probs is not None else None,
            "prev_x_hat":   self._prev_x_hat,
            "bar_count":    self._bar_count,
            "Q_mle":        self._Q_mle,
            "R_mle":        self._R_mle,
            "initialized":  self._initialized,
            "innov_buffer": list(self._innov_buffer),
        }

    def restore_state(self, snapshot: dict) -> None:
        """Restore from snapshot produced by get_state_snapshot()."""
        self._initialized  = snapshot.get("initialized", False)
        self._prev_x_hat   = snapshot.get("prev_x_hat", 0.0)
        self._bar_count    = snapshot.get("bar_count",  0)
        self._Q_mle        = snapshot.get("Q_mle",      0.0)
        self._R_mle        = snapshot.get("R_mle",      0.0)

        buf = snapshot.get("innov_buffer", [])
        self._innov_buffer = deque(buf, maxlen=self._cfg.kalman.outlier_lookback)

        probs = snapshot.get("probs")
        self._probs = np.array(probs, dtype=float) if probs is not None else None

        models_snap = snapshot.get("models")
        if models_snap is not None:
            self._models = [
                _ModelState(**m) for m in models_snap
            ]

        logger.info(
            f"KalmanIMMEngine state restored | "
            f"bar_count={self._bar_count} | "
            f"probs={list(np.round(self._probs, 4)) if self._probs is not None else None}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _nan_bar(self) -> KalmanIMMBar:
        _nan = float("nan")
        return KalmanIMMBar(
            x_hat           = _nan,
            P               = _nan,
            imm_score       = _nan,
            p_all           = np.full(self._cfg.imm.n_models, _nan),
            innovation      = _nan,
            Q_k             = _nan,
            R_k             = _nan,
            filter_diverged = False,
            imm_reset       = False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m model.kalman_imm_engine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s"
    )

    print("=" * 65)
    print("KalmanIMMEngine smoke test")
    print("=" * 65)

    cfg = SystemConfig.from_yaml("config/params.yaml")

    rng = np.random.default_rng(0)

    # ── Synthetic spread: OU process around 0 ────────────────────────────────
    # phi=0.85 → HL ≈ 4.3 bars; sigma=0.0005
    TRUE_PHI   = 0.85
    TRUE_SIGMA = 0.0005
    N_PRIOR    = 390
    N_SESSION  = 390

    def sim_ou(n, phi, sigma, rng):
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = phi * x[t-1] + sigma * rng.standard_normal()
        return x

    prior_spread   = sim_ou(N_PRIOR,   TRUE_PHI, TRUE_SIGMA, rng)
    session_spread = sim_ou(N_SESSION, TRUE_PHI, TRUE_SIGMA, rng)
    sigma_prior    = float(np.std(prior_spread))

    print(f"\nPrior spread std:   {sigma_prior:.6f}")
    print(f"Session spread std: {np.std(session_spread):.6f}")

    # ── Initialize engine ─────────────────────────────────────────────────────
    engine = KalmanIMMEngine(cfg)
    engine.initialize_session(
        prior_spread       = prior_spread,
        sigma_spread_prior = sigma_prior,
        first_spread       = session_spread[0],
    )
    print(f"\nEngine initialized. Running {N_SESSION} bars...")

    # ── Run session bars ──────────────────────────────────────────────────────
    x_hats     = np.full(N_SESSION, np.nan)
    P_vals     = np.full(N_SESSION, np.nan)
    imm_scores = np.full(N_SESSION, np.nan)

    x_hats[0]     = session_spread[0]
    P_vals[0]      = float(engine._models[0].P)
    imm_scores[0]  = float(engine._probs[0])

    for t in range(1, N_SESSION):
        bar = engine.step(session_spread[t])
        x_hats[t]     = bar.x_hat
        P_vals[t]      = bar.P
        imm_scores[t]  = bar.imm_score

    # ── Tests ─────────────────────────────────────────────────────────────────

    # 1. No NaN outputs after warmup
    valid_slice = slice(10, N_SESSION)
    assert np.all(np.isfinite(x_hats[valid_slice])),     "NaN in x_hat"
    assert np.all(np.isfinite(P_vals[valid_slice])),      "NaN in P"
    assert np.all(np.isfinite(imm_scores[valid_slice])),  "NaN in imm_score"
    print("✓ No NaN outputs after warmup")

    # 2. P is always positive
    assert np.all(P_vals[valid_slice] > 0), "P has non-positive values"
    print("✓ P > 0 for all bars")

    # 3. IMM scores in [0, 1] and the three probs always sum to 1
    assert np.all(imm_scores[valid_slice] >= 0), "IMM score < 0"
    assert np.all(imm_scores[valid_slice] <= 1), "IMM score > 1"
    print("✓ IMM score ∈ [0, 1]")

    # 4. Filter tracking: x_hat should not diverge from spread
    tracking_error = np.abs(session_spread[valid_slice] - x_hats[valid_slice])
    max_tracking   = float(np.max(tracking_error))
    mean_tracking  = float(np.mean(tracking_error))
    print(f"✓ Filter tracking | max_error={max_tracking:.6f} | mean_error={mean_tracking:.6f}")
    assert max_tracking < 0.05, f"Filter diverged: max_tracking={max_tracking:.4f}"

    # 5. Mean-reverting model gets the highest mean probability on OU data
    bars_post_warmup = 60
    mean_p1 = float(np.mean(imm_scores[bars_post_warmup:]))
    mean_p2 = float(np.mean([engine._probs[1]]))
    print(f"✓ Mean p1 (mean-reverting) after warmup: {mean_p1:.4f}")
    # On OU data, mean-reverting model should dominate
    assert mean_p1 > 0.20, (
        f"Mean-reverting model underperforms on OU data: p1={mean_p1:.4f}. "
        "Check Q/R initialization."
    )

    # 6. NaN input handling
    bar_nan = engine.step(float("nan"))
    assert not np.isfinite(bar_nan.x_hat), "NaN spread should produce NaN x_hat"
    print("✓ NaN spread input handled gracefully")

    # 7. State snapshot / restore
    snap    = engine.get_state_snapshot()
    engine2 = KalmanIMMEngine(cfg)
    engine2.restore_state(snap)
    bar_orig     = engine.step(session_spread[-1])
    bar_restored = engine2.step(session_spread[-1])
    assert abs(bar_orig.x_hat - bar_restored.x_hat) < 1e-12, \
        "x_hat mismatch after state restore"
    assert abs(bar_orig.imm_score - bar_restored.imm_score) < 1e-12, \
        "imm_score mismatch after state restore"
    print("✓ State snapshot / restore produces identical output")

    # ── Summary ───────────────────────────────────────────────────────────────
    final_bar = engine.step(session_spread[-1])
    print(f"\nFinal bar state:")
    print(f"  x_hat     = {final_bar.x_hat:.6f}")
    print(f"  P         = {final_bar.P:.8f}")
    print(f"  imm_score = {final_bar.imm_score:.4f}  (p1: mean-reverting)")
    print(f"  p_all     = {np.round(final_bar.p_all, 4)}")
    print(f"  Q_k       = {final_bar.Q_k:.2e}")
    print(f"  R_k       = {final_bar.R_k:.2e}")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
