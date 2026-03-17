"""
config/config_loader.py
─────────────────────────────────────────────────────────────────────────────
Load, validate, and expose typed configuration for the entire pipeline.

Design contract:
  - Load once at process startup via SystemConfig.from_yaml(path)
  - Pass the SystemConfig object to every module at construction time
  - No module reads YAML directly
  - No module has any hardcoded numeric parameter
  - Any invalid config raises ConfigValidationError at load time, not at runtime

All subconfigs are frozen dataclasses. Mutation after construction is a bug.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Exception
# ─────────────────────────────────────────────────────────────────────────────

class ConfigValidationError(ValueError):
    """Raised when config fails a semantic validity check."""


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs (frozen — immutable after construction)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetaConfig:
    version: str
    instrument_universe: List[str]
    bar_interval_minutes: int
    description: str


@dataclass(frozen=True)
class SessionConfig:
    open_time: str
    close_time: str
    force_flat_time: str
    warmup_bars: int
    min_session_bars: int
    timezone: str


@dataclass(frozen=True)
class SpreadConfig:
    leg1: str
    leg2: str
    beta_window_sessions: int
    beta_min: float
    beta_max: float
    sigma_spread_window_sessions: int
    log_prices: bool


@dataclass(frozen=True)
class KalmanConfig:
    mle_warmup_bars: int
    vb_lambda: float
    alpha_prior: float
    outlier_sigma_threshold: float
    outlier_lookback: int
    P_init_multiplier: float


@dataclass(frozen=True)
class IMMConfig:
    n_models: int
    model_names: List[str]
    Q_factors: List[float]
    R_factors: List[float]
    initial_probs: List[float]
    prob_floor: float


@dataclass(frozen=True)
class OUConfig:
    window: int
    phi_min: float
    phi_max: float
    hl_min_bars: float
    hl_max_bars: float


@dataclass(frozen=True)
class StationarityConfig:
    adf_pvalue_threshold: float
    hurst_threshold: float
    hurst_trending_exit: float
    window: int
    hurst_method: str


@dataclass(frozen=True)
class SignalConfig:
    z_scale: float
    entry_threshold: float
    hard_exit_z: float
    P_max: float


@dataclass(frozen=True)
class RiskConfig:
    z_risk_onset: float
    z_risk_full: float
    regime_risk_onset: float
    time_risk_hl_multiplier: float
    hl_jump_risk_multiplier: float
    adaptive_window: int
    stability_min_obs: int
    dampening: float
    regime_sensitivity: float
    prior_weights_mean_reverting: List[float]
    prior_weights_transitional: List[float]
    prior_weights_trending: List[float]


@dataclass(frozen=True)
class SizingConfig:
    mode: str
    risk_pct_fixed: float
    risk_pct_min: float
    risk_pct_max: float
    account_size: float
    portfolio_heat_limit: float
    position_cap_pct: float


@dataclass(frozen=True)
class ExecutionConfig:
    min_delta_threshold: float
    tc_bps: float
    slippage_ticks: int
    tick_size: float
    partial_exit_mode: str
    partial_exit_gap_threshold: float


@dataclass(frozen=True)
class SessionRiskConfig:
    daily_loss_limit_pct: float
    cooldown_bars_after_stop: int
    stronger_signal_threshold: float


@dataclass(frozen=True)
class LoggingConfig:
    session_log_dir: str
    trade_log_dir: str
    state_dir: str
    log_level: str
    log_every_bar: bool


# ─────────────────────────────────────────────────────────────────────────────
# Top-level config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SystemConfig:
    meta: MetaConfig
    session: SessionConfig
    spread: SpreadConfig
    kalman: KalmanConfig
    imm: IMMConfig
    ou: OUConfig
    stationarity: StationarityConfig
    signal: SignalConfig
    risk: RiskConfig
    sizing: SizingConfig
    execution: ExecutionConfig
    session_risk: SessionRiskConfig
    logging: LoggingConfig

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SystemConfig":
        """
        Load params.yaml, construct typed config, validate semantics.
        Raises ConfigValidationError on the first failed constraint.
        Raises FileNotFoundError if path does not exist.
        Raises yaml.YAMLError if the file is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as f:
            raw = yaml.safe_load(f)

        cfg = cls._build(raw)
        cfg._validate()
        return cfg

    # ── Internal builder ─────────────────────────────────────────────────────

    @classmethod
    def _build(cls, raw: dict) -> "SystemConfig":
        def _s(section: str) -> dict:
            if section not in raw:
                raise ConfigValidationError(f"Missing required config section: '{section}'")
            return raw[section]

        return cls(
            meta=MetaConfig(**_s("meta")),
            session=SessionConfig(**_s("session")),
            spread=SpreadConfig(**_s("spread")),
            kalman=KalmanConfig(**_s("kalman")),
            imm=IMMConfig(**_s("imm")),
            ou=OUConfig(**_s("ou")),
            stationarity=StationarityConfig(**_s("stationarity")),
            signal=SignalConfig(**_s("signal")),
            risk=RiskConfig(**_s("risk")),
            sizing=SizingConfig(**_s("sizing")),
            execution=ExecutionConfig(**_s("execution")),
            session_risk=SessionRiskConfig(**_s("session_risk")),
            logging=LoggingConfig(**_s("logging")),
        )

    # ── Semantic validation ───────────────────────────────────────────────────

    def _validate(self) -> None:
        """
        Enforce all semantic constraints that YAML schema cannot express.
        Fails fast on the first violated constraint.
        """
        self._validate_imm()
        self._validate_ou()
        self._validate_stationarity()
        self._validate_signal()
        self._validate_risk()
        self._validate_sizing()
        self._validate_execution()
        self._validate_session_risk()
        self._validate_session()

    def _validate_imm(self) -> None:
        c = self.imm
        if c.n_models != 3:
            raise ConfigValidationError(
                f"imm.n_models must be 3 (got {c.n_models}). "
                "Architecture assumes exactly 3 IMM sub-filters."
            )
        for lst_name, lst in [("Q_factors", c.Q_factors), ("R_factors", c.R_factors),
                                ("initial_probs", c.initial_probs), ("model_names", c.model_names)]:
            if len(lst) != c.n_models:
                raise ConfigValidationError(
                    f"imm.{lst_name} must have length {c.n_models} (got {len(lst)})"
                )
        for qf in c.Q_factors:
            if qf <= 0:
                raise ConfigValidationError(f"imm.Q_factors must all be > 0 (got {qf})")
        for rf in c.R_factors:
            if rf <= 0:
                raise ConfigValidationError(f"imm.R_factors must all be > 0 (got {rf})")
        if abs(sum(c.initial_probs) - 1.0) > 1e-6:
            raise ConfigValidationError(
                f"imm.initial_probs must sum to 1.0 (got {sum(c.initial_probs):.6f})"
            )
        if c.prob_floor <= 0 or c.prob_floor >= 0.01:
            raise ConfigValidationError(
                f"imm.prob_floor must be in (0, 0.01) (got {c.prob_floor})"
            )

    def _validate_ou(self) -> None:
        c = self.ou
        if not (0 < c.phi_min < c.phi_max < 1.0):
            raise ConfigValidationError(
                f"ou: required 0 < phi_min < phi_max < 1.0, "
                f"got phi_min={c.phi_min}, phi_max={c.phi_max}"
            )
        if c.hl_min_bars <= 0:
            raise ConfigValidationError(f"ou.hl_min_bars must be > 0 (got {c.hl_min_bars})")
        if c.hl_max_bars <= c.hl_min_bars:
            raise ConfigValidationError(
                f"ou.hl_max_bars must be > hl_min_bars "
                f"(got hl_min={c.hl_min_bars}, hl_max={c.hl_max_bars})"
            )
        if c.window < 20:
            raise ConfigValidationError(
                f"ou.window must be >= 20 for OLS to be meaningful (got {c.window})"
            )

    def _validate_stationarity(self) -> None:
        c = self.stationarity
        if not (0 < c.adf_pvalue_threshold < 1):
            raise ConfigValidationError(
                f"stationarity.adf_pvalue_threshold must be in (0, 1) "
                f"(got {c.adf_pvalue_threshold})"
            )
        if not (0 < c.hurst_threshold < c.hurst_trending_exit < 1):
            raise ConfigValidationError(
                f"stationarity: required 0 < hurst_threshold < hurst_trending_exit < 1, "
                f"got {c.hurst_threshold}, {c.hurst_trending_exit}"
            )
        if c.hurst_method not in ("rs","ac1_diff"):
            raise ConfigValidationError(
                f"stationarity.hurst_method must be 'rs' (got '{c.hurst_method}')"
            )
        if c.window < 20:
            raise ConfigValidationError(
                f"stationarity.window must be >= 20 (got {c.window})"
            )

    def _validate_signal(self) -> None:
        c = self.signal
        if c.z_scale <= 0:
            raise ConfigValidationError(f"signal.z_scale must be > 0 (got {c.z_scale})")
        if not (0 < c.entry_threshold < 1):
            raise ConfigValidationError(
                f"signal.entry_threshold must be in (0, 1) (got {c.entry_threshold})"
            )
        if c.hard_exit_z <= 0:
            raise ConfigValidationError(
                f"signal.hard_exit_z must be > 0 (got {c.hard_exit_z})"
            )
        if c.P_max <= 0:
            raise ConfigValidationError(f"signal.P_max must be > 0 (got {c.P_max})")

    def _validate_risk(self) -> None:
        c = self.risk
        if c.z_risk_onset >= c.z_risk_full:
            raise ConfigValidationError(
                f"risk: required z_risk_onset < z_risk_full "
                f"(got {c.z_risk_onset}, {c.z_risk_full})"
            )
        if not (0 < c.regime_risk_onset < 1):
            raise ConfigValidationError(
                f"risk.regime_risk_onset must be in (0, 1) (got {c.regime_risk_onset})"
            )
        if c.time_risk_hl_multiplier <= 0:
            raise ConfigValidationError(
                f"risk.time_risk_hl_multiplier must be > 0 (got {c.time_risk_hl_multiplier})"
            )
        if not (0 < c.dampening <= 1):
            raise ConfigValidationError(
                f"risk.dampening must be in (0, 1] (got {c.dampening})"
            )
        if c.stability_min_obs > c.adaptive_window:
            raise ConfigValidationError(
                f"risk.stability_min_obs ({c.stability_min_obs}) cannot exceed "
                f"adaptive_window ({c.adaptive_window})"
            )
        for wname, wvec in [
            ("prior_weights_mean_reverting", c.prior_weights_mean_reverting),
            ("prior_weights_transitional",   c.prior_weights_transitional),
            ("prior_weights_trending",        c.prior_weights_trending),
        ]:
            if len(wvec) != 5:
                raise ConfigValidationError(
                    f"risk.{wname} must have length 5 (got {len(wvec)})"
                )
            if any(w < 0 for w in wvec):
                raise ConfigValidationError(
                    f"risk.{wname} must have all non-negative entries"
                )
            if abs(sum(wvec) - 1.0) > 1e-6:
                raise ConfigValidationError(
                    f"risk.{wname} must sum to 1.0 (got {sum(wvec):.6f})"
                )

    def _validate_sizing(self) -> None:
        c = self.sizing
        if c.mode not in ("A", "B"):
            raise ConfigValidationError(
                f"sizing.mode must be 'A' or 'B' (got '{c.mode}')"
            )
        if not (0 < c.risk_pct_fixed < 1):
            raise ConfigValidationError(
                f"sizing.risk_pct_fixed must be in (0, 1) (got {c.risk_pct_fixed})"
            )
        if not (0 < c.risk_pct_min < c.risk_pct_max < 1):
            raise ConfigValidationError(
                f"sizing: required 0 < risk_pct_min < risk_pct_max < 1, "
                f"got {c.risk_pct_min}, {c.risk_pct_max}"
            )
        if c.account_size <= 0:
            raise ConfigValidationError(
                f"sizing.account_size must be > 0 (got {c.account_size})"
            )
        if not (0 < c.portfolio_heat_limit < 1):
            raise ConfigValidationError(
                f"sizing.portfolio_heat_limit must be in (0, 1) "
                f"(got {c.portfolio_heat_limit})"
            )

    def _validate_execution(self) -> None:
        c = self.execution
        if c.min_delta_threshold < 0:
            raise ConfigValidationError(
                f"execution.min_delta_threshold must be >= 0 (got {c.min_delta_threshold})"
            )
        if c.tc_bps < 0:
            raise ConfigValidationError(
                f"execution.tc_bps must be >= 0 (got {c.tc_bps})"
            )
        if c.slippage_ticks < 0:
            raise ConfigValidationError(
                f"execution.slippage_ticks must be >= 0 (got {c.slippage_ticks})"
            )
        if c.tick_size <= 0:
            raise ConfigValidationError(
                f"execution.tick_size must be > 0 (got {c.tick_size})"
            )
        if c.partial_exit_mode not in ("A", "B"):
            raise ConfigValidationError(
                f"execution.partial_exit_mode must be 'A' or 'B' "
                f"(got '{c.partial_exit_mode}')"
            )
        if not (0 < c.partial_exit_gap_threshold < 1):
            raise ConfigValidationError(
                f"execution.partial_exit_gap_threshold must be in (0, 1) "
                f"(got {c.partial_exit_gap_threshold})"
            )

    def _validate_session_risk(self) -> None:
        c = self.session_risk
        if not (0 < c.daily_loss_limit_pct < 1):
            raise ConfigValidationError(
                f"session_risk.daily_loss_limit_pct must be in (0, 1) "
                f"(got {c.daily_loss_limit_pct})"
            )
        if c.cooldown_bars_after_stop < 0:
            raise ConfigValidationError(
                f"session_risk.cooldown_bars_after_stop must be >= 0 "
                f"(got {c.cooldown_bars_after_stop})"
            )
        if not (0 < c.stronger_signal_threshold <= 1):
            raise ConfigValidationError(
                f"session_risk.stronger_signal_threshold must be in (0, 1] "
                f"(got {c.stronger_signal_threshold})"
            )

    def _validate_session(self) -> None:
        c = self.session
        if c.warmup_bars < 20:
            raise ConfigValidationError(
                f"session.warmup_bars must be >= 20 (got {c.warmup_bars})"
            )
        if c.min_session_bars <= c.warmup_bars:
            raise ConfigValidationError(
                f"session.min_session_bars ({c.min_session_bars}) must be > "
                f"warmup_bars ({c.warmup_bars})"
            )

    # ── Derived quantities (convenience properties) ───────────────────────────
    # These are computed from config — they are NOT parameters.
    # They exist here so no module recomputes them inline.

    @property
    def tc_decimal(self) -> float:
        """Round-trip transaction cost as a decimal fraction."""
        return self.execution.tc_bps / 10_000.0

    @property
    def daily_loss_limit_usd(self) -> float:
        """Absolute session loss limit in USD."""
        return self.sizing.account_size * self.session_risk.daily_loss_limit_pct

    @property
    def slippage_usd_per_leg(self) -> float:
        """Dollar slippage per leg per fill."""
        return self.execution.slippage_ticks * self.execution.tick_size

    @property
    def prior_weights(self) -> dict:
        """Prior weight vectors keyed by regime name, as numpy-ready lists."""
        return {
            "mean_reverting": list(self.risk.prior_weights_mean_reverting),
            "transitional":   list(self.risk.prior_weights_transitional),
            "trending":        list(self.risk.prior_weights_trending),
        }
 