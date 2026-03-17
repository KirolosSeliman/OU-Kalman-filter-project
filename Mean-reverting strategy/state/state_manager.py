"""
state/state_manager.py
─────────────────────────────────────────────────────────────────────────────
Persistent state manager for cross-session continuity.

Responsibilities:
    1. Serialize on_session_close() snapshots to disk (JSON)
    2. Load and validate snapshots at on_session_open()
    3. Maintain a rolling archive of prior session snapshots
    4. Detect and handle corruption gracefully — never crash the live loop

Storage layout:
    state/
        latest.json          ← always the most recent valid snapshot
        archive/
            YYYY-MM-DD.json  ← one file per session, retained for N days

Design invariants:
    - A missing or corrupt latest.json → return None (cold start, not crash)
    - All writes are atomic: write to .tmp, then rename → no partial writes
    - Schema version checked on load — forward-incompatible changes detected
    - All float values serialized with full precision (repr, not str)
    - numpy arrays serialized as lists; restored as np.ndarray on load

Schema version:
    SCHEMA_VERSION = "1.0.0"
    Stored in every snapshot. If loaded version != current, returns None
    and logs a warning. Forces cold start rather than silent corruption.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

from config.config_loader import SystemConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# JSON encoder/decoder with numpy and float precision support
# ─────────────────────────────────────────────────────────────────────────────

class _PrecisionEncoder(json.JSONEncoder):
    """
    Custom JSON encoder.
    - numpy arrays  → {"__ndarray__": [values]}
    - numpy scalars → Python float/int
    - float nan/inf → {"__special_float__": "nan"/"inf"/"-inf"}
    - all other floats → full repr precision
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, float):
            if np.isnan(obj):
                return {"__special_float__": "nan"}
            if np.isposinf(obj):
                return {"__special_float__": "inf"}
            if np.isneginf(obj):
                return {"__special_float__": "-inf"}
        return super().default(obj)


def _decode_hook(obj: dict) -> Any:
    """
    Custom JSON object hook.
    Restores __ndarray__ → np.ndarray and __special_float__ → float.
    """
    if "__ndarray__" in obj:
        return np.array(obj["__ndarray__"], dtype=float)
    if "__special_float__" in obj:
        val = obj["__special_float__"]
        if val == "nan":
            return float("nan")
        if val == "inf":
            return float("inf")
        if val == "-inf":
            return float("-inf")
    return obj


def _dumps(obj: Any) -> str:
    return json.dumps(obj, cls=_PrecisionEncoder, indent=2)


def _loads(s: str) -> Any:
    return json.loads(s, object_hook=_decode_hook)


# ─────────────────────────────────────────────────────────────────────────────
# StateManager
# ─────────────────────────────────────────────────────────────────────────────

class StateManager:
    """
    Manages persistence of PipelineRunner session snapshots.

    Usage:
        sm = StateManager(cfg)

        # At session open:
        snapshot = sm.load_latest()          # None on first run or corruption
        runner.on_session_open(init_data, state_snapshot=snapshot)

        # At session close:
        snapshot = runner.on_session_close()
        sm.save(snapshot, session_date="2026-03-14")

        # Optional: periodic archive cleanup
        sm.purge_old_archives(retain_days=30)
    """

    def __init__(self, cfg: SystemConfig) -> None:
        self._cfg         = cfg
        self._state_dir   = Path(cfg.logging.state_dir)
        self._archive_dir = self._state_dir / "archive"
        self._latest_path = self._state_dir / "latest.json"

        # Ensure directories exist
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # Load
    # ─────────────────────────────────────────────────────────────────────

    def load_latest(self) -> Optional[dict]:
        """
        Load the most recent valid session snapshot.

        Returns
        -------
        dict
            The snapshot produced by PipelineRunner.on_session_close(),
            ready to pass directly to PipelineRunner.on_session_open().
        None
            If no snapshot exists, the file is corrupt, or the schema
            version is incompatible. Caller must handle cold start.
        """
        if not self._latest_path.exists():
            logger.info(
                "StateManager: no latest.json found. "
                "Cold start — all engines initialize from scratch."
            )
            return None

        try:
            raw = self._latest_path.read_text(encoding="utf-8")
            data = _loads(raw)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.error(
                f"StateManager: failed to parse latest.json: {exc}. "
                "Returning None — cold start."
            )
            self._quarantine(self._latest_path)
            return None

        # Schema version check
        stored_version = data.get("__schema_version__")
        if stored_version != SCHEMA_VERSION:
            logger.warning(
                f"StateManager: schema version mismatch. "
                f"Stored={stored_version}, current={SCHEMA_VERSION}. "
                "Returning None — cold start."
            )
            self._quarantine(self._latest_path)
            return None

        # Validate required top-level keys
        required = {"spread", "kalman", "station", "risk", "exec", "hl_entry"}
        missing  = required - set(data.keys())
        if missing:
            logger.error(
                f"StateManager: latest.json missing required keys: {missing}. "
                "Returning None — cold start."
            )
            self._quarantine(self._latest_path)
            return None

        session_date = data.get("__session_date__", "unknown")
        logger.info(
            f"StateManager: loaded snapshot from session {session_date}."
        )

        # Strip metadata keys before returning to PipelineRunner
        snapshot = {k: v for k, v in data.items()
                    if not k.startswith("__")}
        return snapshot

    # ─────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────

    def save(
        self,
        snapshot:     dict,
        session_date: str,
    ) -> None:
        """
        Atomically save a session snapshot to disk.

        Parameters
        ----------
        snapshot : dict
            Output of PipelineRunner.on_session_close().
        session_date : str
            ISO date string, e.g. "2026-03-14". Used as archive filename.

        The write is atomic: JSON written to .tmp, then renamed.
        If the rename fails (cross-device), falls back to copy + delete.
        """
        # Attach metadata
        data = dict(snapshot)
        data["__schema_version__"] = SCHEMA_VERSION
        data["__session_date__"]   = session_date
        data["__saved_at__"]       = datetime.utcnow().isoformat() + "Z"

        try:
            serialized = _dumps(data)
        except (TypeError, ValueError) as exc:
            logger.error(
                f"StateManager: serialization failed: {exc}. "
                "State NOT saved. Next session will cold-start."
            )
            return

        # ── Atomic write to latest.json ───────────────────────────────────
        tmp_path = self._latest_path.with_suffix(".tmp")
        try:
            tmp_path.write_text(serialized, encoding="utf-8")
            self._atomic_replace(tmp_path, self._latest_path)
            logger.info(
                f"StateManager: snapshot saved → {self._latest_path}"
            )
        except OSError as exc:
            logger.error(
                f"StateManager: failed to write latest.json: {exc}. "
                "State NOT saved."
            )
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return

        # ── Write to archive ──────────────────────────────────────────────
        archive_path = self._archive_dir / f"{session_date}.json"
        try:
            archive_path.write_text(serialized, encoding="utf-8")
            logger.info(
                f"StateManager: archived → {archive_path}"
            )
        except OSError as exc:
            # Archive failure is non-fatal — latest.json already written
            logger.warning(
                f"StateManager: archive write failed: {exc}. "
                "latest.json is intact."
            )

    # ─────────────────────────────────────────────────────────────────────
    # Archive management
    # ─────────────────────────────────────────────────────────────────────

    def load_archive(self, session_date: str) -> Optional[dict]:
        """
        Load a specific archived session snapshot by date.

        Parameters
        ----------
        session_date : str
            ISO date string, e.g. "2026-03-13".

        Returns
        -------
        dict or None — same semantics as load_latest().
        """
        archive_path = self._archive_dir / f"{session_date}.json"
        if not archive_path.exists():
            logger.warning(
                f"StateManager: no archive found for {session_date}."
            )
            return None

        try:
            raw  = archive_path.read_text(encoding="utf-8")
            data = _loads(raw)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.error(
                f"StateManager: failed to parse archive {session_date}: {exc}."
            )
            return None

        stored_version = data.get("__schema_version__")
        if stored_version != SCHEMA_VERSION:
            logger.warning(
                f"StateManager: archive {session_date} schema mismatch "
                f"({stored_version} != {SCHEMA_VERSION})."
            )
            return None

        return {k: v for k, v in data.items() if not k.startswith("__")}

    def purge_old_archives(self, retain_days: int = 30) -> int:
        """
        Delete archive files older than retain_days.

        Parameters
        ----------
        retain_days : int
            Number of calendar days of archives to retain.

        Returns
        -------
        int — number of files deleted.
        """
        cutoff  = datetime.utcnow() - timedelta(days=retain_days)
        deleted = 0

        for path in self._archive_dir.glob("*.json"):
            try:
                # Parse date from filename YYYY-MM-DD.json
                stem = path.stem  # "2026-01-01"
                file_date = datetime.strptime(stem, "%Y-%m-%d")
                if file_date < cutoff:
                    path.unlink()
                    deleted += 1
                    logger.debug(f"StateManager: purged archive {path.name}")
            except (ValueError, OSError):
                pass  # non-date filenames or permission errors — skip silently

        if deleted > 0:
            logger.info(
                f"StateManager: purged {deleted} archive file(s) "
                f"older than {retain_days} days."
            )
        return deleted

    def list_archives(self) -> list[str]:
        """
        Return sorted list of available archive session dates (YYYY-MM-DD).
        """
        dates = []
        for path in self._archive_dir.glob("*.json"):
            try:
                datetime.strptime(path.stem, "%Y-%m-%d")
                dates.append(path.stem)
            except ValueError:
                pass
        return sorted(dates)

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _atomic_replace(src: Path, dst: Path) -> None:
        """
        Rename src → dst atomically. Falls back to copy+delete if
        os.replace raises (e.g. cross-device on some Windows configs).
        """
        try:
            os.replace(src, dst)
        except OSError:
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)

    def _quarantine(self, path: Path) -> None:
        """
        Rename a corrupt file to .corrupt so it is not loaded again
        but is retained for post-mortem analysis.
        """
        corrupt_path = path.with_suffix(".corrupt")
        try:
            os.replace(path, corrupt_path)
            logger.warning(
                f"StateManager: quarantined corrupt file → {corrupt_path}"
            )
        except OSError as exc:
            logger.warning(
                f"StateManager: could not quarantine {path}: {exc}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# Run from project root: python -m state.state_manager
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s | %(name)s | %(message)s")

    print("=" * 65)
    print("StateManager smoke test")
    print("=" * 65)

    cfg = SystemConfig.from_yaml("config/params.yaml")

    # Override state_dir to a temp directory — never pollute production state
    import dataclasses
    cfg = dataclasses.replace(
        cfg,
        logging=dataclasses.replace(cfg.logging, state_dir="state/_test_tmp")
    )

    sm = StateManager(cfg)

    # ── Build a realistic snapshot mimicking PipelineRunner.on_session_close
    def make_snapshot() -> dict:
        return {
            "spread": {
                "beta":               1.3501,
                "alpha":              0.1198,
                "sigma_spread_prior": 0.000412,
                "beta_valid":         True,
                "initialized":        True,
                "n_sessions_used":    30,
            },
            "kalman": {
                "models": [
                    {"x_hat": 0.00015, "P": 0.000003,
                     "Q_k": 1e-7, "R_k": 2e-7,
                     "alpha_R": 12.5, "beta_R": 2.5e-6,
                     "alpha_Q": 12.5, "beta_Q": 1.2e-7},
                ] * 3,
                "probs":       [0.72, 0.18, 0.10],
                "prev_x_hat":  0.00012,
                "bar_count":   385,
                "Q_mle":       8e-8,
                "R_mle":       2e-7,
                "initialized": True,
                "innov_buffer": [1e-5, -2e-5, 3e-5, -1e-5, 2e-5],
            },
            "station": {
                "last_phi":     0.8777,
                "last_mu":      0.0001,
                "last_sigma_ou":0.000981,
                "last_hl_bars": 5.32,
                "ou_ever_valid":True,
            },
            "risk": {
                "obs_buf": [
                    {"z_risk": 0.1, "regime_risk": 0.05,
                     "filter_risk": 0.02, "time_risk": 0.0,
                     "hl_jump_risk": 0.0, "next_move": 0.0003}
                ] * 45,
                "pending": None,
            },
            "exec": {
                "current_position":  0,
                "entry_spread":      float("nan"),
                "trade_age":        -1,
                "realized_pnl":     12.50,
                "cooldown_until":   -1,
                "daily_loss_halted": False,
            },
            "hl_entry": float("nan"),
        }

    snap = make_snapshot()

    # ── Test 1: Save and load round-trip ──────────────────────────────────
    print("\n[Test 1] Save and load round-trip")
    sm.save(snap, session_date="2026-03-14")
    loaded = sm.load_latest()
    assert loaded is not None, "load_latest() returned None after save"

    # Verify key values preserved
    assert abs(loaded["spread"]["beta"] - 1.3501) < 1e-10
    assert abs(loaded["station"]["last_phi"] - 0.8777) < 1e-10
    assert abs(loaded["exec"]["realized_pnl"] - 12.50) < 1e-10
    assert not loaded["exec"]["daily_loss_halted"]
    assert not np.isfinite(loaded["hl_entry"]), "NaN hl_entry should round-trip as NaN"
    print("  ✓ All values preserved through JSON round-trip")

    # ── Test 2: numpy array round-trip ────────────────────────────────────
    print("\n[Test 2] numpy array serialization")
    snap2          = make_snapshot()
    snap2["test_array"] = np.array([0.1, 0.2, 0.3, float("nan"), float("inf")])
    sm.save(snap2, session_date="2026-03-14")
    loaded2 = sm.load_latest()
    arr = loaded2["test_array"]
    assert isinstance(arr, np.ndarray), "Should restore as np.ndarray"
    assert abs(arr[0] - 0.1) < 1e-15
    assert np.isnan(arr[3]),  "NaN element should survive round-trip"
    assert np.isposinf(arr[4]), "inf element should survive round-trip"
    print("  ✓ numpy arrays with NaN/inf round-trip correctly")

    # ── Test 3: Archive written ────────────────────────────────────────────
    print("\n[Test 3] Archive written")
    archives = sm.list_archives()
    assert "2026-03-14" in archives, f"Expected archive for 2026-03-14, got {archives}"
    print(f"  Archives found: {archives}  ✓")

    # ── Test 4: Archive load ───────────────────────────────────────────────
    print("\n[Test 4] Load specific archive")
    arch = sm.load_archive("2026-03-14")
    assert arch is not None
    assert abs(arch["station"]["last_phi"] - 0.8777) < 1e-10
    print("  ✓ Archive load correct")

    # ── Test 5: Missing latest → None (cold start) ────────────────────────
    print("\n[Test 5] Missing latest.json → None (cold start)")
    sm._latest_path.unlink(missing_ok=True)
    result = sm.load_latest()
    assert result is None, "Should return None when latest.json missing"
    print("  ✓ Missing file returns None correctly")

    # ── Test 6: Corrupt file → None + quarantine ──────────────────────────
    print("\n[Test 6] Corrupt JSON → None + quarantine")
    sm._latest_path.write_text("{invalid json{{{{", encoding="utf-8")
    result = sm.load_latest()
    assert result is None, "Corrupt JSON should return None"
    corrupt_path = sm._latest_path.with_suffix(".corrupt")
    assert corrupt_path.exists(), "Corrupt file should be quarantined"
    print("  ✓ Corrupt file returns None and is quarantined")

    # ── Test 7: Schema version mismatch → None ────────────────────────────
    print("\n[Test 7] Schema version mismatch → None (cold start)")
    bad_snap          = make_snapshot()
    bad_snap["__schema_version__"] = "0.9.0"  # wrong version
    bad_snap["__session_date__"]   = "2026-03-14"
    bad_snap["__saved_at__"]       = "2026-03-14T00:00:00Z"
    sm._latest_path.write_text(_dumps(bad_snap), encoding="utf-8")
    result = sm.load_latest()
    assert result is None, "Schema mismatch should return None"
    print("  ✓ Schema mismatch returns None correctly")

    # ── Test 8: Missing required key → None ───────────────────────────────
    print("\n[Test 8] Missing required key → None")
    bad_snap2 = make_snapshot()
    bad_snap2["__schema_version__"] = SCHEMA_VERSION
    bad_snap2["__session_date__"]   = "2026-03-14"
    bad_snap2["__saved_at__"]       = "2026-03-14T00:00:00Z"
    del bad_snap2["kalman"]  # remove required key
    sm._latest_path.write_text(_dumps(bad_snap2), encoding="utf-8")
    result = sm.load_latest()
    assert result is None, "Missing required key should return None"
    print("  ✓ Missing required key returns None")

    # ── Test 9: Metadata keys stripped from returned snapshot ─────────────
    print("\n[Test 9] Metadata keys stripped from returned snapshot")
    sm.save(make_snapshot(), session_date="2026-03-14")
    loaded3 = sm.load_latest()
    meta_keys = [k for k in loaded3.keys() if k.startswith("__")]
    assert len(meta_keys) == 0, f"Metadata keys should be stripped: {meta_keys}"
    print("  ✓ No __ metadata keys in returned snapshot")

    # ── Test 10: Purge old archives ────────────────────────────────────────
    print("\n[Test 10] Archive purge")
    # Write a fake old archive
    old_archive = sm._archive_dir / "2020-01-01.json"
    old_archive.write_text(_dumps({"__schema_version__": SCHEMA_VERSION,
                                   "__session_date__": "2020-01-01"}),
                           encoding="utf-8")
    n_deleted = sm.purge_old_archives(retain_days=30)
    assert n_deleted >= 1, f"Expected at least 1 deletion, got {n_deleted}"
    assert not old_archive.exists(), "Old archive should be deleted"
    print(f"  ✓ Purged {n_deleted} old archive(s)")

    # ── Cleanup temp directory ─────────────────────────────────────────────
    import shutil as _shutil
    _shutil.rmtree("state/_test_tmp", ignore_errors=True)
    print("\n  ✓ Temp test directory cleaned up")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED")
    print("=" * 65)
