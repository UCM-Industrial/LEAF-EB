"""Keep LEAF-EB console output concise while retaining detailed diagnostics."""

from __future__ import annotations

import os
from typing import Any


_LEVELS = {"quiet": 0, "standard": 1, "detailed": 2}
_PRINTED_KEYS: set[str] = set()


def console_level(config: dict[str, Any] | None) -> str:
    """Return the configured console detail level."""

    if not isinstance(config, dict):
        return "standard"
    raw = config.get("console_output")
    if raw is None:
        simulation = config.get("simulation", {})
        if isinstance(simulation, dict):
            raw = simulation.get("console_output")
    value = str(raw or "standard").strip().lower()
    return value if value in _LEVELS else "standard"


def is_worker_process() -> bool:
    """Return whether the current process is an isolated simulation worker."""

    return os.environ.get("LEAF_WORKER", "0") == "1"


def is_pipeline_child() -> bool:
    """Return whether Pattern/Predictor runs under the main Runner process."""

    return os.environ.get("LEAF_PIPELINE_CHILD", "0") == "1"


def emit(
        config: dict[str, Any] | None, message: str,
        level: str = "standard", allow_worker: bool = False) -> None:
    """Print one message when its detail level is enabled."""

    configured = _LEVELS[console_level(config)]
    required = _LEVELS.get(level, _LEVELS["standard"])
    if configured < required:
        return
    if configured < _LEVELS["detailed"]:
        if is_worker_process() and not allow_worker:
            return
        if is_pipeline_child():
            return
    print(message, flush=True)


def emit_once(
        config: dict[str, Any] | None, message: str, key: str,
        level: str = "standard", allow_worker: bool = False) -> None:
    """Print a message at most once per Python process."""

    if key in _PRINTED_KEYS:
        return
    configured = _LEVELS[console_level(config)]
    required = _LEVELS.get(level, _LEVELS["standard"])
    if configured < required:
        return
    if configured < _LEVELS["detailed"]:
        if is_worker_process() and not allow_worker:
            return
        if is_pipeline_child():
            return
    _PRINTED_KEYS.add(key)
    print(message, flush=True)
