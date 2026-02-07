from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist (including parents)."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(
    out_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_name: str = "run.log",
) -> None:
    """
    Configure root logging to output to console and optionally to a file in out_dir.
    Clears existing handlers to prevent duplicated logs.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if out_dir is not None:
        ensure_dir(out_dir)
        handlers.append(logging.FileHandler(out_dir / log_name, encoding="utf-8"))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    for h in handlers:
        h.setLevel(level)
        h.setFormatter(formatter)
        root.addHandler(h)


def pretty_label(label: str) -> str:
    return "not real" if label == "not_real" else label
