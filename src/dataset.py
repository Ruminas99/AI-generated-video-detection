from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass(frozen=True)
class VideoSample:
    path: Path
    label: str  # "real" or "not_real"


def normalize_label(label: str) -> str:
    raw = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if raw in {"real", "original", "authentic"}:
        return "real"
    if raw in {"not_real", "fake", "deepfake", "ai", "generated", "synthetic"}:
        return "not_real"
    return raw


def parse_class_dir_map(map_str: str) -> Dict[str, str]:
    """
    Parse folder->label mapping string.
    Example: "real=real,not_real=not_real"
    """
    mapping: Dict[str, str] = {}
    if not map_str:
        return mapping

    parts = [p.strip() for p in map_str.split(",") if p.strip()]
    for part in parts:
        if "=" in part:
            folder, label = part.split("=", 1)
        elif ":" in part:
            folder, label = part.split(":", 1)
        else:
            raise ValueError(f"Invalid class_dir_map entry '{part}'. Use 'folder=label'.")

        folder = folder.strip()
        label = normalize_label(label.strip())
        if not folder:
            raise ValueError(f"Invalid folder name in entry: '{part}'")
        mapping[folder] = label
    return mapping


def _discover_videos_in_dir(dir_path: Path) -> List[Path]:
    vids: List[Path] = []
    if not dir_path.exists():
        return vids
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return sorted(vids)


def _choose_path_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    lower_map = {str(c).strip().lower(): c for c in cols}

    path_col = None
    for candidate in ["video_path", "filepath", "path", "file", "filename"]:
        if candidate in lower_map:
            path_col = lower_map[candidate]
            break
    label_col = lower_map.get("label", None)

    if path_col is None:
        path_col = cols[0]
    if label_col is None:
        if len(cols) < 2:
            raise ValueError("labels_csv must have >=2 columns: <path>,label")
        label_col = cols[1]
    return str(path_col), str(label_col)


def _resolve_video_path(dataset_root: Path, vp: str) -> Optional[Path]:
    s = str(vp).strip().strip('"').strip("'")
    if not s:
        return None

    # normalize slashes for cross-platform
    if os.sep == "/" and "\\" in s:
        s = s.replace("\\", "/")

    p = Path(s)

    if p.is_absolute() and p.exists():
        return p.resolve()

    # relative to CWD
    if p.exists():
        return p.resolve()

    # relative to dataset_root
    p2 = (dataset_root / p).resolve()
    if p2.exists():
        return p2

    return None


def load_samples(
    dataset_root: Path,
    labels_csv: Optional[Path] = None,
    class_dir_map: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[VideoSample]:
    logger = logger or logging.getLogger(__name__)
    dataset_root = dataset_root.expanduser().resolve()

    samples: List[VideoSample] = []

    if labels_csv is not None:
        labels_csv = labels_csv.expanduser().resolve()
        if not labels_csv.exists():
            raise FileNotFoundError(f"labels_csv not found: {labels_csv}")

        df = pd.read_csv(labels_csv)
        path_col, label_col = _choose_path_label_columns(df)

        missing = 0
        for _, row in df.iterrows():
            vp = str(row[path_col]).strip()
            lb = normalize_label(row[label_col])

            rp = _resolve_video_path(dataset_root, vp)
            if rp is None:
                missing += 1
                continue

            if lb not in {"real", "not_real"}:
                missing += 1
                continue

            samples.append(VideoSample(path=rp, label=lb))

        logger.info("Loaded %d samples from labels_csv (%d skipped).", len(samples), missing)
        return samples

    if not class_dir_map:
        raise ValueError("class_dir_map must be provided when labels_csv is not used.")

    for folder_name, label in class_dir_map.items():
        class_dir = dataset_root / folder_name
        vids = _discover_videos_in_dir(class_dir)
        for v in vids:
            samples.append(VideoSample(path=v.resolve(), label=normalize_label(label)))
        logger.info("Class '%s' from folder '%s': %d videos", label, class_dir, len(vids))

    logger.info("Loaded %d samples total.", len(samples))
    return samples
