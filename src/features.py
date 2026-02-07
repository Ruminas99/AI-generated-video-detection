from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

from .dataset import VideoSample
from .utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureConfig:
    """
    Defaults set to match your sdfvd2_best_v3 config.json:
      img_size=256, denoise=True, P=16, R=2, uniform, stride=1, max_frames=10000,
      face_crop=True, grid_size=2, include_std=True
    """
    img_size: int = 256
    denoise: bool = True

    lbp_points: int = 16
    lbp_radius: int = 2
    lbp_method: str = "uniform"

    frame_stride: int = 1
    fps_sample: Optional[float] = None
    max_frames: int = 10000

    face_crop: bool = True
    face_margin: float = 0.25
    grid_size: int = 2

    include_std: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "FeatureConfig":
        if not d:
            return FeatureConfig()
        return FeatureConfig(
            img_size=int(d.get("img_size", 256)),
            denoise=bool(d.get("denoise", True)),
            lbp_points=int(d.get("lbp_points", 16)),
            lbp_radius=int(d.get("lbp_radius", 2)),
            lbp_method=str(d.get("lbp_method", "uniform")),
            frame_stride=int(d.get("frame_stride", 1)),
            fps_sample=d.get("fps_sample", None),
            max_frames=int(d.get("max_frames", 10000)),
            face_crop=bool(d.get("face_crop", True)),
            face_margin=float(d.get("face_margin", 0.25)),
            grid_size=int(d.get("grid_size", 2)),
            include_std=bool(d.get("include_std", True)),
        )


def lbp_num_bins(P: int, method: str) -> int:
    return (P + 2) if method == "uniform" else (2 ** P)


def feature_dim(cfg: FeatureConfig) -> int:
    base_bins = lbp_num_bins(cfg.lbp_points, cfg.lbp_method)
    grid_mul = (cfg.grid_size ** 2) if cfg.grid_size and cfg.grid_size > 1 else 1
    dim = base_bins * grid_mul
    if cfg.include_std:
        dim *= 2
    return dim


def preprocess_frame(frame_bgr: np.ndarray, img_size: int, denoise: bool) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.uint8)


_FACE_CASCADE = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def crop_face_or_center(gray: np.ndarray, margin: float = 0.25) -> np.ndarray:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = _get_face_cascade()

    h, w = gray.shape[:2]
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        mx = int(fw * margin)
        my = int(fh * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(w, x + fw + mx)
        y1 = min(h, y + fh + my)
        cropped = gray[y0:y1, x0:x1]
        if cropped.size > 0:
            return cropped

    # fallback: center crop 70%
    cw = int(w * 0.7)
    ch = int(h * 0.7)
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    cropped = gray[y0:y0 + ch, x0:x0 + cw]
    return cropped if cropped.size > 0 else gray


def lbp_histogram(gray: np.ndarray, P: int, R: int, method: str) -> np.ndarray:
    lbp = local_binary_pattern(gray, P, R, method=method)
    bins = lbp_num_bins(P, method)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-8)
    return hist


def lbp_histogram_grid(gray: np.ndarray, P: int, R: int, method: str, grid_size: int) -> np.ndarray:
    if grid_size <= 1:
        return lbp_histogram(gray, P, R, method)

    h, w = gray.shape[:2]
    gh = max(1, h // grid_size)
    gw = max(1, w // grid_size)

    feats: List[np.ndarray] = []
    for i in range(grid_size):
        for j in range(grid_size):
            y0 = i * gh
            x0 = j * gw
            y1 = h if i == grid_size - 1 else (i + 1) * gh
            x1 = w if j == grid_size - 1 else (j + 1) * gw
            patch = gray[y0:y1, x0:x1]
            feats.append(lbp_histogram(patch, P, R, method))

    return np.concatenate(feats, axis=0).astype(np.float32)


def _compute_stride(cap: cv2.VideoCapture, frame_stride: int, fps_sample: Optional[float]) -> int:
    if fps_sample is None:
        return max(1, int(frame_stride))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0.0:
        return max(1, int(frame_stride))
    stride = int(round(fps / float(fps_sample)))
    return max(1, stride)


def extract_video_features(
    video_path: Path,
    cfg: FeatureConfig,
    logger_: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], int]:
    logger_ = logger_ or logger
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger_.warning("Cannot open video (skip): %s", video_path)
        return None, 0

    stride = _compute_stride(cap, cfg.frame_stride, cfg.fps_sample)
    per_frame_len = lbp_num_bins(cfg.lbp_points, cfg.lbp_method) * ((cfg.grid_size ** 2) if cfg.grid_size > 1 else 1)

    hists: List[np.ndarray] = []
    frames_used = 0
    idx = 0

    try:
        while frames_used < cfg.max_frames:
            grabbed = cap.grab()
            if not grabbed:
                break

            if idx % stride == 0:
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    idx += 1
                    continue

                gray = preprocess_frame(frame, cfg.img_size, cfg.denoise)

                if cfg.face_crop:
                    cropped = crop_face_or_center(gray, margin=cfg.face_margin)
                    gray = cv2.resize(cropped, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)

                hist = lbp_histogram_grid(gray, cfg.lbp_points, cfg.lbp_radius, cfg.lbp_method, cfg.grid_size)
                if hist.shape[0] != per_frame_len:
                    # safety
                    hist = hist[:per_frame_len] if hist.shape[0] > per_frame_len else np.pad(hist, (0, per_frame_len - hist.shape[0]))
                hists.append(hist.astype(np.float32))
                frames_used += 1

            idx += 1
    except Exception as e:
        logger_.exception("Feature extraction error on %s: %s", video_path, e)
        return None, 0
    finally:
        cap.release()

    if frames_used == 0:
        logger_.warning("No frames extracted (skip): %s", video_path)
        return None, 0

    H = np.stack(hists, axis=0)
    mean_vec = H.mean(axis=0).astype(np.float32)

    if cfg.include_std:
        std_vec = H.std(axis=0).astype(np.float32)
        feat = np.concatenate([mean_vec, std_vec], axis=0)
    else:
        feat = mean_vec

    # final safety
    dim = feature_dim(cfg)
    if feat.shape[0] != dim:
        feat = feat[:dim] if feat.shape[0] > dim else np.pad(feat, (0, dim - feat.shape[0]))

    return feat.astype(np.float32), int(frames_used)


def extract_dataset_features(
    samples: List[VideoSample],
    cfg: FeatureConfig,
    out_csv: Path,
    logger_: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger_ = logger_ or logger
    out_csv = out_csv.expanduser().resolve()
    ensure_dir(out_csv.parent)

    dim = feature_dim(cfg)
    logger_.info("Feature config: %s | feature_dim=%d", cfg.to_dict(), dim)

    rows: List[Dict] = []
    skipped = 0

    for s in tqdm(samples, desc="Extracting LBP features", unit="video"):
        feat, used = extract_video_features(s.path, cfg, logger_=logger_)
        if feat is None:
            skipped += 1
            continue

        row: Dict[str, object] = {
            "video_path": str(s.path),
            "label": s.label,
            "frames_used": int(used),
        }
        for i in range(dim):
            row[f"feat_{i}"] = float(feat[i])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No features extracted. Check dataset/codecs.")

    df.to_csv(out_csv, index=False)
    logger_.info("Saved features: %s (n=%d, skipped=%d)", out_csv, len(df), skipped)
    return df


def load_features_csv(csv_path: Path, use_frames_used: bool) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      df, X, y, input_columns (order)
    If use_frames_used=True, X includes 'frames_used' + feat_*.
    Otherwise only feat_*.
    """
    df = pd.read_csv(csv_path)

    if "video_path" not in df.columns or "label" not in df.columns:
        raise ValueError("features.csv must contain 'video_path' and 'label'.")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    feat_cols = sorted(feat_cols, key=lambda c: int(c.split("_")[1]))

    input_cols: List[str] = []
    if use_frames_used:
        if "frames_used" not in df.columns:
            raise ValueError("use_frames_used=True but 'frames_used' column missing.")
        input_cols.append("frames_used")
    input_cols.extend(feat_cols)

    X = df[input_cols].to_numpy(dtype=np.float32)
    y = df["label"].astype(str).to_numpy()
    return df, X, y, input_cols
