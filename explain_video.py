from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern

# Import extractor yang sama persis dari project kamu
from src.features import FeatureConfig, feature_dim, lbp_num_bins, preprocess_frame, crop_face_or_center
from src.utils import setup_logging, ensure_dir


def normpath_key(p: str) -> str:
    """Normalize path for matching (case-insensitive, slash-insensitive)."""
    return os.path.normcase(os.path.normpath(str(p).strip().strip('"').strip("'")))


def load_bundle(model_path: Path) -> Dict:
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "model" not in obj:
        raise ValueError("model.joblib must be a dict bundle with key 'model'.")
    return obj


def load_cfg_from_outdir_or_bundle(out_dir: Path, bundle: Dict) -> FeatureConfig:
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        d = json.loads(cfg_path.read_text(encoding="utf-8"))
        return FeatureConfig.from_dict(d)
    return FeatureConfig.from_dict(bundle.get("feature_config", {}) or {})


def load_cached_feature_row(features_csv: Path, video_path: Path) -> Optional[pd.Series]:
    if not features_csv.exists():
        return None
    df = pd.read_csv(features_csv)

    if "video_path" not in df.columns:
        return None

    key_target = normpath_key(str(video_path.resolve()))

    # exact normalized match
    keys = df["video_path"].astype(str).map(normpath_key)
    m = keys == key_target
    if m.any():
        return df[m].iloc[0]

    # fallback by basename (warning: can be ambiguous)
    base = video_path.name.lower()
    m2 = df["video_path"].astype(str).map(lambda s: Path(str(s)).name.lower()) == base
    if m2.sum() == 1:
        return df[m2].iloc[0]
    return None


def grid_slices(h: int, w: int, g: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Return list of (cell_id, y0, y1, x0, x1) in row-major order.
    Matches src.features.lbp_histogram_grid() ordering.
    """
    if g <= 1:
        return [(0, 0, h, 0, w)]
    gh = max(1, h // g)
    gw = max(1, w // g)
    out = []
    cell_id = 0
    for i in range(g):
        for j in range(g):
            y0 = i * gh
            x0 = j * gw
            y1 = h if i == g - 1 else (i + 1) * gh
            x1 = w if j == g - 1 else (j + 1) * gw
            out.append((cell_id, y0, y1, x0, x1, i, j))
            cell_id += 1
    return out


def lbp_hist_counts_and_norm(patch_gray: np.ndarray, P: int, R: int, method: str) -> Tuple[np.ndarray, np.ndarray]:
    lbp = local_binary_pattern(patch_gray, P, R, method=method)
    bins = lbp_num_bins(P, method)
    counts, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    counts = counts.astype(np.int64)
    norm = counts.astype(np.float32)
    norm /= (norm.sum() + 1e-8)
    return counts, norm


def describe_base_feature(idx: int, cfg: FeatureConfig) -> str:
    """
    Describe base (per-frame) feature index (0..D-1) where D=bins*grid^2
    """
    bins = lbp_num_bins(cfg.lbp_points, cfg.lbp_method)
    g = int(cfg.grid_size)
    cell_id = idx // bins
    bin_id = idx % bins
    r = cell_id // g if g > 1 else 0
    c = cell_id % g if g > 1 else 0
    # interpret uniform bins (rough meaning)
    if cfg.lbp_method == "uniform":
        if bin_id == bins - 1:
            meaning = "non-uniform"
        else:
            meaning = f"uniform (~{bin_id} ones)"
    else:
        meaning = "pattern-id"
    return f"grid({r},{c}) bin={bin_id} [{meaning}]"


def describe_final_feature(idx: int, cfg: FeatureConfig) -> str:
    """
    Describe final video-level feature index.
    If include_std=True, first half is mean, second half is std.
    """
    base_dim = lbp_num_bins(cfg.lbp_points, cfg.lbp_method) * ((cfg.grid_size ** 2) if cfg.grid_size > 1 else 1)
    if cfg.include_std:
        if idx < base_dim:
            return "mean " + describe_base_feature(idx, cfg)
        return "std  " + describe_base_feature(idx - base_dim, cfg)
    return "mean " + describe_base_feature(idx, cfg)


def extract_video_debug(video_path: Path, cfg: FeatureConfig, out_dir: Path, log: logging.Logger) -> Tuple[np.ndarray, int]:
    """
    Extract features for one video AND dump all intermediate numbers to out_dir.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Determine stride (same logic as src.features)
    stride = max(1, int(cfg.frame_stride))
    if cfg.fps_sample is not None:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0.0 and not math.isnan(fps):
            stride = max(1, int(round(fps / float(cfg.fps_sample))))

    bins = lbp_num_bins(cfg.lbp_points, cfg.lbp_method)
    g = int(cfg.grid_size)
    base_dim = bins * ((g ** 2) if g > 1 else 1)

    frame_rows = []
    long_rows = []  # per frame per cell per bin
    frame_vecs = []  # list of (frame_no, vec72)
    used_frames = 0
    frame_no = 0

    while used_frames < int(cfg.max_frames):
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_no % stride == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                frame_no += 1
                continue

            gray = preprocess_frame(frame, cfg.img_size, cfg.denoise)

            face_used = False
            if cfg.face_crop:
                cropped = crop_face_or_center(gray, margin=float(cfg.face_margin))
                gray = cv2.resize(cropped, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_AREA)
                face_used = True

            h, w = gray.shape[:2]
            slices = grid_slices(h, w, g)

            vec_parts = []
            # collect per-cell hist
            for (cell_id, y0, y1, x0, x1, rr, cc) in slices:
                patch = gray[y0:y1, x0:x1]
                counts, norm = lbp_hist_counts_and_norm(patch, cfg.lbp_points, cfg.lbp_radius, cfg.lbp_method)

                # dump per-bin
                for b in range(bins):
                    long_rows.append({
                        "frame_index": used_frames,
                        "video_frame_no": frame_no,
                        "cell_id": cell_id,
                        "cell_row": rr,
                        "cell_col": cc,
                        "bin": b,
                        "count": int(counts[b]),
                        "value_norm": float(norm[b]),
                    })

                vec_parts.append(norm)

            frame_vec = np.concatenate(vec_parts, axis=0).astype(np.float32)
            if frame_vec.shape[0] != base_dim:
                raise RuntimeError(f"Unexpected frame feature dim: got {frame_vec.shape[0]}, expected {base_dim}")

            frame_vecs.append(frame_vec)
            frame_rows.append({
                "frame_index": used_frames,
                "video_frame_no": frame_no,
                "used_stride": stride,
                "face_crop_applied": bool(face_used),
            })

            used_frames += 1

        frame_no += 1

    cap.release()

    if used_frames == 0:
        raise RuntimeError("No frames used. Check video/codec.")

    # Save per-frame metadata
    df_frames = pd.DataFrame(frame_rows)
    df_frames.to_csv(out_dir / "frames_used.csv", index=False)

    # Save per-frame histogram long format
    df_long = pd.DataFrame(long_rows)
    df_long.to_csv(out_dir / "frame_histograms_long.csv", index=False)

    # Stack and aggregate
    H = np.stack(frame_vecs, axis=0)  # (T, base_dim)
    mean_vec = H.mean(axis=0).astype(np.float32)
    std_vec = H.std(axis=0).astype(np.float32)

    if cfg.include_std:
        video_feat = np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)
    else:
        video_feat = mean_vec

    # Save base feature table (mean/std)
    base_table = []
    for i in range(base_dim):
        base_table.append({
            "base_feature_index": i,
            "desc": describe_base_feature(i, cfg),
            "mean": float(mean_vec[i]),
            "std": float(std_vec[i]),
        })
    pd.DataFrame(base_table).to_csv(out_dir / "video_base_features_mean_std.csv", index=False)

    # Save final video feature vector
    final_table = []
    for i in range(video_feat.shape[0]):
        final_table.append({
            "feature_index": i,
            "desc": describe_final_feature(i, cfg),
            "value": float(video_feat[i]),
        })
    pd.DataFrame(final_table).to_csv(out_dir / "video_feature_vector.csv", index=False)

    # Save per-frame feature vectors (wide, optional)
    # (kalau mau benar-benar "lihat angka 72 per frame", ini paling jelas)
    df_framevec = pd.DataFrame(H)
    df_framevec.insert(0, "frame_index", np.arange(H.shape[0]))
    df_framevec.to_csv(out_dir / "frame_feature_vectors_72.csv", index=False)

    return video_feat, used_frames


def explain_model_steps(
    bundle: Dict,
    video_feat: np.ndarray,
    frames_used: int,
    out_dir: Path,
    log: logging.Logger,
) -> None:
    model = bundle["model"]
    scaler = model.named_steps["scaler"]
    svm = model.named_steps["svm"]
    input_cols: List[str] = bundle.get("input_columns", [])

    # Build raw input vector in exactly the same order as training
    parts = []
    for c in input_cols:
        if c == "frames_used":
            parts.append(np.array([float(frames_used)], dtype=np.float32))
        elif c.startswith("feat_"):
            idx = int(c.split("_")[1])
            parts.append(np.array([float(video_feat[idx])], dtype=np.float32))
        else:
            raise ValueError(f"Unknown input column in model bundle: {c}")

    x_raw = np.concatenate(parts, axis=0).astype(np.float32)
    x_raw_2d = x_raw.reshape(1, -1)

    # Manual scaler
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    z = (x_raw - mean) / scale

    # Decision + proba
    decision = float(svm.decision_function(z.reshape(1, -1))[0])
    pred = model.predict(x_raw_2d)[0]
    proba = model.predict_proba(x_raw_2d)[0]
    classes = list(getattr(model, "classes_", ["real", "not_real"]))

    # Dump pipeline table
    rows = []
    weights = None
    if hasattr(svm, "coef_") and svm.kernel == "linear":
        weights = svm.coef_[0].astype(np.float32)
        b = float(svm.intercept_[0])
    else:
        b = None

    for i, name in enumerate(input_cols):
        row = {
            "input_index": i,
            "input_name": name,
            "raw_value": float(x_raw[i]),
            "scaler_mean": float(mean[i]),
            "scaler_scale": float(scale[i]),
            "z": float(z[i]),
        }
        if weights is not None:
            row["svm_weight"] = float(weights[i])
            row["contribution_wz"] = float(weights[i] * z[i])
        rows.append(row)

    df_pipe = pd.DataFrame(rows)
    df_pipe.to_csv(out_dir / "pipeline_input_scaler_svm.csv", index=False)

    if weights is not None:
        df_top = df_pipe.copy()
        df_top["abs_contribution"] = df_top["contribution_wz"].abs()
        df_top = df_top.sort_values("abs_contribution", ascending=False).head(30)
        df_top.to_csv(out_dir / "top30_feature_contributions.csv", index=False)
        log.info("Saved top 30 contributions: %s", out_dir / "top30_feature_contributions.csv")

    result = {
        "decision_function": decision,
        "svm_intercept": b,
        "classes": classes,
        "probabilities": {classes[i]: float(proba[i]) for i in range(len(classes))},
        "predicted_label": str(pred),
    }
    (out_dir / "prediction_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Explain (fully) how one video -> LBP features -> scaler -> SVM prediction.")
    ap.add_argument("--video", type=str, required=True, help="Path to one video from your dataset.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output folder for explanation artifacts.")
    ap.add_argument("--model", type=str, default="outputs/sdfvd2_best_v3/model.joblib")
    ap.add_argument("--features_csv", type=str, default="outputs/sdfvd2_best_v3/features.csv")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    setup_logging(out_dir=out_dir, level=logging.INFO, log_name="explain.log")
    log = logging.getLogger("explain")

    video_path = Path(args.video).resolve()
    model_path = Path(args.model).resolve()
    features_csv = Path(args.features_csv).resolve()

    log.info("Video: %s", video_path)
    log.info("Model: %s", model_path)
    log.info("Features CSV: %s", features_csv)

    bundle = load_bundle(model_path)
    cfg = load_cfg_from_outdir_or_bundle(model_path.parent, bundle)
    log.info("Using feature config: %s", cfg.to_dict())

    # 1) extract (and dump intermediate numbers)
    video_feat, used = extract_video_debug(video_path, cfg, out_dir, log)
    log.info("Frames used: %d", used)
    log.info("Video feature dim: %d", int(video_feat.shape[0]))

    # 2) compare with cached features.csv (if row exists)
    cached = load_cached_feature_row(features_csv, video_path)
    if cached is not None:
        # reconstruct cached feature vector in the same order feat_0..feat_n
        feat_cols = [c for c in cached.index if str(c).startswith("feat_")]
        feat_cols = sorted(feat_cols, key=lambda c: int(str(c).split("_")[1]))
        cached_vec = np.array([float(cached[c]) for c in feat_cols], dtype=np.float32)

        # Compare (truncate to common length)
        m = min(len(cached_vec), len(video_feat))
        diff = cached_vec[:m] - video_feat[:m]
        max_abs = float(np.max(np.abs(diff)))
        mse = float(np.mean(diff * diff))
        log.info("Cache match found in features.csv. max_abs_diff=%.8f | mse=%.10f", max_abs, mse)

        (out_dir / "cache_compare.json").write_text(
            json.dumps({"max_abs_diff": max_abs, "mse": mse, "cached_len": int(len(cached_vec)), "computed_len": int(len(video_feat))}, indent=2),
            encoding="utf-8",
        )
    else:
        log.warning("No matching row found in features.csv for this video path (still OK).")

    # 3) explain scaler + svm steps
    explain_model_steps(bundle, video_feat, used, out_dir, log)
    log.info("DONE. See output folder: %s", out_dir)


if __name__ == "__main__":
    main()
