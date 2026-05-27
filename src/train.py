from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import load_samples, parse_class_dir_map
from .features import FeatureConfig, extract_dataset_features, load_features_csv
from .utils import ensure_dir, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train LBP+SVM AI-generated video detector (SDFVD2 best_v3 consistent)."
    )

    # IO
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--labels_csv", type=str, default=None)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--class_dir_map", type=str, default="real=real,not_real=not_real")

    # Cache
    p.add_argument("--reuse_features", action="store_true")
    p.add_argument(
        "--ignore_saved_config",
        action="store_true",
        help="If set, do NOT load out_dir/config.json when reusing features.",
    )
    p.add_argument(
        "--extract_only",
        action="store_true",
        help="Only extract and save features/config, then exit without training.",
    )

    # Feature params
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--fps_sample", type=float, default=None)
    p.add_argument("--max_frames", type=int, default=10000)

    p.add_argument("--img_size", type=int, default=256)

    # Boolean flags with explicit off switches (biar log tidak misleading)
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    p.add_argument("--no_denoise", dest="denoise", action="store_false")

    p.add_argument("--lbp_points", type=int, default=16)
    p.add_argument("--lbp_radius", type=int, default=2)
    p.add_argument("--lbp_method", type=str, default="uniform")

    p.add_argument("--face_crop", dest="face_crop", action="store_true", default=True)
    p.add_argument("--no_face_crop", dest="face_crop", action="store_false")
    p.add_argument("--face_margin", type=float, default=0.25)

    p.add_argument("--grid_size", type=int, default=2)

    p.add_argument("--include_std", dest="include_std", action="store_true", default=True)
    p.add_argument("--no_include_std", dest="include_std", action="store_false")

    # Split (match notebook-like by default)
    p.add_argument("--split_mode", choices=["random", "group"], default="random")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)

    p.add_argument("--stratify", dest="stratify", action="store_true", default=False)
    p.add_argument("--no_stratify", dest="stratify", action="store_false")

    # Include frames_used as feature (DEFAULT OFF — bukan fitur visual)
    p.add_argument("--use_frames_used", dest="use_frames_used", action="store_true", default=False)
    p.add_argument("--no_frames_used", dest="use_frames_used", action="store_false")

    # SVM defaults (match your good run)
    p.add_argument("--svm_kernel", choices=["linear", "rbf", "poly"], default="linear")
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--gamma", type=str, default="scale")  # rbf/poly
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--class_weight", choices=["balanced", "none"], default="balanced")

    return p


def _parse_gamma(gamma_str: str) -> object:
    try:
        return float(gamma_str)
    except (TypeError, ValueError):
        return gamma_str


def base_group_id(video_path: str) -> str:
    """
    Group id untuk SDFVD2:
      v10, real_v10_aug_*, vs10, fake_vs10_aug_* -> group "10"
    """
    stem = Path(video_path).stem.lower()
    stem = re.sub(r"^(real_|fake_)", "", stem)
    if "_aug_" in stem:
        stem = stem.split("_aug_", 1)[0]

    m = re.fullmatch(r"vs(\d+)", stem)
    if m:
        return m.group(1)
    m = re.fullmatch(r"v(\d+)", stem)
    if m:
        return m.group(1)
    return stem


def _load_saved_config_if_any(out_dir: Path, log: logging.Logger) -> Optional[FeatureConfig]:
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        return None
    try:
        d = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg = FeatureConfig.from_dict(d)
        log.info("Loaded saved feature config from %s", cfg_path)
        return cfg
    except Exception as e:
        log.warning("Failed to read saved config.json (%s). Will use CLI config. Error: %s", cfg_path, e)
        return None


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    out_dir = ensure_dir(Path(args.out_dir).expanduser().resolve())
    setup_logging(out_dir=out_dir, level=logging.INFO, log_name="train.log")
    log = logging.getLogger("train")

    log.info("Args: %s", vars(args))

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve() if args.labels_csv else None
    class_dir_map = parse_class_dir_map(args.class_dir_map) if not labels_csv else {}

    samples = load_samples(
        dataset_root=dataset_root,
        labels_csv=labels_csv,
        class_dir_map=class_dir_map if not labels_csv else None,
        logger=log,
    )
    if not samples:
        raise RuntimeError("No samples found.")

    features_csv = out_dir / "features.csv"

    # Decide feature config (CRITICAL for consistency)
    saved_cfg = None
    if args.reuse_features and features_csv.exists() and (not args.ignore_saved_config):
        saved_cfg = _load_saved_config_if_any(out_dir, log)

    if saved_cfg is not None:
        cfg = saved_cfg
    else:
        # use CLI config
        cfg = FeatureConfig(
            img_size=args.img_size,
            denoise=bool(args.denoise),
            lbp_points=args.lbp_points,
            lbp_radius=args.lbp_radius,
            lbp_method=args.lbp_method,
            frame_stride=args.frame_stride,
            fps_sample=args.fps_sample,
            max_frames=args.max_frames,
            face_crop=bool(args.face_crop),
            face_margin=float(args.face_margin),
            grid_size=int(args.grid_size),
            include_std=bool(args.include_std),
        )

    log.info("Using feature config: %s", cfg.to_dict())

    # Feature extraction
    if args.reuse_features and features_csv.exists():
        log.info("Reusing features: %s", features_csv)
    else:
        log.info("Extracting features -> %s", features_csv)
        extract_dataset_features(samples, cfg, features_csv, logger_=log)
        # Save config that matches these features
        (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    if args.extract_only:
        log.info("extract_only=True. Skipping split, training, and evaluation.")
        log.info("Saved features: %s", features_csv)
        log.info("Saved config: %s", out_dir / "config.json")
        log.info("DONE (extract only).")
        return

    # Load feature matrix
    df, X, y, input_cols = load_features_csv(features_csv, use_frames_used=bool(args.use_frames_used))
    log.info("Training inputs: use_frames_used=%s | n_samples=%d | n_features=%d",
             args.use_frames_used, X.shape[0], X.shape[1])

    # Split
    if args.split_mode == "group":
        groups = np.array([base_group_id(vp) for vp in df["video_path"].tolist()])
        gss = GroupShuffleSplit(n_splits=1, test_size=float(args.test_size), random_state=int(args.random_state))
        train_idx, test_idx = next(gss.split(X, y, groups=groups))
        log.info("Split mode=group: train=%d test=%d", len(train_idx), len(test_idx))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_df = df.iloc[test_idx].copy()
    else:
        strat = y if args.stratify else None
        idx_all = np.arange(len(y))
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, idx_all,
            test_size=float(args.test_size),
            random_state=int(args.random_state),
            stratify=strat,
        )
        log.info("Split mode=random%s: train=%d test=%d",
                 " (stratified)" if args.stratify else " (non-stratified / notebook-like)",
                 len(y_train), len(y_test))
        test_df = df.iloc[idx_test].copy()

    # Train model
    class_weight_val = None if args.class_weight == "none" else "balanced"
    gamma_val = _parse_gamma(args.gamma)

    svm = SVC(
        kernel=args.svm_kernel,
        C=float(args.C),
        gamma=gamma_val,
        degree=int(args.degree),
        probability=True,
        class_weight=class_weight_val,
        random_state=int(args.random_state),
    )
    model = Pipeline([("scaler", StandardScaler()), ("svm", svm)])

    log.info(
        "Training SVM: kernel=%s, C=%.4f, gamma=%s, class_weight=%s",
        args.svm_kernel, args.C, str(gamma_val), str(class_weight_val)
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=["real", "not_real"])
    report = classification_report(y_test, y_pred, digits=4)

    log.info("Confusion matrix (labels=[real, not_real]):\n%s", cm)
    log.info("Classification report:\n%s", report)

    # Save bundle (include input_columns for predict)
    bundle: Dict[str, Any] = {
        "model": model,
        "feature_config": cfg.to_dict(),
        "input_columns": input_cols,
        "split": {
            "split_mode": args.split_mode,
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "stratify": bool(args.stratify),
            "use_frames_used": bool(args.use_frames_used),
        },
        "svm_params": {
            "kernel": args.svm_kernel,
            "C": float(args.C),
            "gamma": gamma_val,
            "degree": int(args.degree),
            "class_weight": class_weight_val,
        },
    }
    joblib.dump(bundle, out_dir / "model.joblib")

    # Save test predictions
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)
        cls = list(getattr(model, "classes_", ["real", "not_real"]))
        prob_map = {c: probs[:, i] for i, c in enumerate(cls)}

        test_df["y_true"] = y_test
        test_df["y_pred"] = y_pred
        if "not_real" in prob_map:
            test_df["prob_not_real"] = prob_map["not_real"]
        test_df.to_csv(out_dir / "test_predictions.csv", index=False)

    log.info("Saved model: %s", out_dir / "model.joblib")
    log.info("Saved test predictions: %s", out_dir / "test_predictions.csv")
    log.info("DONE.")


if __name__ == "__main__":
    main()
