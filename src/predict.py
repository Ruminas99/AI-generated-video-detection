from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from .features import FeatureConfig, extract_video_features
from .utils import pretty_label, setup_logging


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict label for 1 video using trained model.joblib")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--video", type=str, required=True)
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    setup_logging(out_dir=None, level=logging.INFO)
    log = logging.getLogger("predict")

    model_path = Path(args.model).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()

    bundle = joblib.load(model_path)
    model = bundle["model"]
    cfg = FeatureConfig.from_dict(bundle.get("feature_config", {}) or {})
    input_cols = bundle.get("input_columns", [])

    feat, frames_used = extract_video_features(video_path, cfg, logger_=log)
    if feat is None:
        raise RuntimeError(f"Failed to extract features from {video_path}")

    # build X in the SAME order as training
    vec_parts = []
    for c in input_cols:
        if c == "frames_used":
            vec_parts.append(np.array([float(frames_used)], dtype=np.float32))
        elif c.startswith("feat_"):
            # feat_0..feat_n in order
            idx = int(c.split("_")[1])
            vec_parts.append(np.array([float(feat[idx])], dtype=np.float32))
        else:
            raise ValueError(f"Unknown input column in model bundle: {c}")

    X = np.concatenate(vec_parts, axis=0).reshape(1, -1)

    pred = model.predict(X)[0]
    log.info("Video: %s", video_path)
    log.info("Frames used: %d", frames_used)
    log.info("Predicted: %s", pretty_label(str(pred)))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is not None:
            probs_str = ", ".join([f"{pretty_label(str(c))}={float(p):.4f}" for c, p in zip(classes, proba)])
            log.info("Probabilities: %s", probs_str)

    print(pretty_label(str(pred)))


if __name__ == "__main__":
    main()
