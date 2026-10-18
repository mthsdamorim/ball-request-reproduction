"""Apply frozen simulation models to real windows without fitting on their labels."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from ball_request_lab.motion import MOTION_CLASSES
from ball_request_lab.motion_models import (
    load_ridge,
    load_tcn_checkpoint,
    normalize_raw_windows,
    predict_on_raw_windows,
    raised_arm_baseline,
    ridge_probabilities,
    temporal_ridge_features,
)
from ball_request_lab.util import atomic_json, file_hash


def summarize_real_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Show case counts; no recall for classes absent from the eligible labels."""
    labeled = [
        r for r in rows if r["evaluable_identity"] is True and r["gesture_type"] in MOTION_CLASSES
    ]
    support = {name: sum(r["gesture_type"] == name for r in labeled) for name in MOTION_CLASSES}
    models: dict[str, Any] = {}
    for model in ("geometry", "ridge", "tcn"):
        assessed = [r for r in labeled if r[model] != "abstain"]
        label_key = "raised_label" if model == "geometry" else "gesture_type"
        models[model] = {
            "evaluated": len(assessed),
            "abstained": len(labeled) - len(assessed),
            "correct": sum(r[model] == r[label_key] for r in assessed),
            "unit": "visually assessable sampled player-window; not an event rate",
        }
    return {
        "selected": len(rows),
        "identity_and_label_evaluable": len(labeled),
        "excluded": len(rows) - len(labeled),
        "observed_class_support": support,
        "can_estimate_raised_recall": support["one_arm_raise"] + support["two_arm_raise"] > 0,
        "models": models,
        "claim_limit": "Descriptive counts for a small visually reviewed sample. "
        "No overall five-class F1, population frequency, external-match validation "
        "or request-intent claim. Absent classes have no measurable recall.",
    }


def evaluate_real_windows(ml_run: Path, real_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("Transfer report exists; choose another output path")
    selection_path, annotations_path = (
        real_run / "selection.json",
        real_run / "annotations.json",
    )
    selection = json.loads(selection_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    ids = [c["case_id"] for c in selection["cases"]]
    label_ids = [a["case_id"] for a in annotations]
    if len(set(label_ids)) != len(label_ids) or set(ids) != set(label_ids):
        raise ValueError("Annotation IDs must match the frozen selection exactly")
    by_id = {a["case_id"]: a for a in annotations}
    windows_path = real_run / "windows.npz"
    if file_hash(windows_path) != selection["windows_sha256"]:
        raise ValueError("Real pose tensors differ from frozen selection")
    with np.load(windows_path, allow_pickle=False) as arrays:
        if arrays["case_ids"].tolist() != ids:
            raise ValueError("Pose tensor ordering differs from selection")
        raw, errors = arrays["raw_windows_m"], arrays["errors_cm"]
    metrics = json.loads((ml_run / "metrics.json").read_text())
    config = tomllib.loads((ml_run / "config.toml").read_text())
    max_error = float(config["motion"]["max_joint_error_cm"])
    checkpoint = ml_run / metrics["chosen_checkpoint"]
    model_hashes = {
        str(p.relative_to(ml_run)): file_hash(p)
        for p in (
            checkpoint,
            ml_run / "models/ridge.npz",
            ml_run / "models/ridge.json",
            ml_run / "config.toml",
            ml_run / "metrics.json",
        )
    }
    tcn = load_tcn_checkpoint(checkpoint)
    neural = predict_on_raw_windows(tcn, raw, errors, max_error)
    ridge = load_ridge(ml_run / "models")
    normalized = normalize_raw_windows(raw, errors, max_error)
    ridge_scores = ridge_probabilities(ridge, temporal_ridge_features(normalized.values))
    geometry = raised_arm_baseline(raw, errors)
    # Use exactly the frozen experiment's observation threshold.
    minimum = float(config["model"]["abstain_min_valid_fraction"])
    valid_fraction = normalized.valid.mean(axis=(1, 2))
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(selection["cases"]):
        label = by_id[case["case_id"]]
        gesture = label["gesture_type"]
        is_raised = gesture in ("one_arm_raise", "two_arm_raise")
        rows.append(
            {
                "case_id": case["case_id"],
                "match_id": case["match_id"],
                "gesture_type": gesture,
                "evaluable_identity": label["evaluable_identity"] is True,
                "identity_confidence": label["identity_confidence"],
                "raised_label": "raised" if is_raised else "other",
                "valid_fraction": float(valid_fraction[i]),
                "geometry": "abstain"
                if geometry[i] < 0
                else ("raised" if geometry[i] else "other"),
                "ridge": "abstain"
                if valid_fraction[i] < minimum
                else MOTION_CLASSES[int(ridge_scores[i].argmax())],
                "tcn": "abstain"
                if valid_fraction[i] < minimum
                else MOTION_CLASSES[int(neural["predicted"][i])],
                "ridge_softmax_scores": ridge_scores[i].tolist(),
                "tcn_softmax_scores": neural["probabilities"][i].tolist(),
                "scores_limit": "uncalibrated model outputs, "
                "not probabilities of requesting a pass",
            }
        )
    result = {
        "summary": summarize_real_predictions(rows),
        "cases": rows,
        "provenance": {
            "model_files_sha256": model_hashes,
            "selection_sha256": file_hash(selection_path),
            "annotations_sha256": file_hash(annotations_path),
            "windows_sha256": file_hash(windows_path),
            "evaluator_sha256": file_hash(Path(__file__)),
            "no_fitting_or_threshold_tuning": True,
        },
    }
    if any(file_hash(ml_run / name) != digest for name, digest in model_hashes.items()):
        raise RuntimeError("Frozen model files changed during evaluation")
    atomic_json(output, result)
    return result
