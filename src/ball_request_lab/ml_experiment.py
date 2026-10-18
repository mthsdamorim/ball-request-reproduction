"""Reproducible synthetic motion-recognition positive control.

This module deliberately evaluates a wholly procedural simulation.  Its labels
are movement families authored by the generator, not football intent or an
estimate of an in-match event rate.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ball_request_lab.motion import (
    CLASS_INDEX,
    MOTION_CLASSES,
    MotionConfig,
    SyntheticMotions,
    generate_synthetic_motions,
    segment_lengths,
)
from ball_request_lab.motion_models import (
    TcnTraining,
    fit_ridge_classifier,
    load_ridge,
    load_tcn_checkpoint,
    model_input,
    normalize_raw_windows,
    raised_arm_baseline,
    ridge_payload,
    ridge_probabilities,
    save_tcn_checkpoint,
    tcn_probabilities,
    temporal_ridge_features,
    train_tcn,
)

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ExperimentConfig:
    motion: MotionConfig
    ridge_penalty: float = 1.0
    tcn_epochs: int = 25
    tcn_batch_size: int = 64
    tcn_learning_rate: float = 0.003
    tcn_seeds: tuple[int, ...] = (7, 17, 27)
    abstain_min_valid_fraction: float = 0.15

    def __post_init__(self) -> None:
        if self.tcn_epochs < 1 or self.tcn_batch_size < 1 or self.tcn_learning_rate <= 0:
            raise ValueError("Invalid TCN training configuration")
        if not self.tcn_seeds:
            raise ValueError("At least one TCN seed is required")
        if not 0.0 <= self.abstain_min_valid_fraction <= 1.0:
            raise ValueError("Invalid abstention fraction")


def load_config(path: Path) -> ExperimentConfig:
    """Load a small frozen TOML configuration without hidden defaults."""
    parsed = tomllib.loads(path.read_text())
    motion_values = dict(parsed["motion"])
    model = dict(parsed["model"])
    seeds = tuple(int(value) for value in model["tcn_seeds"])
    return ExperimentConfig(
        motion=MotionConfig(**motion_values),
        ridge_penalty=float(model["ridge_penalty"]),
        tcn_epochs=int(model["tcn_epochs"]),
        tcn_batch_size=int(model["tcn_batch_size"]),
        tcn_learning_rate=float(model["tcn_learning_rate"]),
        tcn_seeds=seeds,
        abstain_min_valid_fraction=float(model["abstain_min_valid_fraction"]),
    )


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_hashes() -> dict[str, str]:
    here = Path(__file__).parent
    return {
        name: hashlib.sha256((here / name).read_bytes()).hexdigest()
        for name in ("motion.py", "motion_models.py", "ml_experiment.py")
    }


def _split_mask(data: SyntheticMotions, split: str) -> NDArray[np.bool_]:
    return data.splits == split


def _training_mask(data: SyntheticMotions, split: str) -> NDArray[np.bool_]:
    return _split_mask(data, split) & np.isin(data.profiles, ("clean", "medium"))


def _metrics(
    labels: IntArray, predictions: IntArray, class_names: tuple[str, ...]
) -> dict[str, Any]:
    """Report coverage explicitly: abstentions never become negative examples."""
    known = predictions >= 0
    labels_known = labels[known]
    predictions_known = predictions[known]
    count = len(class_names)
    confusion = np.zeros((count, count), dtype=np.int64)
    if len(labels_known):
        np.add.at(confusion, (labels_known, predictions_known), 1)
    recalls = np.divide(
        np.diag(confusion),
        confusion.sum(axis=1),
        out=np.zeros(count),
        where=confusion.sum(axis=1) > 0,
    )
    precisions = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(count),
        where=confusion.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2.0 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(count),
        where=(precisions + recalls) > 0,
    )
    return {
        "total": len(labels),
        "evaluated": int(known.sum()),
        "abstained": int((~known).sum()),
        "coverage": float(known.mean()) if len(known) else 0.0,
        "accuracy": float((labels_known == predictions_known).mean())
        if len(labels_known)
        else None,
        "balanced_accuracy": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "confusion": confusion.tolist(),
        "per_class": {
            name: {
                "support": int(confusion[index].sum()),
                "recall": float(recalls[index]),
                "precision": float(precisions[index]),
                "f1": float(f1[index]),
            }
            for index, name in enumerate(class_names)
        },
    }


def _with_abstention(
    probabilities: FloatArray, valid_fraction: FloatArray, minimum: float
) -> IntArray:
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    predictions[valid_fraction < minimum] = -1
    return predictions


def _profile_metrics(
    labels: IntArray,
    profiles: NDArray[np.str_],
    predictions: IntArray,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in ("clean", "noisy", "occluded"):
        local = profiles == profile
        result[profile] = _metrics(labels[local], predictions[local], class_names)
    return result


def _binary_predictions(
    probabilities: FloatArray, valid_fraction: FloatArray, minimum: float
) -> IntArray:
    raised_probability = probabilities[
        :, (CLASS_INDEX["one_arm_raise"], CLASS_INDEX["two_arm_raise"])
    ].sum(axis=1)
    predictions = (raised_probability >= 0.5).astype(np.int64)
    predictions[valid_fraction < minimum] = -1
    return predictions


def _common_binary_metrics(labels: IntArray, predictions: dict[str, IntArray]) -> dict[str, Any]:
    common = np.ones(len(labels), dtype=bool)
    for values in predictions.values():
        common &= values >= 0
    return {
        "common_evaluable": int(common.sum()),
        "models": {
            name: _metrics(labels[common], values[common], ("other", "raised"))
            for name, values in predictions.items()
        },
    }


def _paired_profile_change(
    data: SyntheticMotions,
    mask: NDArray[np.bool_],
    predictions: IntArray,
    label: str,
) -> dict[str, Any]:
    """Pair clean and degraded variants by their held-out procedural base id."""
    positions = np.flatnonzero(mask)
    table: dict[tuple[str, str], tuple[int, int]] = {}
    for local, global_index in enumerate(positions):
        table[(str(data.base_ids[global_index]), str(data.profiles[global_index]))] = (
            int(data.labels[global_index]),
            int(predictions[local]),
        )
    differences: list[int] = []
    for base_id in sorted(set(data.base_ids[mask].tolist())):
        clean = table.get((str(base_id), "clean"))
        other = table.get((str(base_id), label))
        if clean is None or other is None:
            continue
        clean_ok = clean[1] == clean[0]
        other_ok = other[1] == other[0]
        differences.append(int(other_ok) - int(clean_ok))
    return {
        "paired_bases": len(differences),
        "mean_degraded_minus_clean_correct": float(np.mean(differences)) if differences else None,
    }


def _manifest(data: SyntheticMotions) -> list[dict[str, Any]]:
    return [
        {
            "row": row,
            "base_id": str(data.base_ids[row]),
            "split": str(data.splits[row]),
            "profile": str(data.profiles[row]),
            "variant": int(data.variants[row]),
            "temporal_family": str(data.temporal_families[row]),
            "onset_frame": int(data.onset_frames[row]),
            "duration_frames": int(data.duration_frames[row]),
            "amplitude": float(data.amplitudes[row]),
            "class_index": int(data.labels[row]),
            "class_name": MOTION_CLASSES[int(data.labels[row])],
            "label_unit": "procedural observable movement family",
        }
        for row in range(len(data.labels))
    ]


def _articulation_audit(data: SyntheticMotions) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, lengths in segment_lengths(data.clean).items():
        checks[name] = {
            "min_m": float(lengths.min()),
            "max_m": float(lengths.max()),
            "max_within_base_range_m": float(np.ptp(lengths, axis=1).max()),
            "bases_checked": int(lengths.shape[0]),
        }
    return {
        "checked_before_observation_corruption": True,
        "all_clean_bases": checks,
    }


def _freeze_config(config_path: Path, run_dir: Path) -> None:
    destination = run_dir / "config.toml"
    content = config_path.read_bytes()
    if (run_dir / "metrics.json").exists() or (run_dir / "provenance.json").exists():
        raise FileExistsError("ML run already has persisted results; choose a new directory")
    if destination.exists() and destination.read_bytes() != content:
        raise ValueError("Existing ML run directory has a different frozen config")
    run_dir.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)


def _frozen_training(frozen: Path, record: dict[str, Any], name: str, seed: int) -> TcnTraining:
    return TcnTraining(
        load_tcn_checkpoint(frozen / f"{name}.pt"),
        record.get("history", []),
        seed,
        int(record["best_epoch"]),
        float(record["best_validation_loss"]),
    )


def run_experiment(config_path: Path, run_dir: Path, frozen: Path | None = None) -> dict[str, Any]:
    """Generate, train and evaluate the fixed synthetic positive-control protocol.

    With ``frozen``, the ridge weights and network checkpoints of the run behind the
    article are loaded instead of fitted, so the evaluation does not depend on the
    CPU architecture or the BLAS build. Without it, everything is trained from scratch.
    """
    config = load_config(config_path)
    _freeze_config(config_path, run_dir)
    data = generate_synthetic_motions(config.motion)
    data.save(str(run_dir / "synthetic-dataset.npz"))
    _json(run_dir / "synthetic-manifest.json", _manifest(data))
    _json(
        run_dir / "provenance.json",
        {
            "config_sha256": _sha256(config_path),
            "code_sha256": _code_hashes(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "label_limit": (
                "Procedural observable movements only; not football requests or rates."
            ),
            "input_unit": "metres before per-frame normalization",
        },
    )
    _json(run_dir / "articulation-audit.json", _articulation_audit(data))

    normalized = normalize_raw_windows(
        data.observed, data.errors_cm, config.motion.max_joint_error_cm
    )
    inputs = model_input(normalized)
    feature_matrix = temporal_ridge_features(normalized.values)
    train = _training_mask(data, "train")
    validation = _training_mask(data, "validation")
    test = _split_mask(data, "test")
    train_groups = set(data.base_ids[train].tolist())
    validation_groups = set(data.base_ids[validation].tolist())
    test_groups = set(data.base_ids[test].tolist())
    if (
        train_groups & validation_groups
        or train_groups & test_groups
        or validation_groups & test_groups
    ):
        raise RuntimeError("Augmentation leakage: base movement appears in multiple splits")

    if frozen is None:
        ridge = fit_ridge_classifier(
            feature_matrix[train],
            data.labels[train],
            len(MOTION_CLASSES),
            config.ridge_penalty,
        )
    else:
        ridge = load_ridge(frozen)
    (run_dir / "models").mkdir(exist_ok=True)
    np.savez_compressed(
        run_dir / "models" / "ridge.npz",
        mean=ridge.mean,
        scale=ridge.scale,
        weights=ridge.weights,
    )
    _json(run_dir / "models" / "ridge.json", ridge_payload(ridge))
    ridge_probabilities_test = ridge_probabilities(ridge, feature_matrix[test])
    ridge_predictions = _with_abstention(
        ridge_probabilities_test,
        normalized.valid[test].mean(axis=(1, 2)).astype(np.float32),
        config.abstain_min_valid_fraction,
    )
    valid_fraction_test = normalized.valid[test].mean(axis=(1, 2)).astype(np.float32)
    ridge_binary_predictions = _binary_predictions(
        ridge_probabilities_test,
        valid_fraction_test,
        config.abstain_min_valid_fraction,
    )
    np.savez_compressed(
        run_dir / "predictions-ridge.npz",
        probabilities=ridge_probabilities_test,
        predictions=ridge_predictions,
        labels=data.labels[test],
        base_ids=data.base_ids[test],
        profiles=data.profiles[test],
    )

    baseline_predictions = raised_arm_baseline(data.observed[test], data.errors_cm[test])
    binary_labels = np.isin(
        data.labels[test], (CLASS_INDEX["one_arm_raise"], CLASS_INDEX["two_arm_raise"])
    ).astype(np.int64)
    test_profiles = data.profiles[test]
    tcn_results: list[dict[str, Any]] = []
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    best_seed = config.tcn_seeds[0]
    record = {} if frozen is None else json.loads((frozen / "training-record.json").read_text())
    best_validation_loss = float("inf")
    for seed in config.tcn_seeds:
        if frozen is None:
            trained = train_tcn(
                inputs[train],
                data.labels[train],
                inputs[validation],
                data.labels[validation],
                len(MOTION_CLASSES),
                seed,
                config.tcn_epochs,
                config.tcn_batch_size,
                config.tcn_learning_rate,
            )
            checkpoint = run_dir / "checkpoints" / f"tcn-seed-{seed}.pt"
            save_tcn_checkpoint(checkpoint, trained.model)
        else:
            trained = _frozen_training(frozen, record["tcn"][str(seed)], f"tcn-seed-{seed}", seed)
            shutil.copyfile(
                frozen / f"tcn-seed-{seed}.pt", run_dir / "checkpoints" / f"tcn-seed-{seed}.pt"
            )
        probabilities = tcn_probabilities(trained.model, inputs[test])
        predictions = _with_abstention(
            probabilities,
            normalized.valid[test].mean(axis=(1, 2)).astype(np.float32),
            config.abstain_min_valid_fraction,
        )
        np.savez_compressed(
            run_dir / f"predictions-tcn-seed-{seed}.npz",
            probabilities=probabilities,
            predictions=predictions,
            labels=data.labels[test],
            base_ids=data.base_ids[test],
            profiles=data.profiles[test],
        )
        binary_predictions = _binary_predictions(
            probabilities, valid_fraction_test, config.abstain_min_valid_fraction
        )
        if trained.best_validation_loss < best_validation_loss:
            best_validation_loss = trained.best_validation_loss
            best_seed = seed
        tcn_results.append(
            {
                "seed": seed,
                "best_epoch": trained.best_epoch,
                "best_validation_loss": trained.best_validation_loss,
                "history": trained.history,
                "overall": _metrics(data.labels[test], predictions, MOTION_CLASSES),
                "by_profile": _profile_metrics(
                    data.labels[test], test_profiles, predictions, MOTION_CLASSES
                ),
                "binary": {
                    "overall": _metrics(binary_labels, binary_predictions, ("other", "raised")),
                    "by_profile": _profile_metrics(
                        binary_labels,
                        test_profiles,
                        binary_predictions,
                        ("other", "raised"),
                    ),
                    "common_with_geometric": _common_binary_metrics(
                        binary_labels,
                        {
                            "geometric": baseline_predictions,
                            "tcn": binary_predictions,
                        },
                    ),
                },
                "paired_noisy": _paired_profile_change(data, test, predictions, "noisy"),
                "paired_occluded": _paired_profile_change(data, test, predictions, "occluded"),
            }
        )

    shuffled = np.random.default_rng(config.motion.seed + 900_000).permutation(data.labels[train])
    if frozen is None:
        null_trained = train_tcn(
            inputs[train],
            shuffled,
            inputs[validation],
            data.labels[validation],
            len(MOTION_CLASSES),
            config.tcn_seeds[0],
            config.tcn_epochs,
            config.tcn_batch_size,
            config.tcn_learning_rate,
        )
    else:
        null_trained = _frozen_training(
            frozen, record["shuffled_labels"], "tcn-shuffled-labels", config.tcn_seeds[0]
        )
    null_probabilities = tcn_probabilities(null_trained.model, inputs[test])
    null_predictions = _with_abstention(
        null_probabilities, valid_fraction_test, config.abstain_min_valid_fraction
    )
    if frozen is None:
        save_tcn_checkpoint(run_dir / "checkpoints" / "tcn-shuffled-labels.pt", null_trained.model)
    else:
        shutil.copyfile(
            frozen / "tcn-shuffled-labels.pt", run_dir / "checkpoints" / "tcn-shuffled-labels.pt"
        )
    np.savez_compressed(
        run_dir / "predictions-tcn-shuffled-labels.npz",
        probabilities=null_probabilities,
        predictions=null_predictions,
        labels=data.labels[test],
    )

    metrics: dict[str, Any] = {
        "protocol": {
            "classes": list(MOTION_CLASSES),
            "window": "11 frames; 2-second trailing window at 5 Hz",
            "split_before_augmentation": True,
            "base_group_counts": {
                split: len(set(data.base_ids[data.splits == split].tolist()))
                for split in ("train", "validation", "test")
            },
            "test_profiles": ["clean", "noisy", "occluded"],
            "temporal_families_by_split": {
                split: sorted(set(data.temporal_families[data.splits == split].tolist()))
                for split in ("train", "validation", "test")
            },
            "rate_limit": ("Synthetic class balance and event rates are not football rates."),
        },
        "ridge": {
            "overall": _metrics(data.labels[test], ridge_predictions, MOTION_CLASSES),
            "by_profile": _profile_metrics(
                data.labels[test], test_profiles, ridge_predictions, MOTION_CLASSES
            ),
            "binary": {
                "overall": _metrics(binary_labels, ridge_binary_predictions, ("other", "raised")),
                "by_profile": _profile_metrics(
                    binary_labels,
                    test_profiles,
                    ridge_binary_predictions,
                    ("other", "raised"),
                ),
                "common_with_geometric": _common_binary_metrics(
                    binary_labels,
                    {
                        "geometric": baseline_predictions,
                        "ridge": ridge_binary_predictions,
                    },
                ),
            },
            "paired_noisy": _paired_profile_change(data, test, ridge_predictions, "noisy"),
            "paired_occluded": _paired_profile_change(data, test, ridge_predictions, "occluded"),
        },
        "raised_arm_baseline_binary": {
            "labels": ["other", "one_or_two_arms_raised"],
            "minimum_consecutive_frames": 3,
            "overall": _metrics(binary_labels, baseline_predictions, ("other", "raised")),
            "by_profile": _profile_metrics(
                binary_labels,
                test_profiles,
                baseline_predictions,
                ("other", "raised"),
            ),
        },
        "tcn": tcn_results,
        "shuffled_label_control": {
            "seed": config.tcn_seeds[0],
            "best_epoch": null_trained.best_epoch,
            "best_validation_loss": null_trained.best_validation_loss,
            "overall": _metrics(data.labels[test], null_predictions, MOTION_CLASSES),
            "interpretation": (
                "A label-permutation control checks for synthetic label shortcuts; "
                "it has no success threshold."
            ),
        },
        "chosen_checkpoint": f"checkpoints/tcn-seed-{best_seed}.pt",
        "selection_rule": ("Lowest final validation loss among fixed seeds; test set not used."),
    }
    _json(run_dir / "metrics.json", metrics)
    groups = metrics["protocol"]["base_group_counts"]
    ridge_overall = metrics["ridge"]["overall"]
    baseline_overall = metrics["raised_arm_baseline_binary"]["overall"]
    report = "\n".join(
        [
            "# Synthetic motion-recognition positive control",
            "",
            (
                "This run uses wholly procedural articulated poses. It tests recovery "
                "of five authored observable movement families, not recognition of "
                "football requests or their in-match frequency."
            ),
            "",
            (
                f"Training bases: {groups['train']}; validation: "
                f"{groups['validation']}; test: {groups['test']}."
            ),
            (
                f"Ridge macro-F1: {ridge_overall['macro_f1']:.3f}; balanced "
                f"accuracy: {ridge_overall['balanced_accuracy']:.3f}."
            ),
            (
                "Binary geometric baseline balanced accuracy: "
                f"{baseline_overall['balanced_accuracy']:.3f}."
            ),
            f"Selected frozen TCN checkpoint: `{metrics['chosen_checkpoint']}`.",
            "",
            (
                "Read `metrics.json` for profile-specific denominators, abstentions, "
                "confusions and paired degradation comparisons."
            ),
        ]
    )
    (run_dir / "report.md").write_text(report + "\n")
    return metrics
