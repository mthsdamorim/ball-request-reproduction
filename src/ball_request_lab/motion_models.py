"""Normalization, geometric baseline and small temporal models for synthetic poses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from ball_request_lab.data import JOINTS, MatchData
from ball_request_lab.gestures import GestureConfig, arm_states

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
_J = {name: index for index, name in enumerate(JOINTS)}


@dataclass(frozen=True)
class NormalizedWindows:
    """Metric pose is transformed framewise; unknown joints remain explicit."""

    values: FloatArray  # (windows, frames, joints, xyz + known-mask)
    valid: BoolArray
    frame_valid: BoolArray


def normalize_raw_windows(
    raw_windows_m: FloatArray, errors_cm: FloatArray, max_error_cm: float = 20.0
) -> NormalizedWindows:
    """Hip-centre and orient each frame independently, without future information."""
    if raw_windows_m.ndim != 4 or raw_windows_m.shape[-2:] != (len(JOINTS), 3):
        raise ValueError("Expected raw windows shaped (N, frames, 29, 3)")
    if errors_cm.shape != raw_windows_m.shape[:-1]:
        raise ValueError("Errors must be shaped (N, frames, 29)")
    known = np.isfinite(raw_windows_m).all(axis=-1) & (errors_cm <= max_error_cm)
    hip = raw_windows_m[:, :, _J["midHip"]]
    left = raw_windows_m[:, :, _J["lShoulder"], :2]
    right = raw_windows_m[:, :, _J["rShoulder"], :2]
    lateral = right - left
    lateral_norm = np.linalg.norm(lateral, axis=-1)
    frame_valid = (
        known[:, :, _J["midHip"]]
        & known[:, :, _J["lShoulder"]]
        & known[:, :, _J["rShoulder"]]
        & (lateral_norm > 0.05)
    )
    safe_norm = np.maximum(lateral_norm, 1e-6)
    y_axis = lateral / safe_norm[..., None]
    x_axis = np.stack((-y_axis[..., 1], y_axis[..., 0]), axis=-1)
    relative = raw_windows_m - hip[:, :, None, :]
    result = np.zeros((*raw_windows_m.shape[:-1], 4), dtype=np.float32)
    xy = relative[..., :2]
    result[..., 0] = np.sum(xy * x_axis[:, :, None, :], axis=-1)
    result[..., 1] = np.sum(xy * y_axis[:, :, None, :], axis=-1)
    result[..., 2] = relative[..., 2]
    valid = known & frame_valid[:, :, None]
    result[..., :3] = np.where(valid[..., None], result[..., :3], 0.0)
    result[..., 3] = valid.astype(np.float32)
    return NormalizedWindows(result, valid, frame_valid)


def model_input(normalized: NormalizedWindows) -> FloatArray:
    """Return channels-first temporal input (N, 29 * 4, T)."""
    values = normalized.values.transpose(0, 2, 3, 1)
    return values.reshape(len(values), len(JOINTS) * 4, values.shape[-1])


def _match_data_from_windows(raw: FloatArray, errors: FloatArray) -> MatchData:
    """Adapt independent pose windows to the established geometric arm-state rule."""
    n, frames = raw.shape[:2]
    # Independent windows become parallel player columns.  ``arm_states`` only
    # uses a player's own joints, so this preserves every temporal sequence.
    joints = raw.transpose(1, 0, 2, 3)
    joint_errors = errors.transpose(1, 0, 2)
    active = np.ones((frames, n), dtype=bool)
    xy = np.zeros((frames, n, 2), dtype=np.float32)
    return MatchData(
        xy=xy,
        detected=active.copy(),
        active=active,
        ball=np.zeros((frames, 2), dtype=np.float32),
        ball_detected=np.ones(frames, dtype=bool),
        joints=joints,
        errors=joint_errors,
        period=np.ones(frames, dtype=np.int64),
        clock=np.arange(frames, dtype=np.float64) / 5.0,
        owner=np.full(frames, -1, dtype=np.int64),
        in_play=np.ones(frames, dtype=bool),
        players=np.arange(n, dtype=np.int64),
        teams=np.ones(n, dtype=np.int64),
        outfield=np.ones(n, dtype=bool),
    )


def raised_arm_baseline(
    raw_windows_m: FloatArray, errors_cm: FloatArray, minimum_frames: int = 3
) -> NDArray[np.int64]:
    """Use the established arm rule as a temporal binary raised-arm comparator."""
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be positive")
    data = _match_data_from_windows(raw_windows_m, errors_cm)
    states, _ = arm_states(data, GestureConfig())
    n, _frames = raw_windows_m.shape[:2]
    result = np.zeros(n, dtype=np.int64)
    for window in range(n):
        present = states[:, window] > 0
        consecutive = np.convolve(present.astype(np.int64), np.ones(minimum_frames), "valid")
        arm_indices = [
            _J[name]
            for side in ("l", "r")
            for name in (f"{side}Shoulder", f"{side}Elbow", f"{side}Wrist")
        ]
        arm_coordinates = raw_windows_m[window][:, arm_indices]
        arm_errors = errors_cm[window][:, arm_indices]
        observable = np.isfinite(arm_coordinates).all(axis=(-1, -2)) & (arm_errors <= 20.0).all(
            axis=-1
        )
        result[window] = -1 if not observable.any() else int(np.any(consecutive >= minimum_frames))
    return result


@dataclass(frozen=True)
class RidgeClassifier:
    mean: FloatArray
    scale: FloatArray
    weights: FloatArray
    class_count: int
    penalty: float


def temporal_ridge_features(values: FloatArray) -> FloatArray:
    """Flatten coordinates/masks and first differences for a transparent baseline."""
    if values.ndim != 4 or values.shape[-1] != 4:
        raise ValueError("Expected normalized (N, frames, joints, 4) windows")
    difference = np.diff(values[..., :3], axis=1, prepend=values[:, :1, :, :3])
    return np.concatenate(
        [values.reshape(len(values), -1), difference.reshape(len(values), -1)], axis=1
    ).astype(np.float32)


def fit_ridge_classifier(
    features: FloatArray, labels: IntArray, class_count: int, penalty: float = 1.0
) -> RidgeClassifier:
    """Fit regularized least squares on one-hot labels, not logistic regression."""
    if len(features) != len(labels) or class_count < 2:
        raise ValueError("Features, labels and class count are incompatible")
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(features.std(axis=0, dtype=np.float64), 1e-5).astype(np.float32)
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(features), dtype=np.float32), standardized))
    target = np.eye(class_count, dtype=np.float32)[labels]
    regularizer = np.eye(design.shape[1], dtype=np.float32) * penalty
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return RidgeClassifier(mean, scale, weights.astype(np.float32), class_count, penalty)


def ridge_probabilities(model: RidgeClassifier, features: FloatArray) -> FloatArray:
    standardized = (features - model.mean) / model.scale
    design = np.column_stack((np.ones(len(features), dtype=np.float32), standardized))
    scores = design @ model.weights
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


class TemporalConvNet(nn.Module):
    """Small CPU TCN: local temporal convolutions followed by global averaging."""

    def __init__(self, input_channels: int, class_count: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.class_count = class_count
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(32, class_count)

    def forward(self, value: Tensor) -> Tensor:
        return self.classifier(self.network(value).mean(dim=-1))


@dataclass(frozen=True)
class TcnTraining:
    model: TemporalConvNet
    history: list[dict[str, float]]
    seed: int
    best_epoch: int
    best_validation_loss: float


def train_tcn(
    train_x: FloatArray,
    train_y: IntArray,
    validation_x: FloatArray,
    validation_y: IntArray,
    class_count: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> TcnTraining:
    """Train one deterministic CPU seed and retain the best validation epoch."""
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = TemporalConvNet(train_x.shape[1], class_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_tensor = torch.from_numpy(train_x)  # pyright: ignore[reportUnknownMemberType]
    target_tensor = torch.from_numpy(train_y)  # pyright: ignore[reportUnknownMemberType]
    validation_tensor = torch.from_numpy(validation_x)  # pyright: ignore[reportUnknownMemberType]
    validation_target = torch.from_numpy(validation_y)  # pyright: ignore[reportUnknownMemberType]
    best_state: dict[str, Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_x))
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
                order[start : start + batch_size]
            )
            optimizer.zero_grad()
            logits = model(train_tensor[index])
            loss = F.cross_entropy(logits, target_tensor[index])
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(F.cross_entropy(model(validation_tensor), validation_target))
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("TCN training did not execute")
    model.load_state_dict(best_state)
    return TcnTraining(model, history, seed, best_epoch, best_loss)


def tcn_probabilities(model: TemporalConvNet, inputs: FloatArray) -> FloatArray:
    model.eval()
    with torch.no_grad():
        return (
            torch.softmax(
                model(torch.from_numpy(inputs)),  # pyright: ignore[reportUnknownMemberType]
                dim=1,
            )
            .numpy()
            .astype(np.float32)
        )


def save_tcn_checkpoint(path: Path, model: TemporalConvNet) -> None:
    torch.save(
        {
            "input_channels": model.input_channels,
            "class_count": model.class_count,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_tcn_checkpoint(path: Path) -> TemporalConvNet:
    payload: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
    model = TemporalConvNet(int(payload["input_channels"]), int(payload["class_count"]))
    state = payload["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint state")
    model.load_state_dict(cast(dict[str, Tensor], state))
    model.eval()
    return model


def predict_on_raw_windows(
    model: TemporalConvNet,
    raw_windows_m: FloatArray,
    errors_cm: FloatArray,
    max_error_cm: float = 20.0,
) -> dict[str, NDArray[Any]]:
    """Apply a frozen TCN to raw pose windows without fitting or calibration."""
    normalized = normalize_raw_windows(raw_windows_m, errors_cm, max_error_cm)
    probabilities = tcn_probabilities(model, model_input(normalized))
    return {
        "probabilities": probabilities,
        "predicted": probabilities.argmax(axis=1).astype(np.int64),
        "valid_fraction": normalized.valid.mean(axis=(1, 2)).astype(np.float32),
    }


def load_ridge(models_dir: Path) -> RidgeClassifier:
    """Load a ridge classifier saved as ``ridge.npz`` plus ``ridge.json``."""
    with np.load(models_dir / "ridge.npz", allow_pickle=False) as parameters:
        meta = json.loads((models_dir / "ridge.json").read_text())
        return RidgeClassifier(
            parameters["mean"],
            parameters["scale"],
            parameters["weights"],
            int(meta["class_count"]),
            float(meta["penalty"]),
        )


def ridge_payload(model: RidgeClassifier) -> dict[str, Any]:
    """Make parameters inspectable without pickling executable code."""
    return {
        "class_count": model.class_count,
        "penalty": model.penalty,
        "shapes": {
            "mean": list(model.mean.shape),
            "scale": list(model.scale.shape),
            "weights": list(model.weights.shape),
        },
    }
