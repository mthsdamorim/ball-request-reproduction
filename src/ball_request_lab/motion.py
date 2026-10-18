"""Procedural, articulated pose sequences for a bounded motion-recognition test.

The generated sequences are deliberately not football observations.  They are a
positive control for the representation, missing-data handling and temporal
models used by :mod:`ball_request_lab.ml_experiment`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from ball_request_lab.data import JOINTS

MOTION_CLASSES: Final = (
    "no_signal",
    "one_arm_raise",
    "two_arm_raise",
    "directional_point",
    "hand_to_head",
)
CLASS_INDEX: Final = {name: index for index, name in enumerate(MOTION_CLASSES)}

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MotionConfig:
    """Fixed procedural experiment settings, expressed in metres and seconds."""

    seed: int = 20261004
    frames: int = 11
    train_bases_per_class: int = 80
    validation_bases_per_class: int = 20
    test_bases_per_class: int = 30
    train_variants: int = 2
    validation_variants: int = 2
    max_joint_error_cm: float = 20.0

    def __post_init__(self) -> None:
        if self.frames != 11:
            raise ValueError("This protocol fixes 2 s trailing windows at 5 Hz (11 frames)")
        if (
            min(
                self.train_bases_per_class,
                self.validation_bases_per_class,
                self.test_bases_per_class,
                self.train_variants,
                self.validation_variants,
            )
            < 1
        ):
            raise ValueError("Every split needs at least one base and one variant")


@dataclass(frozen=True)
class SyntheticMotions:
    """Generated clean skeletons and their independently corrupted observations."""

    clean: FloatArray
    observed: FloatArray
    errors_cm: FloatArray
    labels: IntArray
    base_ids: NDArray[np.str_]
    splits: NDArray[np.str_]
    profiles: NDArray[np.str_]
    variants: IntArray
    temporal_families: NDArray[np.str_]
    onset_frames: IntArray
    duration_frames: IntArray
    amplitudes: FloatArray

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            clean=self.clean,
            observed=self.observed,
            errors_cm=self.errors_cm,
            labels=self.labels,
            base_ids=self.base_ids,
            splits=self.splits,
            profiles=self.profiles,
            variants=self.variants,
            temporal_families=self.temporal_families,
            onset_frames=self.onset_frames,
            duration_frames=self.duration_frames,
            amplitudes=self.amplitudes,
            classes=np.asarray(MOTION_CLASSES),
        )


_J: Final = {name: index for index, name in enumerate(JOINTS)}


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0])
    return vector / norm


def _two_link_elbow(
    shoulder: NDArray[np.float64],
    wrist: NDArray[np.float64],
    upper: float,
    lower: float,
    bend_sign: float,
) -> NDArray[np.float64]:
    """Solve a three-dimensional elbow point with exact arm segment lengths."""
    delta = wrist - shoulder
    distance = float(np.linalg.norm(delta))
    # The action targets are authored inside this reachable range.  The clip is
    # nevertheless useful against round-off and custom configuration mistakes.
    distance = min(max(distance, abs(upper - lower) + 1e-5), upper + lower - 1e-5)
    direction = _unit(delta)
    perpendicular = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    if float(np.linalg.norm(perpendicular)) < 1e-5:
        perpendicular = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    perpendicular = _unit(perpendicular) * bend_sign
    along = (upper * upper - lower * lower + distance * distance) / (2.0 * distance)
    height = float(np.sqrt(max(upper * upper - along * along, 0.0)))
    return shoulder + along * direction + height * perpendicular


def _base_skeleton(rng: np.random.Generator) -> NDArray[np.float64]:
    """Return a 29-joint person in local coordinates, scaled per base movement."""
    scale = float(rng.uniform(0.90, 1.12))
    points = np.zeros((len(JOINTS), 3), dtype=np.float64)
    points[_J["midHip"]] = (0.0, 0.0, 0.0)
    points[_J["neck"]] = (0.0, 0.0, 0.56 * scale)
    points[_J["nose"]] = (0.10 * scale, 0.0, 0.75 * scale)
    for side, sign in (("l", 1.0), ("r", -1.0)):
        points[_J[f"{side}Shoulder"]] = (0.0, sign * 0.21 * scale, 0.48 * scale)
        points[_J[f"{side}Elbow"]] = (0.04 * scale, sign * 0.32 * scale, 0.25 * scale)
        points[_J[f"{side}Wrist"]] = (0.08 * scale, sign * 0.38 * scale, 0.04 * scale)
        points[_J[f"{side}Hip"]] = (0.0, sign * 0.15 * scale, -0.02 * scale)
        points[_J[f"{side}Knee"]] = (0.03 * scale, sign * 0.13 * scale, -0.46 * scale)
        points[_J[f"{side}Ankle"]] = (0.0, sign * 0.12 * scale, -0.88 * scale)
        points[_J[f"{side}Heel"]] = (-0.08 * scale, sign * 0.12 * scale, -0.89 * scale)
        points[_J[f"{side}BigToe"]] = (0.17 * scale, sign * 0.08 * scale, -0.90 * scale)
        points[_J[f"{side}SmallToe"]] = (
            0.14 * scale,
            sign * 0.17 * scale,
            -0.90 * scale,
        )
        points[_J[f"{side}Eye"]] = (0.10 * scale, sign * 0.035 * scale, 0.78 * scale)
        points[_J[f"{side}Ear"]] = (0.02 * scale, sign * 0.10 * scale, 0.73 * scale)
        wrist = points[_J[f"{side}Wrist"]]
        points[_J[f"{side}Thumb"]] = wrist + np.array((0.03 * scale, sign * 0.015 * scale, 0.0))
        points[_J[f"{side}Pinky"]] = wrist + np.array(
            (0.02 * scale, sign * 0.025 * scale, -0.015 * scale)
        )
    return points


def _set_arm(frame: NDArray[np.float64], side: str, wrist: NDArray[np.float64]) -> None:
    shoulder_index = _J[f"{side}Shoulder"]
    elbow_index = _J[f"{side}Elbow"]
    wrist_index = _J[f"{side}Wrist"]
    shoulder = cast(NDArray[np.float64], frame[shoulder_index])
    elbow = cast(NDArray[np.float64], frame[elbow_index])
    prior_wrist = cast(NDArray[np.float64], frame[wrist_index])
    upper = float(np.linalg.norm(elbow - shoulder))
    lower = float(np.linalg.norm(prior_wrist - elbow))
    sign = 1.0 if side == "l" else -1.0
    delta = wrist - shoulder
    distance = float(np.linalg.norm(delta))
    clipped_distance = min(max(distance, abs(upper - lower) + 1e-5), upper + lower - 1e-5)
    wrist = shoulder + _unit(delta) * clipped_distance
    frame[elbow_index] = _two_link_elbow(shoulder, wrist, upper, lower, sign)
    frame[wrist_index] = wrist
    direction = _unit(wrist - shoulder)
    lateral = np.array([0.0, sign, 0.0])
    frame[_J[f"{side}Thumb"]] = wrist + 0.035 * direction + 0.015 * lateral
    frame[_J[f"{side}Pinky"]] = wrist + 0.04 * direction - 0.012 * lateral


def _smoothstep(progress: float) -> float:
    clipped = min(max(progress, 0.0), 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


@dataclass(frozen=True)
class TemporalProfile:
    """A preassigned temporal family for one base movement."""

    family: str
    onset: int
    duration: int
    amplitude: float


def _temporal_profile(
    rng: np.random.Generator, split: str, ordinal: int, frames: int
) -> TemporalProfile:
    """Choose families by split before any observation augmentation exists."""
    if split == "train":
        family = ("smooth", "linear")[ordinal % 2]
        onset = int(rng.integers(1, 3))
        duration = int(rng.integers(5, 7))
    elif split == "validation":
        family = "cosine"
        onset = int(rng.integers(1, 3))
        duration = int(rng.integers(5, 7))
    elif split == "test":
        family = ("late", "pulse")[ordinal % 2]
        onset = int(rng.integers(4, 6)) if family == "late" else int(rng.integers(2, 4))
        duration = int(rng.integers(3, 5))
    else:
        raise ValueError(f"Unknown split: {split}")
    if onset + duration > frames:
        duration = frames - onset
    return TemporalProfile(
        family=family,
        onset=onset,
        duration=max(2, duration),
        amplitude=float(rng.uniform(0.82, 1.12)),
    )


def _temporal_progress(frame: int, profile: TemporalProfile) -> float:
    point = (frame - profile.onset) / max(profile.duration, 1)
    if profile.family == "linear":
        value = min(max(point, 0.0), 1.0)
    elif profile.family == "cosine":
        clipped = min(max(point, 0.0), 1.0)
        value = 0.5 - 0.5 * float(np.cos(np.pi * clipped))
    elif profile.family == "late":
        value = _smoothstep(point)
    elif profile.family == "pulse":
        clipped = min(max(point, 0.0), 1.0)
        value = float(np.sin(np.pi * clipped))
    elif profile.family == "smooth":
        value = _smoothstep(point)
    else:
        raise ValueError(f"Unknown temporal family: {profile.family}")
    return value * profile.amplitude


def _gesture_targets(
    base: NDArray[np.float64], label: int, progress: float, side: str
) -> dict[str, NDArray[np.float64]]:
    """Give reachable temporal wrist targets for the five observable classes."""
    result: dict[str, NDArray[np.float64]] = {}
    raised_sides: tuple[str, ...]
    name = MOTION_CLASSES[label]
    if name == "one_arm_raise":
        raised_sides = (side,)
    elif name == "two_arm_raise":
        raised_sides = ("l", "r")
    else:
        raised_sides = ()
    for arm in raised_sides:
        start = base[_J[f"{arm}Wrist"]]
        shoulder = base[_J[f"{arm}Shoulder"]]
        sign = 1.0 if arm == "l" else -1.0
        target = shoulder + np.array((0.02, sign * 0.10, 0.40))
        result[arm] = (1.0 - progress) * start + progress * target
    if name == "directional_point":
        start = base[_J[f"{side}Wrist"]]
        shoulder = base[_J[f"{side}Shoulder"]]
        target = shoulder + np.array((0.40, 0.02 if side == "l" else -0.02, 0.02))
        result[side] = (1.0 - progress) * start + progress * target
    if name == "hand_to_head":
        start = base[_J[f"{side}Wrist"]]
        target = base[_J["nose"]] + np.array((0.0, 0.06 if side == "l" else -0.06, -0.03))
        result[side] = (1.0 - progress) * start + progress * target
    return result


def make_base_motion(
    label: int,
    base_seed: int,
    frames: int = 11,
    temporal_profile: TemporalProfile | None = None,
) -> FloatArray:
    """Create one clean articulated movement; it has no measurement corruption."""
    if label not in range(len(MOTION_CLASSES)):
        raise ValueError("Unknown motion label")
    rng = np.random.default_rng(base_seed)
    base = _base_skeleton(rng)
    side = "l" if bool(rng.integers(0, 2)) else "r"
    heading = float(rng.uniform(-np.pi, np.pi))
    speed = float(rng.uniform(0.8, 3.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    profile = temporal_profile or TemporalProfile("smooth", 2, frames - 5, 1.0)
    rotation = np.array(((np.cos(heading), -np.sin(heading)), (np.sin(heading), np.cos(heading))))
    output = np.empty((frames, len(JOINTS), 3), dtype=np.float64)
    for t in range(frames):
        progress = _temporal_progress(t, profile)
        frame = base.copy()
        # A small coherent running oscillation is shared by every class.
        oscillation = 0.045 * np.sin(phase + t * 0.8)
        for arm, sign in (("l", 1.0), ("r", -1.0)):
            wrist = frame[_J[f"{arm}Wrist"]].copy()
            wrist[0] += oscillation * sign
            wrist[2] -= abs(oscillation) * 0.35
            _set_arm(frame, arm, wrist)
        for arm, wrist in _gesture_targets(base, label, progress, side).items():
            _set_arm(frame, arm, wrist)
        # Lower-body points stay rigid.  This limited upper-body generator does
        # not claim to simulate running biomechanics.
        frame[:, :2] = frame[:, :2] @ rotation.T
        frame[:, :2] += (speed * t / 5.0, -0.3 + 0.15 * np.sin(phase))
        output[t] = frame
    return output.astype(np.float32)


def corrupt_observation(
    clean: FloatArray, profile: str, seed: int
) -> tuple[FloatArray, FloatArray]:
    """Apply correlated coordinate error and joint absence after articulation."""
    profiles = {
        "clean": (0.0, 0.0, 0.0),
        "medium": (0.012, 0.035, 0.0),
        "noisy": (0.045, 0.10, 0.0),
        "occluded": (0.018, 0.06, 0.35),
    }
    if profile not in profiles:
        raise ValueError(f"Unknown observation profile: {profile}")
    noise_m, dropout, arm_block = profiles[profile]
    rng = np.random.default_rng(seed)
    observed = clean.astype(np.float32).copy()
    errors = np.full(clean.shape[:2], 5.0, dtype=np.float32)
    if noise_m:
        innovation = rng.normal(0.0, noise_m, size=clean.shape).astype(np.float32)
        correlated = np.cumsum(innovation, axis=0)
        correlated -= correlated.mean(axis=0, keepdims=True)
        observed += correlated
        errors += np.float32(noise_m * 100.0)
    missing = rng.random(clean.shape[:2]) < dropout
    if arm_block:
        arm_joints = [
            _J[name]
            for name in JOINTS
            if name.startswith(
                (
                    "lShoulder",
                    "rShoulder",
                    "lElbow",
                    "rElbow",
                    "lWrist",
                    "rWrist",
                    "lThumb",
                    "rThumb",
                    "lPinky",
                    "rPinky",
                )
            )
        ]
        start = int(rng.integers(2, max(3, clean.shape[0] - 2)))
        stop = min(clean.shape[0], start + 3)
        block = rng.random((stop - start, len(arm_joints))) < arm_block
        missing[start:stop, arm_joints] = missing[start:stop, arm_joints] | block
    observed[missing] = np.nan
    errors[missing] = np.inf
    return observed, errors


def _base_counts(config: MotionConfig) -> tuple[tuple[str, int], ...]:
    return (
        ("train", config.train_bases_per_class),
        ("validation", config.validation_bases_per_class),
        ("test", config.test_bases_per_class),
    )


def generate_synthetic_motions(config: MotionConfig) -> SyntheticMotions:
    """Generate separated base groups before making any observation variants."""
    clean_rows: list[FloatArray] = []
    observed_rows: list[FloatArray] = []
    error_rows: list[FloatArray] = []
    labels: list[int] = []
    base_ids: list[str] = []
    splits: list[str] = []
    profiles: list[str] = []
    variants: list[int] = []
    temporal_families: list[str] = []
    onset_frames: list[int] = []
    duration_frames: list[int] = []
    amplitudes: list[float] = []
    base_number = 0
    for label, name in enumerate(MOTION_CLASSES):
        for split, count in _base_counts(config):
            for ordinal in range(count):
                base_seed = config.seed + 10_000 * label + base_number
                profile_rng = np.random.default_rng(base_seed + 50_000)
                temporal = _temporal_profile(profile_rng, split, ordinal, config.frames)
                clean = make_base_motion(label, base_seed, config.frames, temporal)
                identifier = f"{split}-{name}-{ordinal:03d}"
                if split == "train":
                    choices = ("clean", "medium")[: config.train_variants]
                elif split == "validation":
                    choices = ("clean", "medium")[: config.validation_variants]
                else:
                    choices = ("clean", "noisy", "occluded")
                for variant, profile in enumerate(choices):
                    observed, errors = corrupt_observation(
                        clean, profile, base_seed + 100_000 + variant
                    )
                    clean_rows.append(clean)
                    observed_rows.append(observed)
                    error_rows.append(errors)
                    labels.append(label)
                    base_ids.append(identifier)
                    splits.append(split)
                    profiles.append(profile)
                    variants.append(variant)
                    temporal_families.append(temporal.family)
                    onset_frames.append(temporal.onset)
                    duration_frames.append(temporal.duration)
                    amplitudes.append(temporal.amplitude)
                base_number += 1
    return SyntheticMotions(
        clean=np.stack(clean_rows).astype(np.float32),
        observed=np.stack(observed_rows).astype(np.float32),
        errors_cm=np.stack(error_rows).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        base_ids=np.asarray(base_ids),
        splits=np.asarray(splits),
        profiles=np.asarray(profiles),
        variants=np.asarray(variants, dtype=np.int64),
        temporal_families=np.asarray(temporal_families),
        onset_frames=np.asarray(onset_frames, dtype=np.int64),
        duration_frames=np.asarray(duration_frames, dtype=np.int64),
        amplitudes=np.asarray(amplitudes, dtype=np.float32),
    )


def segment_lengths(pose: FloatArray) -> dict[str, FloatArray]:
    """Expose fixed arm and leg segment lengths before observation corruption."""
    result: dict[str, FloatArray] = {}
    for side in ("l", "r"):
        for name, first, second in (
            ("upper", f"{side}Shoulder", f"{side}Elbow"),
            ("lower", f"{side}Elbow", f"{side}Wrist"),
            ("thigh", f"{side}Hip", f"{side}Knee"),
            ("shin", f"{side}Knee", f"{side}Ankle"),
        ):
            delta = pose[..., _J[first], :] - pose[..., _J[second], :]
            result[f"{side}_{name}"] = np.linalg.norm(delta, axis=-1).astype(np.float32)
    return result
