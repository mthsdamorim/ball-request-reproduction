"""Independent real windows: freeze selection before labels or model predictions."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from ball_request_lab.data import HZ, JOINTS, MatchData, metadata
from ball_request_lab.gestures import eligible_samples
from ball_request_lab.util import atomic_json, file_hash

STEPS = 11


def select_indices(
    data: MatchData,
    period: int,
    previous_clocks: list[float],
    *,
    count: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Sample player-windows by time stratum, without arm states or pass outcomes.

    The population is quality-eligible, detected off-ball teammates of a known
    holder, not all players or all match time. Prior review clips are excluded
    for every player, not just their original target.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    arm = [
        JOINTS.index(side + part) for side in ("l", "r") for part in ("Shoulder", "Elbow", "Wrist")
    ]
    core = [JOINTS.index(name) for name in ("midHip", "neck", "lShoulder", "rShoulder")]
    good = np.all(np.isfinite(data.joints), axis=-1) & (data.errors <= 20.0)
    eligible = eligible_samples(data) & data.detected
    eligible &= good[:, :, arm].mean(axis=-1) >= 0.8
    eligible &= np.all(good[:, :, core], axis=-1)
    eligible &= (data.period == period)[:, None]
    # Every sample of a trailing window must satisfy the declared population.
    consecutive = np.zeros_like(eligible, dtype=np.int64)
    for t in range(len(eligible)):
        consecutive[t] = (consecutive[t - 1] + 1) * eligible[t] if t else eligible[t]
    anchors = np.argwhere(consecutive >= STEPS)
    if not len(anchors):
        return []
    clocks = data.clock[anchors[:, 0]]
    keep = np.isfinite(clocks)
    for old in previous_clocks:
        keep &= np.abs(clocks - old) > 15.0
    anchors, clocks = anchors[keep], clocks[keep]
    if not len(anchors):
        return []
    edges = np.linspace(float(clocks.min()), float(clocks.max()) + 1e-6, count + 1)
    rng = np.random.default_rng(seed)
    chosen: list[tuple[int, int]] = []
    for low, high in pairwise(edges):
        indices = np.flatnonzero((clocks >= low) & (clocks < high))
        for candidate in rng.permutation(indices):
            t, p = (int(value) for value in anchors[candidate])
            if all(abs(float(data.clock[t] - data.clock[old])) > 10.0 for old, _ in chosen):
                chosen.append((t, p))
                break
    return chosen


def build_real_windows(
    data_dir: Path,
    run_dir: Path,
    previous_selection: Path,
    *,
    per_half: int = 4,
    seed: int = 20261004,
) -> dict[str, Any]:
    """Persist a new source-grouped review set and raw trailing pose tensors."""
    if (run_dir / "selection.json").exists():
        raise FileExistsError("Selection is immutable; choose another run directory")
    previous = json.loads(previous_selection.read_text())
    cases: list[dict[str, Any]] = []
    poses: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    sources: dict[str, str] = {}
    for match_id in (1925299, 1996435):
        prepared = data_dir / "prepared" / f"{match_id}.npz"
        data = MatchData.load(prepared)
        sources[str(match_id)] = file_hash(prepared)
        meta = metadata(data_dir, match_id)
        people = {int(p["id"]): p for p in meta["players"]}
        teams = {int(meta[k]["id"]): meta[k]["short_name"] for k in ("home_team", "away_team")}
        for period in (1, 2):
            prior = [
                float(c["source_clock_s"])
                for c in previous["cases"]
                if c["match_id"] == match_id and c["period"] == period
            ]
            for ordinal, (t, p) in enumerate(
                select_indices(data, period, prior, count=per_half, seed=seed + match_id + period),
                start=1,
            ):
                person = people[int(data.players[p])]
                case_id = f"independent-{match_id}-{period}-{ordinal:02d}"
                poses.append(data.joints[t - STEPS + 1 : t + 1, p].copy())
                errors.append(data.errors[t - STEPS + 1 : t + 1, p].copy())
                cases.append(
                    {
                        "case_id": case_id,
                        "match_id": match_id,
                        "period": period,
                        "player_id": int(data.players[p]),
                        "player_index": p,
                        "player_name": person["short_name"],
                        "shirt_number": person["number"],
                        "team_id": int(data.teams[p]),
                        "team_name": teams[int(data.teams[p])],
                        "start_index": t - STEPS + 1,
                        "analysis_index": t,
                        "source_clock_s": float(data.clock[t]),
                        "source_start_s": float(data.clock[t - STEPS + 1]),
                        "cohort": "independent_quality_eligible",
                        "use": "reserved_transfer_audit_not_training",
                    }
                )
    if not cases:
        raise ValueError("No eligible independent real windows")
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "windows.npz",
        raw_windows_m=np.stack(poses),
        errors_cm=np.stack(errors),
        case_ids=np.array([c["case_id"] for c in cases]),
    )
    selection: dict[str, Any] = {
        "schema_version": 1,
        "seed": seed,
        "per_half_requested": per_half,
        "steps": STEPS,
        "sample_hz": HZ,
        "window_duration_s": (STEPS - 1) / HZ,
        "method": "uniform player-window draw within chronological strata; "
        "no detector, gesture or pass selection",
        "population": "detected active outfield off-ball teammate, known holder, "
        "live play, core and arm quality at all 11 samples",
        "exclusion": "all players within 15s of prior review anchors; "
        "selected anchors separated by >10s",
        "prepared_sha256": sources,
        "previous_selection_sha256": file_hash(previous_selection),
        "windows_sha256": file_hash(run_dir / "windows.npz"),
        "selector_sha256": file_hash(Path(__file__)),
        "cases": cases,
        "interpretation": "New windows from two previously inspected matches; "
        "not external match generalization. Model and generator may not be "
        "adjusted against these labels.",
    }
    atomic_json(run_dir / "selection.json", selection)
    atomic_json(
        run_dir / "annotations.json",
        [
            {
                "case_id": c["case_id"],
                "gesture_type": "unreviewed",
                "evaluable_identity": False,
                "identity_confidence": "unreviewed",
                "communicative_function": "unreviewed",
                "reviewer": None,
                "label_window": "only the 11 frames from source_start_s through source_clock_s",
                "notes": "",
            }
            for c in cases
        ],
    )
    return selection
