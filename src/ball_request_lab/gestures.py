"""Temporal geometric proposals for off-ball requests, before visual adjudication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ball_request_lab.data import HZ, JOINTS, MatchData


@dataclass(frozen=True)
class GestureConfig:
    wrist_above_shoulder_m: float = 0.08
    wrist_above_elbow_m: float = 0.02
    max_joint_error_cm: float = 20.0
    min_segment_m: float = 0.12
    max_segment_m: float = 0.65
    min_duration_s: float = 0.4

    def __post_init__(self) -> None:
        if self.min_duration_s <= 0 or self.max_joint_error_cm <= 0:
            raise ValueError("Duration and allowed joint error must be positive")
        if not 0 < self.min_segment_m < self.max_segment_m:
            raise ValueError("Invalid arm-segment bounds")


def arm_states(data: MatchData, cfg: GestureConfig) -> tuple[NDArray[np.int8], NDArray[np.bool_]]:
    """0 neither, 1 left, 2 right, 3 both; separate fully-observed mask."""
    states = np.zeros(data.active.shape, dtype=np.int8)
    observed_sides: list[NDArray[np.bool_]] = []
    for side, code in (("l", 1), ("r", 2)):
        ids = [JOINTS.index(side + part) for part in ("Shoulder", "Elbow", "Wrist")]
        xyz = data.joints[:, :, ids, :]
        quality = np.all(np.isfinite(xyz), axis=(-1, -2)) & np.all(
            data.errors[:, :, ids] <= cfg.max_joint_error_cm, axis=-1
        )
        lengths = np.linalg.norm(np.diff(xyz, axis=2), axis=-1)
        quality &= np.all((lengths >= cfg.min_segment_m) & (lengths <= cfg.max_segment_m), axis=-1)
        raised = (
            quality
            & (xyz[:, :, 2, 2] - xyz[:, :, 0, 2] >= cfg.wrist_above_shoulder_m)
            & (xyz[:, :, 2, 2] - xyz[:, :, 1, 2] >= cfg.wrist_above_elbow_m)
        )
        states[raised] += code
        observed_sides.append(quality)
    return states, observed_sides[0] & observed_sides[1]


def eligible_samples(data: MatchData) -> NDArray[np.bool_]:
    lookup = {int(pid): i for i, pid in enumerate(data.players)}
    holder_index = np.array([lookup.get(int(pid), -1) for pid in data.owner])
    safe = np.maximum(holder_index, 0)
    holder_valid = (holder_index >= 0) & data.active[np.arange(len(safe)), safe]
    same_team = data.teams[None, :] == data.teams[safe, None]
    return (
        data.active
        & data.outfield[None, :]
        & data.in_play[:, None]
        & (data.period[:, None] > 0)
        & holder_valid[:, None]
        & same_team
        & (data.players[None, :] != data.owner[:, None])
        & np.all(np.isfinite(data.xy), axis=-1)
    )


def detect(
    data: MatchData, match_id: int, cfg: GestureConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    states, fully_observed = arm_states(data, cfg)
    eligible = eligible_samples(data)
    proposed = (states > 0) & eligible
    events: list[dict[str, Any]] = []
    minimum_steps = int(np.ceil(cfg.min_duration_s * HZ - 1e-9))
    for p, player_id in enumerate(data.players):
        indices = np.flatnonzero(proposed[:, p])
        start = 0
        while start < len(indices):
            stop = start + 1
            while (
                stop < len(indices)
                and indices[stop] == indices[stop - 1] + 1
                and data.owner[indices[stop]] == data.owner[indices[start]]
                and data.period[indices[stop]] == data.period[indices[start]]
            ):
                stop += 1
            first, last = int(indices[start]), int(indices[stop - 1])
            if last - first >= minimum_steps:
                t = first + minimum_steps
                codes = states[first : last + 1, p]
                arm = (
                    "both"
                    if np.any(codes == 3)
                    else "left"
                    if np.all(codes == 1)
                    else "right"
                    if np.all(codes == 2)
                    else "alternating"
                )
                events.append(
                    {
                        "event_id": f"{match_id}-{int(player_id)}-{first * 2}",
                        "match_id": match_id,
                        "period": int(data.period[t]),
                        "player_id": int(player_id),
                        "holder_id": int(data.owner[t]),
                        "team_id": int(data.teams[p]),
                        "start_index": first,
                        "end_index": last,
                        "analysis_index": t,
                        "start_frame": first * 2,
                        "end_frame": last * 2,
                        "analysis_frame": t * 2,
                        "clock_s": float(data.clock[t]),
                        "duration_s": (last - first) / HZ,
                        "arms": arm,
                        "kind": "raised_arm_candidate",
                        "review_label": "unreviewed",
                        "x": float(data.xy[t, p, 0]),
                        "y": float(data.xy[t, p, 1]),
                    }
                )
            start = stop
    events.sort(key=lambda e: (e["analysis_index"], e["player_id"]))
    audit = {
        "match_id": match_id,
        "config": asdict(cfg),
        "eligible_player_samples": int(eligible.sum()),
        "eligible_both_arms_observed": int((eligible & fully_observed).sum()),
        "raised_player_samples": int(proposed.sum()),
        "candidate_episodes": len(events),
        "interpretation": "Geometric proposals, not confirmed requests for a pass",
    }
    return events, audit
