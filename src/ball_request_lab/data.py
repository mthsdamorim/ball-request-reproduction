"""Read locked raw sources on the exact shared 5 Hz grid; preserve missingness."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
JOINTS = (
    "nose",
    "neck",
    "lEye",
    "rEye",
    "lEar",
    "rEar",
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
    "midHip",
    "lHip",
    "rHip",
    "lKnee",
    "rKnee",
    "lAnkle",
    "rAnkle",
    "lHeel",
    "rHeel",
    "lBigToe",
    "rBigToe",
    "lSmallToe",
    "rSmallToe",
)
HZ = 5


@dataclass
class MatchData:
    xy: FloatArray
    detected: BoolArray
    active: BoolArray
    ball: FloatArray
    ball_detected: BoolArray
    joints: FloatArray
    errors: FloatArray
    period: IntArray
    clock: FloatArray
    owner: IntArray
    in_play: BoolArray
    players: IntArray
    teams: IntArray
    outfield: BoolArray

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: Path) -> MatchData:
        with np.load(path, allow_pickle=False) as archive:
            return cls(**{f.name: archive[f.name] for f in fields(cls)})


def clock_seconds(value: str | None) -> float:
    if value is None:
        return float("nan")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def metadata(root: Path, match_id: int) -> dict[str, Any]:
    return json.loads((root / "matches" / str(match_id) / f"{match_id}_match.json").read_text())


def event_rows(root: Path, match_id: int) -> list[dict[str, str]]:
    path = root / "matches" / str(match_id) / f"{match_id}_dynamic_events.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(root: Path, match_id: int, output: Path) -> dict[str, Any]:
    """Stream both sources; no upsampling or pose interpolation enters analysis."""
    meta = metadata(root, match_id)
    roster: list[dict[str, Any]] = meta["players"]
    count = len(roster)
    last = max(int(p["end_frame"]) for p in meta["match_periods"])
    n = last // 2 + 2
    index = {int(player["id"]): i for i, player in enumerate(roster)}
    a = MatchData(
        xy=np.full((n, count, 2), np.nan, dtype=np.float32),
        detected=np.zeros((n, count), dtype=bool),
        active=np.zeros((n, count), dtype=bool),
        ball=np.full((n, 2), np.nan, dtype=np.float32),
        ball_detected=np.zeros(n, dtype=bool),
        joints=np.full((n, count, len(JOINTS), 3), np.nan, dtype=np.float32),
        errors=np.full((n, count, len(JOINTS)), np.inf, dtype=np.float32),
        period=np.zeros(n, dtype=np.int64),
        clock=np.full(n, np.nan),
        owner=np.full(n, -1, dtype=np.int64),
        in_play=np.zeros(n, dtype=bool),
        players=np.array([p["id"] for p in roster], dtype=np.int64),
        teams=np.array([p["team_id"] for p in roster], dtype=np.int64),
        outfield=np.array([p["player_role"]["acronym"] != "GK" for p in roster]),
    )
    seen = np.zeros(n, dtype=bool)
    pose_seen = np.zeros(n, dtype=bool)
    base = root / "matches" / str(match_id)
    with (base / f"{match_id}_tracking_extrapolated.jsonl").open("rb") as handle:
        for line in handle:
            row: dict[str, Any] = orjson.loads(line)
            frame = int(row["frame"])
            if frame % 2:
                continue
            t = frame // 2
            if t >= n or seen[t]:
                raise ValueError(f"Invalid/duplicate tracking frame {frame}")
            seen[t] = True
            a.period[t] = row["period"] or 0
            a.clock[t] = clock_seconds(row["timestamp"])
            possession: dict[str, Any] = row.get("possession") or {}
            a.owner[t] = possession.get("player_id") or -1
            ball = row["ball_data"]
            if ball["x"] is not None and ball["y"] is not None:
                a.ball[t] = [ball["x"], ball["y"]]
                a.ball_detected[t] = bool(ball["is_detected"])
            for p in row["player_data"]:
                j = index[int(p["player_id"])]
                if p["x"] is not None and p["y"] is not None:
                    a.xy[t, j] = [p["x"], p["y"]]
                    a.detected[t, j] = bool(p["is_detected"])
    for j, p in enumerate(roster):
        for interval in p["playing_time"]["by_period"]:
            first = (int(interval["start_frame"]) + 1) // 2
            end = int(interval["end_frame"]) // 2
            a.active[first : end + 1, j] = True
    with (base / f"{match_id}_phases_of_play.csv").open() as handle:
        for row in csv.DictReader(handle):
            first = (int(row["frame_start"]) + 1) // 2
            end = int(row["frame_end"]) // 2
            a.in_play[first : end + 1] = True
    owner_disagreements = 0
    with (
        zipfile.ZipFile(root / "bodypose" / f"{match_id}.jsonl.zip") as archive,
        archive.open(f"{match_id}.jsonl") as handle,
    ):
        for source_line, line in enumerate(handle):
            row = orjson.loads(line)
            frame = int(row["frame"])
            if frame % 5:
                continue
            t = frame // 5
            if t >= n or not seen[t] or pose_seen[t]:
                raise ValueError(f"Unmatched/duplicate pose frame {frame}")
            pose_seen[t] = True
            clock = clock_seconds(row["timestamp"])
            if (
                (row["period"] or 0) != a.period[t]
                or np.isfinite(clock) != np.isfinite(a.clock[t])
                or (np.isfinite(clock) and abs(clock - a.clock[t]) > 0.001)
            ):
                raise ValueError(f"Pose/tracking clock mismatch at {frame}")
            possession = row.get("possession") or {}
            pose_owner = possession.get("player_id") or -1
            if pose_owner != a.owner[t]:
                owner_disagreements += 1
                # The independently generated sources disagree: do not pick a holder.
                a.owner[t] = -1
            for p in row["player_data"]:
                joints = p.get("joints")
                if not joints:
                    continue
                j = index[int(p["player_id"])]
                for k, name in enumerate(JOINTS):
                    joint = joints.get(name)
                    if joint is not None:
                        a.joints[t, j, k] = joint["xyz"]
                        a.errors[t, j, k] = joint["p90_mae_cm"]
            if source_line % 25000 == 0:
                print(f"{match_id}: {source_line:,} pose rows read", flush=True)
    if np.any(seen & (a.period > 0) & ~pose_seen):
        raise ValueError("Missing pose source row on shared grid")
    a.save(output)
    known = np.all(np.isfinite(a.joints[:, :, JOINTS.index("midHip")]), axis=-1)
    audit: dict[str, Any] = {
        "match_id": match_id,
        "sample_hz": HZ,
        "aligned_rows": int(pose_seen.sum()),
        "active_player_samples": int(a.active.sum()),
        "active_with_pose": int((a.active & known).sum()),
        "known_owner_rows": int((a.owner >= 0).sum()),
        "owner_disagreements": owner_disagreements,
        "clock_and_period_checked": True,
        "joint_order": JOINTS,
        "output": str(output),
    }
    output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2))
    return audit
