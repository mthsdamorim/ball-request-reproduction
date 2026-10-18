"""Build the examined case (de Jesus, match 1996435) from the frozen analysis."""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from ball_request_lab.analysis import face_joints
from ball_request_lab.data import MatchData, metadata
from ball_request_lab.integrated_case import build_case
from ball_request_lab.spatial import HeadGeometryConfig, estimate_head_heading
from ball_request_lab.util import atomic_json, file_hash


def require_adjudication(ledger: list[dict[str, Any]], event_id: str) -> None:
    """The case is built only for an event adjudicated as an identified request."""
    row = next((item for item in ledger if item.get("event_id") == event_id), None)
    if (
        row is None
        or row.get("label") != "probable_request"
        or row.get("evaluable_identity") is not True
    ):
        raise ValueError("The selected event needs a probable_request adjudication")


def build_examined_case(
    *,
    data_dir: Path,
    analyzed: Path,
    ledger: Path,
    case_config: Path,
    frozen_config: Path,
    output: Path,
) -> dict[str, Any]:
    """Write ``case.json`` and ``summary.json`` for the event named in the config."""
    if output.exists():
        raise FileExistsError(f"Case output is immutable; remove it first: {output}")
    requested = tomllib.loads(case_config.read_text())
    frozen = tomllib.loads(frozen_config.read_text())
    config = {**frozen, **requested}
    event_id = str(requested["case"]["event_id"])
    event = next(
        (row for row in json.loads(analyzed.read_text()) if row["event_id"] == event_id),
        None,
    )
    if event is None:
        raise ValueError(f"event_id {event_id} is absent from {analyzed}")
    require_adjudication(json.loads(ledger.read_text()), event_id)
    match_id = int(event["match_id"])
    data = MatchData.load(data_dir / "prepared" / f"{match_id}.npz")
    meta = metadata(data_dir, match_id)
    config["pitch"] = {
        "length": float(meta["pitch_length"]),
        "width": float(meta["pitch_width"]),
    }
    roster = {int(row["id"]): row for row in meta["players"]}
    response = event.get("response", {})
    case = build_case(data, event, config, roster=roster, action_frame=response.get("action_frame"))
    outcome = case["case"]["outcome_label"]
    if outcome == "pass_elsewhere":
        recipient = roster.get(response.get("target_id"), {})
        outcome = (
            f"Pass to {recipient.get('short_name', 'another teammate')} "
            f"#{recipient.get('number', '?')} at "
            f"+{case['case']['action_relative_s']:.2f} s."
        )
    case["case"].update(
        {
            "outcome_label": outcome,
            "pitch_length": float(meta["pitch_length"]),
            "pitch_width": float(meta["pitch_width"]),
            "source_match": f"{meta['home_team']['name']} v {meta['away_team']['name']}",
        }
    )
    atomic_json(output / "case.json", case)
    atomic_json(output / "summary.json", case["summary"])
    atomic_json(
        output / "provenance.json",
        {
            "event_id": event_id,
            "inputs_sha256": {
                "prepared": file_hash(data_dir / "prepared" / f"{match_id}.npz"),
                analyzed.name: file_hash(analyzed),
                ledger.name: file_hash(ledger),
                case_config.name: file_hash(case_config),
                frozen_config.name: file_hash(frozen_config),
            },
            "limitations": [
                "Potential view is not observed attention or perception.",
                "Arrival control and corridor margins are not calibrated probabilities.",
                "Offside, body orientation, ball deceleration and first touch are not evaluated.",
            ],
        },
    )
    return case


def _bearing(origin: np.ndarray[Any, Any], target: np.ndarray[Any, Any]) -> float:
    vector = target - origin
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def _separation(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def requester_head_bearings(
    data_dir: Path, analyzed: Path, event_id: str, config_path: Path
) -> dict[str, Any]:
    """Angles between a requester's own head direction and two other players.

    For each 5 Hz sample of the episode with a usable head direction, report the
    angle from that direction to the ball holder and to the requester's own
    goalkeeper. This regenerates the article's side remark about H. Van Der
    Saag's gesture admitting a second reading (an instruction to his goalkeeper).
    """
    event = next(row for row in json.loads(analyzed.read_text()) if row["event_id"] == event_id)
    match_id = int(event["match_id"])
    data = MatchData.load(data_dir / "prepared" / f"{match_id}.npz")
    meta = metadata(data_dir, match_id)
    lookup = {int(pid): i for i, pid in enumerate(data.players)}
    requester, holder = lookup[event["player_id"]], lookup[event["holder_id"]]
    keepers = [
        lookup[int(p["id"])]
        for p in meta["players"]
        if p["player_role"]["acronym"] == "GK" and int(p["team_id"]) == event["team_id"]
    ]
    if len(keepers) != 1:
        raise ValueError("Expected exactly one goalkeeper for the requester's team")
    keeper = keepers[0]
    head_config = HeadGeometryConfig(
        **{
            key: value
            for key, value in tomllib.loads(config_path.read_text())["vision"].items()
            if key != "sensitivity_half_angles_deg"
        }
    )
    samples: list[dict[str, Any]] = []
    for t in range(int(event["start_index"]), int(event["end_index"]) + 1):
        heading = estimate_head_heading(face_joints(data, t, requester), head_config)
        if heading is None:
            continue
        head = math.degrees(math.atan2(heading.direction.y, heading.direction.x))
        origin = data.xy[t, requester]
        samples.append(
            {
                "index": t,
                "relative_s": round((t - int(event["start_index"])) / 5, 1),
                "clock_s": round(float(data.clock[t]), 1),
                "to_holder_deg": round(_separation(_bearing(origin, data.xy[t, holder]), head), 1),
                "to_goalkeeper_deg": round(
                    _separation(_bearing(origin, data.xy[t, keeper]), head), 1
                ),
            }
        )
    confirmation = next(s for s in samples if s["index"] == int(event["analysis_index"]))
    return {
        "event_id": event_id,
        "requester": event["player_name"],
        "holder": event["holder_name"],
        "goalkeeper": meta["players"][keepers[0]]["short_name"],
        "confirmation_sample": confirmation,
        "samples": samples,
    }
