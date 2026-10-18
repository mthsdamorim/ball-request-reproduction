"""Describe the game context around an integrated case from source records.

The script adds no model. It reads the frozen integrated case, the provider's
roster, phases of play and dynamic events, and writes the descriptive facts a
football reader asks for: who plays where, which phase the provider labels, how
far the carrier's options are, who is nearest to each of them, which options the
provider lists with its own completion and threat estimates, and where the ball
went in the possessions that followed. Every number is a source value or a
distance between two source positions.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ball_request_lab.data import event_rows, metadata
from ball_request_lab.util import atomic_json


def frame_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(str(value)))


def number(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return round(float(str(value)), 4)


def sample_at(case: dict[str, Any], relative_s: float) -> dict[str, Any]:
    matches = [s for s in case["samples"] if abs(s["relative_s"] - relative_s) < 1e-6]
    if len(matches) != 1:
        raise ValueError(f"Expected one sample at {relative_s} s")
    return matches[0]


def roles(meta: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for player in meta["players"]:
        role = player["player_role"]
        out[int(player["id"])] = {
            "name": player["short_name"],
            "number": player["number"],
            "team_id": int(player["team_id"]),
            "role": role["name"],
            "acronym": role["acronym"],
            "position_group": role["position_group"],
            "started": player["start_time"] == "00:00:00",
        }
    return out


def distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def nearest_opponent(
    player: dict[str, Any], players: list[dict[str, Any]], team_id: int
) -> dict[str, Any]:
    opponents = [p for p in players if p["team_id"] != team_id]
    best = min(opponents, key=lambda p: distance(player, p))
    return {
        "id": best["id"],
        "name": best["name"],
        "number": best["number"],
        "distance_m": round(distance(player, best), 2),
        "detected": best["detected"],
    }


def phase_for_frame(data_dir: Path, match_id: int, frame: int) -> dict[str, Any]:
    path = data_dir / "matches" / str(match_id) / f"{match_id}_phases_of_play.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    hits = [
        row
        for row in rows
        if (frame_int(row["frame_start"]) or 0) <= frame <= (frame_int(row["frame_end"]) or -1)
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one phase of play covering frame {frame}")
    row = hits[0]
    keys = [
        "index",
        "frame_start",
        "frame_end",
        "time_start",
        "time_end",
        "duration",
        "attacking_side",
        "team_in_possession_shortname",
        "team_in_possession_phase_type",
        "team_out_of_possession_phase_type",
        "n_player_possessions_in_phase",
        "team_possession_loss_in_phase",
        "team_possession_lead_to_shot",
        "third_start",
        "channel_start",
        "third_end",
        "channel_end",
    ]
    return {key: row[key] for key in keys}


def possession_chain(
    rows: list[dict[str, str]], start_frame: int, count: int
) -> list[dict[str, Any]]:
    possessions = sorted(
        (
            row
            for row in rows
            if row["event_type"] == "player_possession"
            and (frame_int(row["frame_start"]) or -1) >= start_frame
        ),
        key=lambda row: frame_int(row["frame_start"]) or 0,
    )
    chain: list[dict[str, Any]] = []
    for row in possessions[:count]:
        start = frame_int(row["frame_start"])
        options = [
            {
                "name": option["player_name"],
                "xpass_completion": number(option.get("xpass_completion")),
                "xthreat": number(option.get("xthreat")),
                "pass_direction": option.get("pass_direction") or None,
                "pass_range": option.get("pass_range") or None,
            }
            for option in rows
            if option["event_type"] == "passing_option"
            and frame_int(option.get("associated_player_possession_frame_start")) == start
        ]
        chain.append(
            {
                "frame_start": start,
                "frame_end": frame_int(row["frame_end"]),
                "player": row["player_name"],
                "team": row["team_shortname"],
                "x_start": number(row["x_start"]),
                "y_start": number(row["y_start"]),
                "pass_outcome": row.get("pass_outcome") or None,
                "pass_target": row.get("player_targeted_name") or None,
                "pass_direction": row.get("pass_direction") or None,
                "pass_range": row.get("pass_range") or None,
                "provider_passing_options": options,
            }
        )
    return chain


def describe(
    case_path: Path, data_dir: Path, match_id: int, follow: int, output_path: Path
) -> dict[str, Any]:
    """Write the descriptive context JSON for one integrated case and return it."""
    case = json.loads(case_path.read_text())
    info = case["case"]
    team_id = int(info["team_id"])
    meta = metadata(data_dir, match_id)
    role_by_id = roles(meta)
    sample = sample_at(case, 0.4)
    players = sample["players"]
    by_id = {int(p["id"]): p for p in players}
    caller = by_id[int(info["caller_id"])]
    holder = by_id[int(info["holder_id"])]

    rows = event_rows(data_dir, match_id)
    holder_possessions = [
        row
        for row in rows
        if row["event_type"] == "player_possession"
        and (frame_int(row["frame_start"]) or 10**9)
        <= info["action_frame"]
        <= (frame_int(row["frame_end"]) or -1)
        and row["player_name"] == info["holder_name"]
    ]
    if len(holder_possessions) != 1:
        raise ValueError("Expected one holder possession covering the action frame")
    possession_start = frame_int(holder_possessions[0]["frame_start"]) or 0
    chain = possession_chain(rows, possession_start, follow)
    target_name = chain[0]["pass_target"]
    target = next(p for p in players if p["name"] == target_name)

    def describe(player: dict[str, Any]) -> dict[str, Any]:
        role = role_by_id[int(player["id"])]
        return {
            "id": player["id"],
            "name": player["name"],
            "number": player["number"],
            "role": role["role"],
            "acronym": role["acronym"],
            "x": round(player["x"], 2),
            "y": round(player["y"], 2),
            "detected": player["detected"],
            "distance_from_holder_m": round(distance(holder, player), 2),
            "distance_from_ball_m": round(
                math.hypot(player["x"] - sample["ball"]["x"], player["y"] - sample["ball"]["y"]),
                2,
            ),
            "nearest_opponent": nearest_opponent(player, players, team_id),
        }

    lineup = [
        {
            "name": p["name"],
            "number": p["number"],
            "team_id": p["team_id"],
            "acronym": role_by_id[int(p["id"])]["acronym"],
            "role": role_by_id[int(p["id"])]["role"],
            "x": round(p["x"], 2),
            "y": round(p["y"], 2),
            "detected": p["detected"],
        }
        for p in sorted(players, key=lambda p: (p["team_id"] != team_id, p["x"]))
    ]
    opponents_ahead = [
        p for p in players if p["team_id"] != team_id and sample["ball"]["x"] < p["x"] < caller["x"]
    ]
    output = {
        "schema": "ball-request-lab/case-context/v1",
        "case_event_id": info["event_id"],
        "match_id": match_id,
        "reference_sample": {
            "relative_s": sample["relative_s"],
            "frame": sample["frame"],
            "source_clock_s": sample["source_clock_s"],
            "ball": {
                k: round(v, 2) if isinstance(v, float) else v for k, v in sample["ball"].items()
            },
            "pitch_length_m": info["pitch_length"],
            "pitch_width_m": info["pitch_width"],
        },
        "phase_of_play": phase_for_frame(data_dir, match_id, sample["frame"]),
        "holder": describe(holder),
        "caller": describe(caller),
        "observed_target": describe(target),
        "opponents_between_ball_and_caller_x": [
            {
                "name": p["name"],
                "number": p["number"],
                "x": round(p["x"], 2),
                "y": round(p["y"], 2),
            }
            for p in sorted(opponents_ahead, key=lambda p: p["x"])
        ],
        "lineup_at_reference": lineup,
        "possession_chain": chain,
        "notes": [
            "Roles are the provider's lineup roles for the match, not positions inferred here.",
            "xpass_completion and xthreat are the provider's own estimates; "
            "this project did not compute or validate them.",
            "Distances are straight-line metres between source positions at the reference sample.",
            "Nothing here enters the pre-action measurements of the integrated case.",
        ],
    }
    atomic_json(output_path, output)
    return output
