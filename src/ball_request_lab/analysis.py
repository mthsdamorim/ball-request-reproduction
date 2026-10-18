"""Join gesture episodes to spatial context, potential view and observed actions."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ball_request_lab.data import HZ, JOINTS, MatchData, event_rows, metadata
from ball_request_lab.outcomes import classify_response
from ball_request_lab.spatial import (
    HeadGeometryConfig,
    InterceptConfig,
    Joint3D,
    Occluder,
    PitchBounds,
    PlayerState,
    RegionalControlConfig,
    Vec2,
    ViewStatus,
    potential_view,
    regional_team_control,
)


def face_joints(data: MatchData, t: int, p: int) -> dict[str, Joint3D]:
    result: dict[str, Joint3D] = {}
    for name in ("nose", "lEye", "rEye", "lEar", "rEar"):
        j = JOINTS.index(name)
        xyz = data.joints[t, p, j]
        error = float(data.errors[t, p, j])
        if np.all(np.isfinite(xyz)) and np.isfinite(error):
            result[name] = Joint3D(float(xyz[0]), float(xyz[1]), float(xyz[2]), error / 100)
    return result


def snapshot(data: MatchData, t: int) -> tuple[list[PlayerState], int]:
    states: list[PlayerState] = []
    missing_velocity = 0
    for p in np.flatnonzero(data.active[t]):
        xy = data.xy[t, p]
        if not np.all(np.isfinite(xy)):
            continue
        v = np.zeros(2)
        prior = t - 2
        if (
            prior >= 0
            and data.period[prior] == data.period[t]
            and data.active[prior, p]
            and np.all(np.isfinite(data.xy[prior, p]))
        ):
            v = (xy - data.xy[prior, p]) / (2 / HZ)
        else:
            missing_velocity += 1
        states.append(
            PlayerState(
                str(data.players[p]),
                str(data.teams[p]),
                Vec2(float(xy[0]), float(xy[1])),
                Vec2(float(v[0]), float(v[1])),
            )
        )
    return states, missing_velocity


def view_during_request(
    data: MatchData,
    episode: dict[str, Any],
    cfg: HeadGeometryConfig,
    horizon_s: float,
    action_frame: int | None = None,
) -> dict[str, Any]:
    lookup = {int(pid): i for i, pid in enumerate(data.players)}
    caller = lookup[episode["player_id"]]
    holder = lookup[episode["holder_id"]]
    first = episode["start_index"]
    last = min(episode["end_index"], episode["analysis_index"] + int(horizon_s * HZ))
    if action_frame is not None:
        last = min(last, (action_frame - 1) // 2)
    statuses: Counter[str] = Counter()
    occluded = 0
    observations: list[dict[str, Any]] = []
    for t in range(first, last + 1):
        positions = data.xy[t]
        if not np.all(np.isfinite(positions[[holder, caller]])):
            statuses["unknown"] += 1
            continue
        observer = Vec2(*map(float, positions[holder]))
        target = Vec2(*map(float, positions[caller]))
        blockers = [
            Occluder(str(data.players[p]), Vec2(*map(float, positions[p])))
            for p in np.flatnonzero(data.active[t])
            if np.all(np.isfinite(positions[p]))
        ]
        result = potential_view(
            observer,
            target,
            face_joints(data, t, holder),
            cfg,
            observer_id=str(episode["holder_id"]),
            target_id=str(episode["player_id"]),
            occluders=blockers,
        )
        statuses[result.status.value] += 1
        occluded += int(result.occlusion_flag)
        observations.append({"frame": t * 2, **asdict(result)})
    if statuses[ViewStatus.IN_VIEW]:
        status = "in_view"
    elif statuses["unknown"] or not statuses:
        status = "unknown"
    else:
        status = "out_of_view"
    known = statuses["in_view"] + statuses["out_of_view"]
    return {
        "status": status,
        "meaning": "At least one usable in-view sample before the holder action",
        "total_samples": sum(statuses.values()),
        "known_samples": known,
        "in_view_samples": statuses["in_view"],
        "in_view_fraction_known": statuses["in_view"] / known if known else None,
        "stop_before_action_frame": action_frame,
        "sample_counts": dict(statuses),
        "occlusion_flag_samples": occluded,
        "observed_interval_s": max(0, last - first) / HZ,
        "observations": observations,
    }


def analyze_match(
    root: Path, run: Path, match_id: int, config: dict[str, Any]
) -> list[dict[str, Any]]:
    data = MatchData.load(root / "prepared" / f"{match_id}.npz")
    meta = metadata(root, match_id)
    roster = {int(p["id"]): p for p in meta["players"]}
    events = event_rows(root, match_id)
    episodes: list[dict[str, Any]] = json.loads((run / f"{match_id}-candidates.json").read_text())
    region_options = config["region"]
    vision_options = {
        k: v for k, v in config["vision"].items() if k != "sensitivity_half_angles_deg"
    }
    head_cfg = HeadGeometryConfig(**vision_options)
    intercept = InterceptConfig(**config["intercept"])
    horizon = float(config["response"]["horizon_s"])
    length, width = float(meta["pitch_length"]), float(meta["pitch_width"])
    bounds = PitchBounds(-length / 2, length / 2, -width / 2, width / 2)
    output: list[dict[str, Any]] = []
    for episode in episodes:
        t = int(episode["analysis_index"])
        p = int(np.flatnonzero(data.players == episode["player_id"])[0])
        xy = Vec2(*map(float, data.xy[t, p]))
        players, missing_velocity = snapshot(data, t)
        complete = len(players) == int(data.active[t].sum())
        opponents = [s for s in players if s.team_id != str(episode["team_id"])]
        nearest = min((s.position.distance_to(xy) for s in opponents), default=None)
        radii = sorted(set([region_options["radius_m"], *region_options["sensitivity_radii_m"]]))
        regional: dict[str, Any] = {}
        if complete and opponents:
            for radius in radii:
                summary = regional_team_control(
                    players,
                    xy,
                    str(episode["team_id"]),
                    RegionalControlConfig(
                        float(radius), region_options["sample_spacing_m"], bounds
                    ),
                    intercept,
                )
                regional[str(float(radius))] = asdict(summary)
        primary = regional.get(str(float(region_options["radius_m"])))
        favorable = (
            None
            if primary is None
            else primary["mean_arrival_control_proxy"] >= region_options["favorable_mean_threshold"]
        )
        last_frame = max(
            int(period["end_frame"])
            for period in meta["match_periods"]
            if int(period["period"]) == episode["period"]
        )
        response = classify_response(
            events,
            frame=t * 2,
            period=episode["period"],
            holder_id=episode["holder_id"],
            requester_id=episode["player_id"],
            horizon_s=horizon,
            observed_until_frame=last_frame,
        )
        action_frame = response["action_frame"]
        cutoff = int(action_frame) if action_frame is not None else None
        view = view_during_request(data, episode, head_cfg, horizon, cutoff)
        vision_sensitivity = {
            str(float(angle)): view_during_request(
                data,
                episode,
                replace(head_cfg, peripheral_half_angle_deg=float(angle)),
                horizon,
                cutoff,
            )["status"]
            for angle in config["vision"]["sensitivity_half_angles_deg"]
        }
        player = roster[episode["player_id"]]
        holder = roster[episode["holder_id"]]
        output.append(
            {
                **episode,
                "player_name": player["short_name"],
                "shirt_number": player["number"],
                "holder_name": holder["short_name"],
                "nearest_opponent_m": nearest,
                "active_players": len(players),
                "missing_velocity_players": missing_velocity,
                "detected_fraction": float(data.detected[t, data.active[t]].mean()),
                "region_uses_extrapolated": bool(np.any(~data.detected[t, data.active[t]])),
                "region": primary,
                "region_sensitivity": regional,
                "region_favorable": favorable,
                "vision": view,
                "vision_sensitivity": vision_sensitivity,
                "response": response,
            }
        )
    return output
