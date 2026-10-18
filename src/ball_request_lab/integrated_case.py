"""Compose one reviewed request into conservative, time-aligned case data.

The output deliberately keeps three questions separate: local arrival control,
potential head-centred view, and a mechanical ground-pass corridor.  It does not
produce a combined score or a pass probability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import ceil, isfinite, sqrt
from typing import Any, cast

import numpy as np

from ball_request_lab.analysis import face_joints, snapshot
from ball_request_lab.data import HZ, JOINTS, MatchData
from ball_request_lab.gestures import GestureConfig, arm_states
from ball_request_lab.pass_corridor import ground_pass_profile
from ball_request_lab.spatial import (
    HeadGeometryConfig,
    InterceptConfig,
    Occluder,
    PitchBounds,
    RegionalControlConfig,
    Vec2,
    arrival_control_proxy,
    potential_view,
    regional_team_control,
)


def build_case(
    data: MatchData,
    event: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    roster: Mapping[int, Mapping[str, Any]] | None = None,
    action_frame: int | None = None,
) -> dict[str, Any]:
    """Build serialisable samples for one event without looking past its action.

    ``event`` comes from the frozen analysed run.  The contextual samples are
    intentionally rendered with unknown possession/view: a future holder is not
    carried backwards into the context window.
    """

    case_cfg = _mapping(config, "case")
    pass_cfg = _mapping(config, "pass")
    visual_cfg = _mapping(config, "visual")
    region_cfg = _mapping(config, "region")
    intercept_cfg = InterceptConfig(**dict(_mapping(config, "intercept")))
    vision_cfg = _head_config(config)
    caller_id = int(event["player_id"])
    holder_id = int(event["holder_id"])
    team_id = str(event["team_id"])
    start = int(event["start_index"])
    end = int(event["end_index"])
    cutoff = int(action_frame) if action_frame is not None else None
    last = end if cutoff is None else min(end, (cutoff - 1) // 2)
    context_steps = round(float(case_cfg.get("context_before_s", 1.0)) * HZ)
    first = max(0, start - context_steps)
    caller_index = _player_index(data, caller_id)
    holder_index = _player_index(data, holder_id)
    primary_angle = float(visual_cfg.get("primary_angle", 90.0))
    angles = [float(value) for value in visual_cfg["angles"]]
    radii = [float(value) for value in visual_cfg["radii"]]
    speeds = [float(value) for value in pass_cfg["speeds_mps"]]
    arm, _ = arm_states(data, GestureConfig(**dict(_mapping(config, "gesture"))))

    samples: list[dict[str, Any]] = []
    for t in range(first, last + 1):
        frame = t * 2
        is_gesture = start <= t <= end
        before_action = cutoff is None or frame < cutoff
        holder_has_ball = bool(is_gesture and before_action and data.owner[t] == holder_id)
        players, missing_velocity = snapshot(data, t)
        active_indices = np.flatnonzero(data.active[t])
        missing_coordinates = [
            int(data.players[index])
            for index in active_indices
            if not np.all(np.isfinite(data.xy[t, index]))
        ]
        extrapolated = [
            int(data.players[index])
            for index in active_indices
            if np.all(np.isfinite(data.xy[t, index])) and not data.detected[t, index]
        ]
        tracking_status, strict_status = _quality_status(
            missing_coordinates, extrapolated, missing_velocity
        )
        row: dict[str, Any] = {
            "frame": frame,
            "relative_s": (t - start) / HZ,
            "source_clock_s": float(data.clock[t]),
            "is_gesture": is_gesture,
            "holder_has_ball": holder_has_ball,
            "players": _players_row(data, t, players, roster),
            "ball": _ball_row(data, t),
            "pose": _local_pose(data, t, caller_index),
            "arm_state": _arm_name(int(arm[t, caller_index])),
            "wrist_height_m": _wrist_height(data, t, caller_index),
            "view": _unknown_view(),
            "control": None,
            "passes": None,
            "quality": {
                "active_players": len(active_indices),
                "detected_players": int(sum(data.detected[t, active_indices])),
                "extrapolated_ids": extrapolated,
                "missing_coordinate_ids": missing_coordinates,
                "missing_velocity_players": missing_velocity,
                "tracking_status": tracking_status,
                "strict_status": strict_status,
            },
        }
        can_model_positions = not missing_coordinates and len(players) == len(active_indices)
        if can_model_positions:
            row["control"] = _regional_controls(
                players,
                _vec(data.xy[t, caller_index]),
                team_id,
                radii,
                float(region_cfg.get("sample_spacing_m", 1.0)),
                _bounds(config),
                intercept_cfg,
            )
        if holder_has_ball:
            row["view"] = _view(
                data,
                t,
                holder_index,
                caller_index,
                players,
                holder_id,
                caller_id,
                vision_cfg,
                primary_angle,
            )
            if _can_profile(data, t, caller_index, holder_index, can_model_positions):
                origin = _vec(data.ball[t])
                target = _vec(data.xy[t, caller_index])
                row["passes"] = {
                    _number_key(speed): ground_pass_profile(
                        players,
                        origin,
                        target,
                        team_id,
                        speed_mps=speed,
                        sample_spacing_m=float(pass_cfg.get("sample_spacing_m", 0.5)),
                        release_delay_s=float(pass_cfg.get("release_delay_s", 0.0)),
                        intercept=intercept_cfg,
                    )
                    for speed in speeds
                }
        samples.append(row)

    case = _case_header(event, roster, data, caller_index, holder_index, cutoff)
    summary = _summary(
        samples,
        angles,
        radii,
        speeds,
        data,
        caller_index,
        holder_index,
        team_id,
        config,
        intercept_cfg,
    )
    return _json_safe(
        {
            "case": case,
            "config": {
                "angles": angles,
                "radii": radii,
                "speeds": speeds,
                "favorable_threshold": float(region_cfg.get("favorable_mean_threshold", 0.65)),
                "legal_status": "not_evaluated",
                "model_scope": "mechanical ground-pass scenarios; no probability",
            },
            "samples": samples,
            "summary": summary,
        }
    )


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    candidate = value.get(key, {})
    if not isinstance(candidate, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return cast(Mapping[str, Any], candidate)


def _head_config(config: Mapping[str, Any]) -> HeadGeometryConfig:
    values = dict(_mapping(config, "vision"))
    values.pop("sensitivity_half_angles_deg", None)
    return HeadGeometryConfig(**values)


def _player_index(data: MatchData, player_id: int) -> int:
    indices = np.flatnonzero(data.players == player_id)
    if len(indices) != 1:
        raise ValueError(f"player_id {player_id} is not unique in prepared data")
    return int(indices[0])


def _vec(value: np.ndarray[Any, Any]) -> Vec2:
    return Vec2(float(value[0]), float(value[1]))


def _number_key(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _quality_status(
    missing_coordinates: list[int], extrapolated: list[int], missing_velocity: int
) -> tuple[str, str]:
    if missing_coordinates:
        return (
            "unknown_missing_active_coordinates",
            "unknown_missing_active_coordinates",
        )
    if extrapolated:
        return (
            "conditional_all_xy_extrapolated_tracking",
            "unknown_extrapolated_tracking",
        )
    if missing_velocity:
        return (
            "conditional_zero_velocity_for_missing_history",
            "unknown_missing_velocity",
        )
    return "observed_tracking", "available"


def _players_row(
    data: MatchData,
    t: int,
    states: list[Any],
    roster: Mapping[int, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    by_id = {int(state.player_id): state for state in states}
    rows: list[dict[str, Any]] = []
    for index in np.flatnonzero(data.active[t]):
        player_id = int(data.players[index])
        state = by_id.get(player_id)
        person = None if roster is None else roster.get(player_id)
        rows.append(
            {
                "id": player_id,
                "name": None if person is None else person.get("short_name"),
                "number": None if person is None else person.get("number"),
                "team_id": int(data.teams[index]),
                "x": None if state is None else state.position.x,
                "y": None if state is None else state.position.y,
                "vx": None if state is None else state.velocity.x,
                "vy": None if state is None else state.velocity.y,
                "detected": bool(data.detected[t, index]),
            }
        )
    return rows


def _ball_row(data: MatchData, t: int) -> dict[str, Any] | None:
    if not np.all(np.isfinite(data.ball[t])):
        return None
    return {
        "x": float(data.ball[t, 0]),
        "y": float(data.ball[t, 1]),
        "detected": bool(data.ball_detected[t]),
    }


def _local_pose(data: MatchData, t: int, player: int) -> list[dict[str, Any]]:
    shoulder_left = data.joints[t, player, JOINTS.index("lShoulder")]
    shoulder_right = data.joints[t, player, JOINTS.index("rShoulder")]
    hip = data.joints[t, player, JOINTS.index("midHip")]
    usable = all(np.all(np.isfinite(point)) for point in (shoulder_left, shoulder_right, hip))
    lateral_x = float(shoulder_right[0] - shoulder_left[0]) if usable else 0.0
    lateral_y = float(shoulder_right[1] - shoulder_left[1]) if usable else 0.0
    magnitude = sqrt(lateral_x**2 + lateral_y**2)
    usable = usable and magnitude > 0
    if not usable:
        return [{"name": name, "x": None, "y": None, "z": None, "valid": False} for name in JOINTS]
    lateral_x /= magnitude
    lateral_y /= magnitude
    forward_x, forward_y = -lateral_y, lateral_x
    nose = data.joints[t, player, JOINTS.index("nose")]
    shoulder_middle_x = float((shoulder_left[0] + shoulder_right[0]) / 2)
    shoulder_middle_y = float((shoulder_left[1] + shoulder_right[1]) / 2)
    if np.all(np.isfinite(nose)) and (
        (float(nose[0]) - shoulder_middle_x) * forward_x
        + (float(nose[1]) - shoulder_middle_y) * forward_y
        < 0
    ):
        forward_x, forward_y = -forward_x, -forward_y
    rows: list[dict[str, Any]] = []
    for name, point in zip(JOINTS, data.joints[t, player], strict=True):
        valid = bool(np.all(np.isfinite(point)))
        relative_x = float(point[0] - hip[0]) if valid else None
        relative_y = float(point[1] - hip[1]) if valid else None
        relative_z = float(point[2] - hip[2]) if valid else None
        rows.append(
            {
                "name": name,
                "x": None
                if relative_x is None or relative_y is None
                else relative_x * lateral_x + relative_y * lateral_y,
                "y": None
                if relative_x is None or relative_y is None
                else relative_x * forward_x + relative_y * forward_y,
                "z": relative_z,
                "valid": valid,
            }
        )
    return rows


def _arm_name(value: int) -> str:
    return {0: "none", 1: "left", 2: "right", 3: "both"}.get(value, "unknown")


def _wrist_height(data: MatchData, t: int, player: int) -> dict[str, float | None]:
    shoulders = {
        side: data.joints[t, player, JOINTS.index(side + "Shoulder"), 2] for side in ("l", "r")
    }
    wrists = {side: data.joints[t, player, JOINTS.index(side + "Wrist"), 2] for side in ("l", "r")}
    return {
        side: None
        if not isfinite(float(shoulders[side])) or not isfinite(float(wrists[side]))
        else float(wrists[side] - shoulders[side])
        for side in ("l", "r")
    }


def _unknown_view() -> dict[str, Any]:
    return {
        "status": "unknown",
        "zone": None,
        "eccentricity_deg": None,
        "target_distance_m": None,
        "heading": None,
        "occlusion_flag": False,
    }


def _view(
    data: MatchData,
    t: int,
    holder: int,
    caller: int,
    players: list[Any],
    holder_id: int,
    caller_id: int,
    config: HeadGeometryConfig,
    angle: float,
) -> dict[str, Any]:
    if not (np.all(np.isfinite(data.xy[t, holder])) and np.all(np.isfinite(data.xy[t, caller]))):
        return _unknown_view()
    result = potential_view(
        _vec(data.xy[t, holder]),
        _vec(data.xy[t, caller]),
        face_joints(data, t, holder),
        config.__class__(**{**asdict(config), "peripheral_half_angle_deg": angle}),
        observer_id=str(holder_id),
        target_id=str(caller_id),
        occluders=[Occluder(str(state.player_id), state.position) for state in players],
    )
    return asdict(result)


def _bounds(config: Mapping[str, Any]) -> PitchBounds:
    pitch = _mapping(config, "pitch")
    length, width = float(pitch.get("length", 105.0)), float(pitch.get("width", 68.0))
    return PitchBounds(-length / 2, length / 2, -width / 2, width / 2)


def _regional_controls(
    players: list[Any],
    centre: Vec2,
    team_id: str,
    radii: list[float],
    spacing: float,
    bounds: PitchBounds,
    intercept: InterceptConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for radius in radii:
        summary = regional_team_control(
            players,
            centre,
            team_id,
            RegionalControlConfig(radius, spacing, bounds),
            intercept,
        )
        points: list[dict[str, float]] = []
        steps = ceil(radius / spacing)
        for x_step in range(-steps, steps + 1):
            for y_step in range(-steps, steps + 1):
                point = Vec2(centre.x + x_step * spacing, centre.y + y_step * spacing)
                if point.distance_to(centre) <= radius and bounds.contains(point):
                    points.append(
                        {
                            "x": point.x,
                            "y": point.y,
                            "value": arrival_control_proxy(
                                players, point, team_id, intercept
                            ).arrival_control_proxy,
                        }
                    )
        result[_number_key(radius)] = {
            "mean": summary.mean_arrival_control_proxy,
            "min": summary.min_arrival_control_proxy,
            "points": points,
        }
    return result


def _can_profile(data: MatchData, t: int, caller: int, holder: int, all_coordinates: bool) -> bool:
    return bool(
        all_coordinates
        and data.in_play[t]
        and data.ball_detected[t]
        and np.all(np.isfinite(data.ball[t]))
        and np.all(np.isfinite(data.xy[t, caller]))
        and np.all(np.isfinite(data.xy[t, holder]))
        and data.teams[caller] == data.teams[holder]
    )


def _case_header(
    event: Mapping[str, Any],
    roster: Mapping[int, Mapping[str, Any]] | None,
    data: MatchData,
    caller: int,
    holder: int,
    action_frame: int | None,
) -> dict[str, Any]:
    def person(player_id: int) -> tuple[str | None, int | None]:
        row = roster.get(player_id) if roster else None
        number = None if row is None else row.get("number")
        return (
            None if row is None else str(row.get("short_name")),
            None if number is None else int(number),
        )

    del caller, holder
    caller_name, caller_number = person(int(event["player_id"]))
    holder_name, holder_number = person(int(event["holder_id"]))
    return {
        "event_id": str(event["event_id"]),
        "caller_id": int(event["player_id"]),
        "caller_name": caller_name or event.get("player_name"),
        "caller_number": caller_number or event.get("shirt_number"),
        "holder_id": int(event["holder_id"]),
        "holder_name": holder_name or event.get("holder_name"),
        "holder_number": holder_number,
        "team_id": int(event["team_id"]),
        "pitch_length": None,
        "pitch_width": None,
        "start_frame": int(event["start_frame"]),
        "analysis_frame": int(event["analysis_frame"]),
        "action_frame": action_frame,
        "action_relative_s": None
        if action_frame is None
        else (action_frame - int(event["start_frame"])) / 10,
        "outcome_label": event.get("response", {}).get("response_category"),
        "source_match": int(event["match_id"]),
    }


def _summary(
    samples: list[dict[str, Any]],
    angles: list[float],
    radii: list[float],
    speeds: list[float],
    data: MatchData,
    caller: int,
    holder: int,
    team_id: str,
    config: Mapping[str, Any],
    intercept: InterceptConfig,
) -> dict[str, Any]:
    main = [row for row in samples if row["is_gesture"] and row["holder_has_ball"]]
    by_angle: dict[str, Any] = {}
    for angle in angles:
        statuses = [
            _view(
                data,
                row["frame"] // 2,
                holder,
                caller,
                snapshot(data, row["frame"] // 2)[0],
                int(data.players[holder]),
                int(data.players[caller]),
                _head_config(config),
                angle,
            )["status"]
            for row in main
        ]
        known = sum(status in {"in_view", "out_of_view"} for status in statuses)
        by_angle[_number_key(angle)] = {
            "known": known,
            "in_view": statuses.count("in_view"),
            "total": len(statuses),
        }
    by_radius: dict[str, Any] = {}
    for radius in radii:
        values = [
            row["control"][_number_key(radius)]["mean"]
            for row in main
            if row["control"] is not None
        ]
        by_radius[_number_key(radius)] = {
            "count": len(values),
            "mean_range": None if not values else [min(values), max(values)],
        }
    pass_summary: dict[str, Any] = {}
    pass_cfg = _mapping(config, "pass")
    verification_spacing = float(pass_cfg.get("verification_spacing_m", 0.25))
    release_delay = float(pass_cfg.get("release_delay_s", 0.0))
    for speed in speeds:
        key = _number_key(speed)
        profiles: list[tuple[int, dict[str, Any]]] = []
        for row in main:
            passes = row["passes"]
            if passes is not None:
                profiles.append((int(row["frame"]), cast(dict[str, Any], passes[key])))
        margins = [profile["minimum_margin_s"] for _, profile in profiles]
        verification: list[tuple[int, dict[str, Any]]] = []
        for row in main:
            if row["passes"] is None:
                continue
            t = row["frame"] // 2
            states, _ = snapshot(data, t)
            verification.append(
                (
                    int(row["frame"]),
                    ground_pass_profile(
                        states,
                        _vec(data.ball[t]),
                        _vec(data.xy[t, caller]),
                        team_id,
                        speed_mps=speed,
                        sample_spacing_m=verification_spacing,
                        release_delay_s=release_delay,
                        intercept=intercept,
                    ),
                )
            )
        coarse_by_frame = {frame: profile for frame, profile in profiles}
        deltas = [
            abs(
                float(profile["minimum_margin_s"])
                - float(coarse_by_frame[frame]["minimum_margin_s"])
            )
            for frame, profile in verification
        ]
        sign_agreement = [
            float(profile["minimum_margin_s"]) * float(coarse_by_frame[frame]["minimum_margin_s"])
            >= 0
            for frame, profile in verification
        ]
        pass_summary[key] = {
            "count": len(profiles),
            "minimum_margin_range_s": None if not margins else [min(margins), max(margins)],
            "critical_player_ids_by_frame": [
                {"frame": frame, "player_id": profile["critical_player_id"]}
                for frame, profile in profiles
            ],
            "discretization_sensitivity": {
                "coarse_spacing_m": float(pass_cfg.get("sample_spacing_m", 0.5)),
                "verification_spacing_m": verification_spacing,
                "maximum_abs_delta_s": None if not deltas else max(deltas),
                "sign_agreement": None if not sign_agreement else all(sign_agreement),
            },
        }
    return {
        "main_sample_count": len(main),
        "view_by_angle": by_angle,
        "control_by_radius": by_radius,
        "pass_by_speed": pass_summary,
        "legal_status": "not_evaluated",
        "interpretation": (
            "Positive corridor margins are model arrival margins, not pass safety or probability."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if isfinite(value) else None
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {str(key): _json_safe(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        items = cast(list[Any] | tuple[Any, ...], value)
        return [_json_safe(item) for item in items]
    return value
