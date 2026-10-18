"""Transparent ground-pass corridor margins for a fixed tracking snapshot.

This module reports arrival-time margins along an assumed straight, ground-level
pass.  It deliberately does not estimate pass-completion probabilities, a
receiver's first-touch readiness, player intent, or offside.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, hypot, isfinite
from typing import Any

from ball_request_lab.spatial import (
    DEFAULT_INTERCEPT_CONFIG,
    InterceptConfig,
    PlayerState,
    Vec2,
    time_to_intercept,
)


def ground_pass_profile(
    players: Sequence[PlayerState],
    origin: Vec2,
    target: Vec2,
    team_id: str,
    *,
    speed_mps: float,
    sample_spacing_m: float = 0.5,
    release_delay_s: float = 0.0,
    intercept: InterceptConfig = DEFAULT_INTERCEPT_CONFIG,
) -> dict[str, Any]:
    """Return opponent-arrival margins along one fixed straight ground pass.

    At distance ``d`` from the origin, the ball time is
    ``release_delay_s + d / speed_mps``.  The margin is the earliest opposing
    player's time-to-intercept minus that ball time.  Positive values mean that
    every supplied opposing player reaches the sampled point after the assumed
    ball arrival; they are not probabilities or guarantees of a successful
    pass.
    """

    _validate_vec2(origin, "origin")
    _validate_vec2(target, "target")
    if not team_id:
        msg = "team_id must be non-empty"
        raise ValueError(msg)
    if not isfinite(speed_mps) or speed_mps <= 0:
        msg = "speed_mps must be finite and positive"
        raise ValueError(msg)
    if not isfinite(sample_spacing_m) or sample_spacing_m <= 0:
        msg = "sample_spacing_m must be finite and positive"
        raise ValueError(msg)
    if not isfinite(release_delay_s) or release_delay_s < 0:
        msg = "release_delay_s must be finite and non-negative"
        raise ValueError(msg)
    for player in players:
        _validate_player(player)

    opponents = [player for player in players if player.team_id != team_id]
    if not opponents:
        msg = "at least one opposing player is required"
        raise ValueError(msg)

    dx = target.x - origin.x
    dy = target.y - origin.y
    length = hypot(dx, dy)
    if length == 0:
        msg = "origin and target must be distinct"
        raise ValueError(msg)

    segments = ceil(length / sample_spacing_m)
    points: list[dict[str, float | str]] = []
    for index in range(segments + 1):
        fraction = index / segments
        distance = length * fraction
        point = Vec2(origin.x + dx * fraction, origin.y + dy * fraction)
        ball_time = release_delay_s + distance / speed_mps
        opponent = min(
            opponents,
            key=lambda player: time_to_intercept(player, point, intercept),
        )
        opponent_arrival = time_to_intercept(opponent, point, intercept)
        points.append(
            {
                "x": point.x,
                "y": point.y,
                "distance_m": distance,
                "ball_time_s": ball_time,
                "opponent_arrival_s": opponent_arrival,
                "margin_s": opponent_arrival - ball_time,
                "opponent_id": opponent.player_id,
            }
        )

    critical = min(points, key=lambda point: float(point["margin_s"]))
    return {
        "speed_mps": speed_mps,
        "length_m": length,
        "flight_time_s": length / speed_mps,
        "minimum_margin_s": critical["margin_s"],
        "critical_player_id": critical["opponent_id"],
        "critical_distance_m": critical["distance_m"],
        "points": points,
    }


def _validate_vec2(value: Vec2, name: str) -> None:
    if not isfinite(value.x) or not isfinite(value.y):
        msg = f"{name} coordinates must be finite"
        raise ValueError(msg)


def _validate_player(player: PlayerState) -> None:
    if not player.player_id or not player.team_id:
        msg = "every player must have non-empty player_id and team_id"
        raise ValueError(msg)
    _validate_vec2(player.position, f"player {player.player_id} position")
    _validate_vec2(player.velocity, f"player {player.player_id} velocity")
