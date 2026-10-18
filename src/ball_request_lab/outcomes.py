"""Outcome labels for confirmed requests for the ball.

The SkillCorner dynamic-events feed is sampled at 10 Hz.  This module labels a
request only from the ball holder's *active* ``player_possession`` event.  A
pass target is read after that possession finishes, solely to produce an
outcome label; it is never an input feature for request detection, regional
control, or visibility models. In particular, a negative follow-up label must
not expose any later known target to features measured at the gesture frame.
"""

EVENTS_PER_SECOND = 10

Response = dict[str, str | int | float | None]


def classify_response(
    events: list[dict[str, str]],
    *,
    frame: int,
    period: int,
    holder_id: int,
    requester_id: int,
    horizon_s: float = 3.0,
    observed_until_frame: int | None = None,
) -> Response:
    """Classify the ball holder's terminal action after a confirmed gesture.

    ``events`` must contain rows from one match's dynamic-events CSV.  The
    active row is a ``player_possession`` whose interval includes ``frame`` and
    whose holder is ``player_id``. Possession events with another player or
    another period cannot determine this label. Multiple overlapping holder
    events are treated as unknown rather than resolved arbitrarily.

    ``observed_until_frame`` marks the last frame available to the caller.  If
    the horizon reaches beyond it before an action is observable, the result is
    ``censored``.  In a complete follow-up (the default), a possession that
    continues past the horizon is a valid ``no_action_within_horizon`` result.
    Frame bounds are inclusive: an action at exactly the horizon or exactly the
    observed boundary is observed.
    """
    if horizon_s < 0:
        raise ValueError("horizon_s must be non-negative")

    horizon_frames = round(horizon_s * EVENTS_PER_SECOND)
    horizon_end = frame + horizon_frames
    active_event = _active_possession(events, frame=frame, period=period, holder_id=holder_id)

    if active_event is None:
        return _response("unknown")

    possession_event_id = active_event.get("event_id") or None
    action_frame = _integer(active_event.get("frame_end"))
    if action_frame is None:
        return _response("unknown", possession_event_id=possession_event_id)

    action_is_observed = observed_until_frame is None or action_frame <= observed_until_frame
    action_is_within_horizon = action_frame <= horizon_end

    if action_is_observed and action_is_within_horizon:
        return _classified_action(
            active_event,
            possession_event_id=possession_event_id,
            action_frame=action_frame,
            request_frame=frame,
            requester_id=requester_id,
        )

    if observed_until_frame is not None and horizon_end > observed_until_frame:
        return _response(
            "censored",
            possession_event_id=possession_event_id,
        )

    return _response("no_action_within_horizon", possession_event_id=possession_event_id)


def _active_possession(
    events: list[dict[str, str]], *, frame: int, period: int, holder_id: int
) -> dict[str, str] | None:
    matches: list[dict[str, str]] = []
    for event in events:
        if event.get("event_type") != "player_possession":
            continue
        if _integer(event.get("period")) != period:
            continue
        if not _is_holder(event, holder_id):
            continue
        frame_start = _integer(event.get("frame_start"))
        frame_end = _integer(event.get("frame_end"))
        if frame_start is None or frame_end is None:
            continue
        if frame_start <= frame <= frame_end:
            matches.append(event)

    if len(matches) != 1:
        return None
    return matches[0]


def _is_holder(event: dict[str, str], holder_id: int) -> bool:
    return _integer(event.get("player_id")) == holder_id


def _classified_action(
    event: dict[str, str],
    *,
    possession_event_id: str | None,
    action_frame: int,
    request_frame: int,
    requester_id: int,
) -> Response:
    terminal_action = (event.get("end_type") or "").strip().lower()
    target_id = _integer(event.get("player_targeted_id"))
    pass_outcome = (event.get("pass_outcome") or "").strip() or None
    latency_s = (action_frame - request_frame) / EVENTS_PER_SECOND

    if terminal_action == "pass":
        if target_id is None:
            category = "unknown"
        elif target_id == requester_id:
            category = "pass_to_requester"
        else:
            category = "pass_elsewhere"
    elif terminal_action:
        category = "other_action"
    else:
        category = "unknown"

    return {
        "possession_event_id": possession_event_id,
        "response_category": category,
        "action_frame": action_frame,
        "latency_s": latency_s,
        "target_id": target_id,
        "pass_outcome": pass_outcome,
    }


def _response(category: str, *, possession_event_id: str | None = None) -> Response:
    return {
        "possession_event_id": possession_event_id,
        "response_category": category,
        "action_frame": None,
        "latency_s": None,
        "target_id": None,
        "pass_outcome": None,
    }


def _integer(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None
