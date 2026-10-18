"""Conservative spatial proxies for the ball-request experiment.

The functions here intentionally return *potential* visibility and an
arrival-time control proxy.  Neither is evidence of visual attention, a
completed pass, physical capacity, or tactical freedom.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import acos, ceil, exp, hypot, sqrt


@dataclass(frozen=True)
class Vec2:
    """A point or vector in pitch coordinates, in metres."""

    x: float
    y: float

    def distance_to(self, other: Vec2) -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Joint3D:
    """A body-pose landmark in global coordinates.

    ``error_m`` is the provider's point-level positional error when available.
    It is not an eye-tracking accuracy measure.
    """

    x: float
    y: float
    z: float
    error_m: float | None = None

    def __post_init__(self) -> None:
        if self.error_m is not None and self.error_m < 0:
            msg = "error_m must be non-negative when supplied"
            raise ValueError(msg)

    @property
    def xy(self) -> Vec2:
        return Vec2(self.x, self.y)


@dataclass(frozen=True)
class PlayerState:
    """One tracked player state at a common timestamp."""

    player_id: str
    team_id: str
    position: Vec2
    velocity: Vec2


@dataclass(frozen=True)
class Occluder:
    """A deliberately simple pitch-plane blocker used only as a flag."""

    player_id: str
    position: Vec2
    radius_m: float = 0.35

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            msg = "radius_m must be positive"
            raise ValueError(msg)


class HeadingSource(StrEnum):
    """The landmarks contributing to a head-centre direction proxy."""

    EARS_SIGNED_BY_NOSE = "ears_signed_by_nose"


class HeadingConfidence(StrEnum):
    """Status of confidence calibration, deliberately separate from geometry gates."""

    NOT_CALIBRATED = "not_calibrated"


class ViewStatus(StrEnum):
    """A three-state potential-visibility outcome."""

    UNKNOWN = "unknown"
    IN_VIEW = "in_view"
    OUT_OF_VIEW = "out_of_view"


class ViewZone(StrEnum):
    """Angular zone when a target is inside the estimated head-centred view."""

    CENTRAL = "central"
    PERIPHERAL = "peripheral"


@dataclass(frozen=True)
class HeadGeometryConfig:
    """Quality and angular assumptions for a head-centred visibility proxy.

    The 20/90-degree defaults are sensitivity hypotheses, not physiological
    calibration for football or an assertion of where a player attended.
    """

    central_half_angle_deg: float = 20.0
    peripheral_half_angle_deg: float = 90.0
    max_target_distance_m: float = 60.0
    min_ear_baseline_m: float = 0.08
    max_ear_baseline_m: float = 0.35
    max_relative_ear_error: float = 0.65
    min_nose_sign_component_m: float = 0.03

    def __post_init__(self) -> None:
        if not 0 < self.central_half_angle_deg <= self.peripheral_half_angle_deg:
            msg = "central_half_angle_deg must be positive and no larger than peripheral"
            raise ValueError(msg)
        if not 0 < self.peripheral_half_angle_deg <= 180:
            msg = "peripheral_half_angle_deg must be in (0, 180]"
            raise ValueError(msg)
        if self.max_target_distance_m <= 0:
            msg = "max_target_distance_m must be positive"
            raise ValueError(msg)
        if not 0 < self.min_ear_baseline_m < self.max_ear_baseline_m:
            msg = "ear baseline bounds must be positive and ordered"
            raise ValueError(msg)
        if self.max_relative_ear_error <= 0:
            msg = "max_relative_ear_error must be positive"
            raise ValueError(msg)
        if self.min_nose_sign_component_m <= 0:
            msg = "min_nose_sign_component_m must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class HeadHeading:
    """A quality-gated head-centre direction proxy, never an eye-gaze vector."""

    direction: Vec2
    source: HeadingSource
    ear_baseline_m: float
    nose_sign_component_m: float
    ear_landmark_error_m: float | None
    sign_landmark_error_m: float | None
    eye_directional_agreement_deg: float | None
    confidence: HeadingConfidence = HeadingConfidence.NOT_CALIBRATED


@dataclass(frozen=True)
class PotentialView:
    """Potential rather than observed visibility of a target from a player."""

    status: ViewStatus
    zone: ViewZone | None
    eccentricity_deg: float | None
    target_distance_m: float | None
    heading: HeadHeading | None
    occlusion_flag: bool


@dataclass(frozen=True)
class InterceptConfig:
    """Parameters for a transparent, uncalibrated arrival-time proxy."""

    reaction_time_s: float = 0.20
    max_speed_mps: float = 5.0
    max_acceleration_mps2: float = 3.0
    arrival_temperature_s: float = 0.45
    control_proxy_threshold: float = 0.50

    def __post_init__(self) -> None:
        if self.reaction_time_s < 0:
            msg = "reaction_time_s must be non-negative"
            raise ValueError(msg)
        if self.max_speed_mps <= 0 or self.max_acceleration_mps2 <= 0:
            msg = "maximum speed and acceleration must be positive"
            raise ValueError(msg)
        if self.arrival_temperature_s <= 0:
            msg = "arrival_temperature_s must be positive"
            raise ValueError(msg)
        if not 0 <= self.control_proxy_threshold <= 1:
            msg = "control_proxy_threshold must be in [0, 1]"
            raise ValueError(msg)


@dataclass(frozen=True)
class ArrivalControlProxy:
    """Closest-team arrival comparison at one pitch location.

    ``arrival_control_proxy`` is a logistic transform of the difference between
    closest-team arrivals.  It is not a calibrated probability of possession.
    """

    team_id: str
    arrival_control_proxy: float
    closest_team_arrival_s: float
    closest_opponent_arrival_s: float
    threshold_met: bool


@dataclass(frozen=True)
class PitchBounds:
    """Coordinate bounds used to clip regional samples."""

    x_min: float = 0.0
    x_max: float = 105.0
    y_min: float = 0.0
    y_max: float = 68.0

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            msg = "pitch bounds must have positive width and height"
            raise ValueError(msg)

    def contains(self, point: Vec2) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


@dataclass(frozen=True)
class RegionalControlConfig:
    """Sampling configuration for a local, pitch-clipped control summary."""

    radius_m: float = 4.0
    sample_spacing_m: float = 1.0
    pitch_bounds: PitchBounds = PitchBounds()

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or self.sample_spacing_m <= 0:
            msg = "radius_m and sample_spacing_m must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class RegionalControlSummary:
    """A sampled local-control summary; it is not a territorial guarantee."""

    team_id: str
    radius_m: float
    sample_count: int
    mean_arrival_control_proxy: float
    min_arrival_control_proxy: float
    threshold_fraction: float


DEFAULT_HEAD_GEOMETRY_CONFIG = HeadGeometryConfig()
DEFAULT_INTERCEPT_CONFIG = InterceptConfig()
DEFAULT_REGIONAL_CONTROL_CONFIG = RegionalControlConfig()


def estimate_head_heading(
    joints: Mapping[str, Joint3D],
    config: HeadGeometryConfig = DEFAULT_HEAD_GEOMETRY_CONFIG,
) -> HeadHeading | None:
    """Estimate a pitch-plane head direction only when face geometry is usable.

    A long left-ear to right-ear baseline defines an unoriented lateral axis; the
    nose position selects its forward normal. The screen is tied to the long ear
    baseline rather than the much shorter nose-to-eye direction. Eye landmarks,
    when present, are a diagnostic only. Shoulder landmarks never substitute
    for head geometry.
    """

    nose = _joint(joints, "nose")
    left_ear = _joint(joints, "leftear")
    right_ear = _joint(joints, "rightear")
    if any(point is None for point in (nose, left_ear, right_ear)):
        return None

    assert nose is not None
    assert left_ear is not None
    assert right_ear is not None
    ear_centre = _midpoint(left_ear.xy, right_ear.xy)
    ear_axis = Vec2(right_ear.x - left_ear.x, right_ear.y - left_ear.y)
    ear_baseline = _length(ear_axis)
    if not _plausible_ear_baseline(ear_baseline, config):
        return None
    ear_error = _mean_known_error((left_ear, right_ear))
    if ear_error is not None and ear_error / ear_baseline > config.max_relative_ear_error:
        return None
    lateral_axis = _unit(ear_axis)
    assert lateral_axis is not None
    ear_normal = Vec2(-lateral_axis.y, lateral_axis.x)
    nose_from_ear_centre = Vec2(nose.x - ear_centre.x, nose.y - ear_centre.y)
    nose_component = _dot(nose_from_ear_centre, ear_normal)
    sign_error = _mean_known_error((nose, left_ear, right_ear))
    sign_threshold = max(config.min_nose_sign_component_m, sign_error or 0.0)
    if abs(nose_component) < sign_threshold:
        return None
    direction = ear_normal if nose_component > 0 else Vec2(-ear_normal.x, -ear_normal.y)
    left_eye = _joint(joints, "lefteye")
    right_eye = _joint(joints, "righteye")
    eye_agreement = _optional_eye_agreement(direction, nose, left_eye, right_eye)
    return HeadHeading(
        direction=direction,
        source=HeadingSource.EARS_SIGNED_BY_NOSE,
        ear_baseline_m=ear_baseline,
        nose_sign_component_m=nose_component,
        ear_landmark_error_m=ear_error,
        sign_landmark_error_m=sign_error,
        eye_directional_agreement_deg=eye_agreement,
    )


def potential_view(
    observer_position: Vec2,
    target_position: Vec2,
    joints: Mapping[str, Joint3D],
    config: HeadGeometryConfig = DEFAULT_HEAD_GEOMETRY_CONFIG,
    *,
    observer_id: str | None = None,
    target_id: str | None = None,
    occluders: Sequence[Occluder] = (),
) -> PotentialView:
    """Classify head-centred potential visibility as unknown, in, or out.

    An occlusion flag makes the result ``UNKNOWN``: pitch-plane discs cannot
    establish that the target was unseen, and body pose does not measure gaze.
    """

    heading = estimate_head_heading(joints, config)
    distance = observer_position.distance_to(target_position)
    if heading is None or distance == 0 or distance > config.max_target_distance_m:
        return PotentialView(ViewStatus.UNKNOWN, None, None, distance, heading, False)
    occluded = _is_pitch_plane_occluded(
        observer_position, target_position, observer_id, target_id, occluders
    )
    if occluded:
        return PotentialView(ViewStatus.UNKNOWN, None, None, distance, heading, True)
    direction_to_target = _unit(
        Vec2(
            target_position.x - observer_position.x,
            target_position.y - observer_position.y,
        )
    )
    assert direction_to_target is not None
    eccentricity = _angle_between_degrees(heading.direction, direction_to_target)
    if eccentricity <= config.central_half_angle_deg:
        return PotentialView(
            ViewStatus.IN_VIEW, ViewZone.CENTRAL, eccentricity, distance, heading, False
        )
    if eccentricity <= config.peripheral_half_angle_deg:
        return PotentialView(
            ViewStatus.IN_VIEW,
            ViewZone.PERIPHERAL,
            eccentricity,
            distance,
            heading,
            False,
        )
    return PotentialView(ViewStatus.OUT_OF_VIEW, None, eccentricity, distance, heading, False)


def time_to_intercept(
    player: PlayerState,
    target: Vec2,
    config: InterceptConfig = DEFAULT_INTERCEPT_CONFIG,
) -> float:
    """Return an arrival-time proxy from projected current velocity and bounds."""

    displacement = Vec2(target.x - player.position.x, target.y - player.position.y)
    distance = _length(displacement)
    if distance == 0:
        return 0.0
    travel_direction = _unit(displacement)
    assert travel_direction is not None
    initial_speed = max(0.0, _dot(player.velocity, travel_direction))
    initial_speed = min(initial_speed, config.max_speed_mps)
    accel_distance = (config.max_speed_mps**2 - initial_speed**2) / (
        2 * config.max_acceleration_mps2
    )
    if distance <= accel_distance:
        travel_time = (
            sqrt(initial_speed**2 + 2 * config.max_acceleration_mps2 * distance) - initial_speed
        ) / config.max_acceleration_mps2
    else:
        time_to_max_speed = (config.max_speed_mps - initial_speed) / config.max_acceleration_mps2
        travel_time = time_to_max_speed + (distance - accel_distance) / config.max_speed_mps
    return config.reaction_time_s + travel_time


def arrival_control_proxy(
    players: Sequence[PlayerState],
    target: Vec2,
    team_id: str,
    config: InterceptConfig = DEFAULT_INTERCEPT_CONFIG,
) -> ArrivalControlProxy:
    """Compare closest same-team and opponent arrival times with a logistic map."""

    own_arrivals = [
        time_to_intercept(player, target, config) for player in players if player.team_id == team_id
    ]
    opponent_arrivals = [
        time_to_intercept(player, target, config) for player in players if player.team_id != team_id
    ]
    if not own_arrivals:
        msg = f"no players supplied for team_id={team_id!r}"
        raise ValueError(msg)
    if not opponent_arrivals:
        msg = "at least one opponent is required for an arrival control proxy"
        raise ValueError(msg)
    closest_own = min(own_arrivals)
    closest_opponent = min(opponent_arrivals)
    delta = (closest_own - closest_opponent) / config.arrival_temperature_s
    proxy = _stable_inverse_logit(-delta)
    return ArrivalControlProxy(
        team_id=team_id,
        arrival_control_proxy=proxy,
        closest_team_arrival_s=closest_own,
        closest_opponent_arrival_s=closest_opponent,
        threshold_met=proxy >= config.control_proxy_threshold,
    )


def regional_team_control(
    players: Sequence[PlayerState],
    centre: Vec2,
    team_id: str,
    region: RegionalControlConfig = DEFAULT_REGIONAL_CONTROL_CONFIG,
    intercept: InterceptConfig = DEFAULT_INTERCEPT_CONFIG,
) -> RegionalControlSummary:
    """Summarise pitch-clipped samples in a configurable circle around a player."""

    samples = [
        point
        for point in _circle_samples(centre, region.radius_m, region.sample_spacing_m)
        if region.pitch_bounds.contains(point)
    ]
    if not samples:
        msg = "the requested region has no samples inside pitch_bounds"
        raise ValueError(msg)
    proxies = [arrival_control_proxy(players, point, team_id, intercept) for point in samples]
    values = [proxy.arrival_control_proxy for proxy in proxies]
    return RegionalControlSummary(
        team_id=team_id,
        radius_m=region.radius_m,
        sample_count=len(values),
        mean_arrival_control_proxy=sum(values) / len(values),
        min_arrival_control_proxy=min(values),
        threshold_fraction=sum(proxy.threshold_met for proxy in proxies) / len(proxies),
    )


def _joint(joints: Mapping[str, Joint3D], canonical_name: str) -> Joint3D | None:
    aliases = {
        "lefteye": {"lefteye", "leye"},
        "righteye": {"righteye", "reye"},
        "leftear": {"leftear", "lear"},
        "rightear": {"rightear", "rear"},
    }
    accepted_names = aliases.get(canonical_name, {canonical_name})
    for name, joint in joints.items():
        normalized_name = "".join(character for character in name.lower() if character.isalnum())
        if normalized_name in accepted_names:
            return joint
    return None


def _midpoint(left: Vec2, right: Vec2) -> Vec2:
    return Vec2((left.x + right.x) / 2, (left.y + right.y) / 2)


def _length(vector: Vec2) -> float:
    return hypot(vector.x, vector.y)


def _unit(vector: Vec2) -> Vec2 | None:
    magnitude = _length(vector)
    if magnitude == 0:
        return None
    return Vec2(vector.x / magnitude, vector.y / magnitude)


def _dot(left: Vec2, right: Vec2) -> float:
    return left.x * right.x + left.y * right.y


def _angle_between_degrees(left: Vec2, right: Vec2) -> float:
    left_unit = _unit(left)
    right_unit = _unit(right)
    if left_unit is None or right_unit is None:
        return 180.0
    cosine = max(-1.0, min(1.0, _dot(left_unit, right_unit)))
    return acos(cosine) * 180.0 / 3.141592653589793


def _plausible_ear_baseline(length: float, config: HeadGeometryConfig) -> bool:
    return config.min_ear_baseline_m <= length <= config.max_ear_baseline_m


def _mean_known_error(points: Sequence[Joint3D]) -> float | None:
    errors = [point.error_m for point in points if point.error_m is not None]
    return sum(errors) / len(errors) if errors else None


def _optional_eye_agreement(
    heading: Vec2,
    nose: Joint3D,
    left_eye: Joint3D | None,
    right_eye: Joint3D | None,
) -> float | None:
    if left_eye is None or right_eye is None:
        return None
    eye_centre = _midpoint(left_eye.xy, right_eye.xy)
    eye_direction = Vec2(nose.x - eye_centre.x, nose.y - eye_centre.y)
    if _unit(eye_direction) is None:
        return None
    return _angle_between_degrees(heading, eye_direction)


def _is_pitch_plane_occluded(
    observer: Vec2,
    target: Vec2,
    observer_id: str | None,
    target_id: str | None,
    occluders: Sequence[Occluder],
) -> bool:
    segment = Vec2(target.x - observer.x, target.y - observer.y)
    segment_squared = _dot(segment, segment)
    if segment_squared == 0:
        return False
    for occluder in occluders:
        if occluder.player_id in {observer_id, target_id}:
            continue
        offset = Vec2(
            occluder.position.x - observer.x,
            occluder.position.y - observer.y,
        )
        projection = _dot(offset, segment) / segment_squared
        if not 0 < projection < 1:
            continue
        closest = Vec2(
            observer.x + projection * segment.x,
            observer.y + projection * segment.y,
        )
        if closest.distance_to(occluder.position) <= occluder.radius_m:
            return True
    return False


def _stable_inverse_logit(value: float) -> float:
    if value >= 0:
        return 1 / (1 + exp(-value))
    exp_value = exp(value)
    return exp_value / (1 + exp_value)


def _circle_samples(centre: Vec2, radius_m: float, spacing_m: float) -> list[Vec2]:
    steps = ceil(radius_m / spacing_m)
    samples: list[Vec2] = []
    for x_step in range(-steps, steps + 1):
        for y_step in range(-steps, steps + 1):
            offset = Vec2(x_step * spacing_m, y_step * spacing_m)
            if _length(offset) <= radius_m:
                samples.append(Vec2(centre.x + offset.x, centre.y + offset.y))
    return samples
