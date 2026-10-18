# ruff: noqa: E501
"""Figures 2 to 6 of the article, drawn only from the reproduced outputs.

Nothing here refits a model, changes a label, or turns a conditional geometric
measure into a probability, a gaze estimate or a pass recommendation.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle, Wedge

from ball_request_lab.util import file_hash

PAPER = "#f6f0e5"
INK = "#183840"
TEAL = "#167b7d"
CORAL = "#ce654e"
AMBER = "#b88224"
STEEL = "#687b83"
GRID = "#c9d1ca"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "savefig.dpi": 200,
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def write_figure(
    figure: plt.Figure,
    output: Path,
    stem: str,
    title: str,
    description: str,
) -> list[Path]:
    paths: list[Path] = []
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        figure.savefig(
            path,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor=PAPER,
            metadata={"Title": title, "Description": description},
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        paths.append(path)
    plt.close(figure)
    return paths


def add_footer(figure: plt.Figure, text: str) -> None:
    figure.text(0.075, 0.025, text, color=STEEL, fontsize=8.8, va="bottom")


def wrist_heights(samples: list[dict[str, Any]], side: str) -> np.ndarray:
    return np.array([sample["wrist_height_m"][side] for sample in samples])


def draw_case_clock(case_data: dict[str, Any], output: Path) -> list[Path]:
    case = case_data["case"]
    samples = [sample for sample in case_data["samples"] if sample["holder_has_ball"]]
    time = np.array([sample["relative_s"] for sample in samples])
    left_wrist = wrist_heights(samples, "l")
    right_wrist = wrist_heights(samples, "r")
    angle = np.array([sample["view"]["eccentricity_deg"] for sample in samples])
    control_4 = np.array([sample["control"]["4"]["mean"] for sample in samples])
    control_5 = np.array([sample["control"]["5"]["mean"] for sample in samples])

    figure, axes = plt.subplots(3, 1, figsize=(10.8, 8.2), sharex=True)
    figure.subplots_adjust(left=0.13, right=0.96, top=0.84, bottom=0.18, hspace=0.42)
    figure.suptitle(
        "A raised arm, head direction and local space",
        x=0.075,
        ha="left",
        y=0.97,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.905,
        f"{case['caller_name']} #{case['caller_number']} requests; "
        f"{case['holder_name']} #{case['holder_number']} holds the ball. "
        "Six pre-action samples, 0.0–1.0 s after detected eligible onset.",
        color=STEEL,
        fontsize=10,
    )

    axes[0].plot(time, left_wrist, "o-", color=AMBER, linewidth=2.3, label="Left wrist")
    axes[0].plot(time, right_wrist, "o-", color=CORAL, linewidth=2.3, label="Right wrist")
    axes[0].axhline(0, color=INK, linewidth=0.9)
    axes[0].set_ylabel("Wrist above\nshoulder (m)")
    axes[0].set_title("Body-pose geometry", loc="left", fontsize=11)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")

    axes[1].plot(time, angle, "o-", color=TEAL, linewidth=2.5)
    for limit, color, label in (
        (60, STEEL, "±60°"),
        (90, AMBER, "±90°"),
        (110, CORAL, "±110°"),
    ):
        axes[1].axhline(limit, color=color, linewidth=1, linestyle="--", alpha=0.85)
        axes[1].text(1.015, limit, label, color=color, fontsize=8.8, va="bottom")
    axes[1].set_ylim(min(45.0, float(np.floor(angle.min() / 10) * 10 - 5)), 125)
    axes[1].set_ylabel("Head-to-caller\nangle (°)")
    axes[1].set_title("Head-direction scenarios", loc="left", fontsize=11)
    axes[1].text(
        0.01,
        0.92,
        "Scenario thresholds describe view angles; they do not measure gaze or attention.",
        transform=axes[1].transAxes,
        color=STEEL,
        fontsize=8.8,
    )

    axes[2].plot(time, control_4, "o-", color=TEAL, linewidth=2.3, label="4 m radius")
    axes[2].plot(time, control_5, "o-", color=AMBER, linewidth=2.3, label="5 m radius")
    axes[2].set_ylim(0, 1)
    axes[2].set_yticks((0, 0.5, 1))
    axes[2].set_ylabel("Local-arrival\nindex")
    axes[2].set_xlabel("Seconds since detected eligible onset")
    axes[2].set_title("Relative arrival advantage around the caller", loc="left", fontsize=11)
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    axes[2].text(
        0.01,
        0.12,
        "Index only — not a calibrated probability of receiving or completing a pass.",
        transform=axes[2].transAxes,
        color=STEEL,
        fontsize=8.8,
    )
    for axis in axes:
        clean_axis(axis)
        axis.set_xlim(-0.03, 1.03)
        axis.axvline(0.4, color=INK, linewidth=1, linestyle=":")
    raised = left_wrist if left_wrist[2] >= right_wrist[2] else right_wrist
    axes[0].annotate(
        "confirmation instant",
        xy=(0.4, raised[2]),
        xytext=(0.54, raised[2] + 0.18),
        arrowprops={"arrowstyle": "-", "color": STEEL, "lw": 0.8},
        color=STEEL,
        fontsize=8.8,
    )
    add_footer(
        figure,
        "Detected eligible onset is not the biomechanical start of the gesture. "
        f"Observed pass to another teammate at +{case['action_relative_s']:.1f} s is not an input.",
    )
    return write_figure(
        figure,
        output,
        "01-integrated-case-clock",
        "A raised arm, head direction and local space",
        "Six pre-action samples show wrist height, head-to-caller angle with hypothetical view-angle thresholds, and local arrival indices.",
    )


def draw_destination_route(case_data: dict[str, Any], output: Path) -> list[Path]:
    case = case_data["case"]
    snapshot = next(
        sample for sample in case_data["samples"] if sample["frame"] == case["analysis_frame"]
    )
    players = {str(player["id"]): player for player in snapshot["players"]}
    caller = players[str(case["caller_id"])]
    holder = players[str(case["holder_id"])]
    route = snapshot["passes"]["15"]
    route_points = route["points"]
    critical = min(route_points, key=lambda point: point["margin_s"])
    heading = snapshot["view"]["heading"]

    figure, (pitch, arrival) = plt.subplots(
        1, 2, figsize=(12.3, 6.8), gridspec_kw={"width_ratios": [1.42, 1]}
    )
    figure.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.25, wspace=0.27)
    figure.suptitle(
        "An open destination can still have a contested route",
        x=0.075,
        ha="left",
        y=0.97,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.905,
        "Snapshot at +0.4 s. Straight, immediate ground-pass scenario to the caller’s current position at 15 m/s.",
        color=STEEL,
        fontsize=10,
    )
    length = case["pitch_length"]
    width = case["pitch_width"]
    pitch.add_patch(
        Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            fill=False,
            edgecolor=INK,
            linewidth=1.1,
        )
    )
    pitch.axvline(0, color=GRID, linewidth=0.8)
    pitch.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor=GRID, linewidth=0.8))
    for side in (-1, 1):
        start = -length / 2 if side == -1 else length / 2 - 16.5
        pitch.add_patch(
            Rectangle((start, -20.16), 16.5, 40.32, fill=False, edgecolor=GRID, linewidth=0.8)
        )
    if heading is not None:
        direction = heading["direction"]
        heading_degrees = np.degrees(np.arctan2(direction["y"], direction["x"]))
        pitch.add_patch(
            Wedge(
                (holder["x"], holder["y"]),
                45,
                heading_degrees - 90,
                heading_degrees + 90,
                facecolor=TEAL,
                alpha=0.08,
                edgecolor="none",
            )
        )
    grid_points = snapshot["control"]["4"]["points"]
    local_index = pitch.scatter(
        [point["x"] for point in grid_points],
        [point["y"] for point in grid_points],
        c=[point["value"] for point in grid_points],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        marker="s",
        s=29,
        alpha=0.65,
        linewidths=0,
        zorder=1,
    )
    for player in players.values():
        same_team = str(player["team_id"]) == str(case["team_id"])
        color = TEAL if same_team else CORAL
        pitch.scatter(
            player["x"],
            player["y"],
            s=52,
            color=color,
            alpha=0.9 if player["detected"] else 0.36,
            zorder=4,
        )
    pitch.plot(
        [point["x"] for point in route_points],
        [point["y"] for point in route_points],
        color=AMBER,
        linewidth=2.6,
        linestyle="--",
        zorder=3,
    )
    pitch.scatter(holder["x"], holder["y"], color=TEAL, s=115, zorder=5)
    pitch.scatter(caller["x"], caller["y"], color=AMBER, marker="*", s=185, zorder=6)
    pitch.scatter(
        critical["x"],
        critical["y"],
        color=INK,
        marker="x",
        s=68,
        linewidths=1.8,
        zorder=6,
    )
    pitch.annotate(
        f"#{case['holder_number']}  holder",
        (holder["x"], holder["y"]),
        xytext=(8, 11),
        textcoords="offset points",
        fontsize=8.8,
    )
    pitch.annotate(
        f"#{case['caller_number']}  caller",
        (caller["x"], caller["y"]),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8.8,
    )
    pitch.annotate(
        "minimum margin",
        (critical["x"], critical["y"]),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
    )
    pitch.set(
        xlim=(-length / 2 - 1, length / 2 + 1),
        ylim=(-width / 2 - 1, width / 2 + 1),
        aspect="equal",
    )
    pitch.set_axis_off()
    pitch.set_title("Destination neighbourhood and hypothetical path", loc="left", fontsize=11)
    colorbar = figure.colorbar(local_index, ax=pitch, fraction=0.035, pad=0.02)
    colorbar.set_label("4 m local-arrival index\n(not probability)", fontsize=8.4)
    colorbar.ax.tick_params(labelsize=7.8)
    pitch.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor=TEAL,
                markeredgecolor=TEAL,
                markersize=6,
                label="Team-mate",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor=CORAL,
                markeredgecolor=CORAL,
                markersize=6,
                label="Opponent",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor=INK,
                markeredgecolor=INK,
                alpha=0.36,
                markersize=6,
                label="Faded = provider extrapolated",
            ),
            Wedge(
                (0, 0),
                0.2,
                0,
                180,
                facecolor=TEAL,
                alpha=0.15,
                label="±90° head-sector proxy",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        frameon=False,
        fontsize=7.5,
        ncol=2,
    )

    distance = [point["distance_m"] for point in route_points]
    ball_time = [point["ball_time_s"] for point in route_points]
    opponent_time = [point["opponent_arrival_s"] for point in route_points]
    arrival.plot(distance, ball_time, color=AMBER, linewidth=2.7, label="Hypothetical ball")
    arrival.plot(distance, opponent_time, color=CORAL, linewidth=2.7, label="Earliest opponent")
    arrival.axvline(critical["distance_m"], color=INK, linewidth=1, linestyle=":")
    arrival.annotate(
        f"minimum margin\n{critical['margin_s']:+.2f} s",
        xy=(critical["distance_m"], critical["ball_time_s"]),
        xytext=(8, -28),
        textcoords="offset points",
        fontsize=8.8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": STEEL, "lw": 0.8},
    )
    arrival.set_xlabel("Distance along route (m)")
    arrival.set_ylabel("Arrival time (s)")
    arrival.set_title("Arrival times along the same route", loc="left", fontsize=11)
    arrival.legend(frameon=False, loc="upper left")
    clean_axis(arrival)
    add_footer(
        figure,
        "Squares show the 4 m local-arrival index (not a probability). "
        "The route ignores ball deceleration, offside, receiver movement, body orientation, and first touch.",
    )
    return write_figure(
        figure,
        output,
        "02-destination-route-snapshot",
        "An open destination can still have a contested route",
        "Pitch snapshot at the confirmation instant, with a local-arrival index around the caller and a 15 metre-per-second hypothetical ground-pass route whose minimum margin is negative.",
    )


def draw_scenarios(case_data: dict[str, Any], output: Path) -> list[Path]:
    samples = [sample for sample in case_data["samples"] if sample["holder_has_ball"]]
    time = np.array([sample["relative_s"] for sample in samples])
    figure, (margins, angles) = plt.subplots(
        1, 2, figsize=(12, 5.9), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    figure.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.21, wspace=0.32)
    figure.suptitle(
        "The conclusion changes with declared scenarios",
        x=0.075,
        ha="left",
        y=0.96,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.895,
        "Sensitivity is a property of the stated mechanical and directional scenarios, not evidence of perception or pass probability.",
        color=STEEL,
        fontsize=10,
    )
    scenarios = (("10", CORAL), ("15", AMBER), ("20", TEAL))
    for speed, color in scenarios:
        values = [sample["passes"][speed]["minimum_margin_s"] for sample in samples]
        margins.plot(time, values, "o-", linewidth=2.6, color=color, label=f"{speed} m/s")
    margins.axhline(0, color=INK, linewidth=1)
    margins.fill_between((-0.03, 1.03), -1.3, 0, color=CORAL, alpha=0.07)
    margins.text(
        0.02,
        0.04,
        "Opponent arrives first in this model",
        transform=margins.transAxes,
        color=CORAL,
        fontsize=8.8,
    )
    margins.set(
        xlim=(-0.03, 1.03),
        ylim=(-1.3, 0.55),
        xlabel="Seconds since detected eligible onset",
        ylabel="Minimum route margin (s)",
    )
    margins.set_title("Ground-pass speed scenarios", loc="left", fontsize=11)
    margins.legend(frameon=False, title="Constant speed")
    clean_axis(margins)

    thresholds = np.array([60, 90, 110])
    observed = np.array(
        [
            sum(sample["view"]["eccentricity_deg"] <= threshold for sample in samples)
            for threshold in thresholds
        ]
    )
    bars = angles.bar(np.arange(len(thresholds)), observed, color=[STEEL, AMBER, CORAL], width=0.58)
    angles.set(
        xticks=np.arange(len(thresholds)),
        xticklabels=["±60°", "±90°", "±110°"],
        ylim=(0, 6.6),
        ylabel="Samples within scenario (of 6)",
    )
    angles.set_title("Head-direction angle scenarios", loc="left", fontsize=11)
    angles.set_xlabel("Hypothetical sector half-angle")
    for bar, value in zip(bars, observed, strict=True):
        angles.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.16,
            f"{value}/6",
            ha="center",
            color=INK,
            fontweight="bold",
        )
    angles.text(
        0.03,
        0.93,
        "Direction from ears + nose to caller XY.\nNot eye tracking, gaze, or attention.",
        transform=angles.transAxes,
        color=STEEL,
        fontsize=8.8,
    )
    clean_axis(angles)
    add_footer(
        figure,
        "Minimum margin = earliest opponent arrival minus hypothetical ball arrival. "
        "Positive values are not pass safety, control, or a recommendation.",
    )
    return write_figure(
        figure,
        output,
        "03-scenario-sensitivity",
        "The conclusion changes with declared scenarios",
        "Six temporal samples show route margins for 10, 15, and 20 metre-per-second ground-pass scenarios, alongside counts within three head-direction angle scenarios.",
    )


def draw_evidence_selection(real_data: dict[str, Any], output: Path) -> list[Path]:
    summary = real_data["summary"]
    selected = summary["selected"]
    evaluable = summary["identity_and_label_evaluable"]
    raised = (
        summary["observed_class_support"]["one_arm_raise"]
        + summary["observed_class_support"]["two_arm_raise"]
    )
    figure, (funnel, audit) = plt.subplots(
        1, 2, figsize=(12, 5.9), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    figure.subplots_adjust(left=0.08, right=0.96, top=0.82, bottom=0.21, wspace=0.31)
    figure.suptitle(
        "Only two of 16 real windows supported player-linked evaluation",
        x=0.075,
        ha="left",
        y=0.96,
        fontsize=16.5,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.895,
        "Reserved, visually reviewed player-windows; selection occurred without detector outputs or later passes.",
        color=STEEL,
        fontsize=10,
    )
    stages = [
        ("Selected windows", selected, STEEL),
        ("Identity + label evaluable", evaluable, TEAL),
        ("Verified raised-arm labels", raised, AMBER),
    ]
    ypos = np.arange(len(stages))[::-1]
    widths = [value / selected for _, value, _ in stages]
    for y, (label, value, color), width in zip(ypos, stages, widths, strict=True):
        if value:
            funnel.barh(y, width, color=color, height=0.56)
            if width > 0.5:
                funnel.text(
                    0.015,
                    y,
                    label,
                    va="center",
                    color=PAPER,
                    fontsize=10,
                    fontweight="bold",
                )
                funnel.text(
                    width + 0.025,
                    y,
                    f"{value}",
                    va="center",
                    color=INK,
                    fontsize=13,
                    fontweight="bold",
                )
            else:
                funnel.text(
                    width / 2,
                    y,
                    f"{value}",
                    ha="center",
                    va="center",
                    color=PAPER,
                    fontsize=12,
                    fontweight="bold",
                )
                funnel.text(
                    width + 0.035,
                    y,
                    label,
                    va="center",
                    color=INK,
                    fontsize=10,
                    fontweight="bold",
                )
        else:
            funnel.text(
                0.015,
                y,
                "0",
                va="center",
                color=AMBER,
                fontsize=13,
                fontweight="bold",
            )
            funnel.text(
                0.065,
                y,
                label,
                va="center",
                color=INK,
                fontsize=10,
                fontweight="bold",
            )
    funnel.set(
        xlim=(0, 1.16),
        ylim=(-0.4, 2.4),
        yticks=[],
        xlabel="Share of the 16 selected windows",
    )
    funnel.set_title("Evidence funnel", loc="left", fontsize=11)
    funnel.spines["left"].set_visible(False)
    funnel.spines["top"].set_visible(False)
    funnel.spines["right"].set_visible(False)
    funnel.grid(False)

    names = ["Rule", "Ridge", "TCN"]
    correct = [summary["models"][key]["correct"] for key in ("geometry", "ridge", "tcn")]
    colors = [STEEL, TEAL, CORAL]
    bars = audit.bar(names, correct, color=colors, width=0.58)
    audit.set(ylim=(0, 2.45), ylabel="Correct among 2 evaluable windows")
    audit.set_title("What the evaluable subset supports", loc="left", fontsize=11)
    for bar, value in zip(bars, correct, strict=True):
        audit.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value}/2",
            ha="center",
            fontweight="bold",
        )
    audit.text(
        0.02,
        0.93,
        "Both evaluable labels were no signal.\nNo raised-arm recall or five-class F1 is estimable.",
        transform=audit.transAxes,
        color=STEEL,
        fontsize=8.8,
    )
    clean_axis(audit)
    add_footer(
        figure,
        "These are descriptive counts for a small sample reviewed visually without independent experts, not an event rate, population estimate, or external-match validation.",
    )
    return write_figure(
        figure,
        output,
        "04-evidence-selection",
        "Only two of 16 real windows supported player-linked evaluation",
        "Sixteen selected video windows reduce to two windows with evaluable identity and labels; both are no-signal examples, leaving no verified raised-arm labels for recall estimation.",
    )


def draw_pitch(axis: plt.Axes, length: float, width: float) -> None:
    axis.add_patch(
        Rectangle(
            (-length / 2, -width / 2),
            length,
            width,
            fill=False,
            edgecolor=INK,
            linewidth=1.1,
        )
    )
    axis.axvline(0, color=GRID, linewidth=0.8)
    axis.add_patch(Circle((0, 0), 9.15, fill=False, edgecolor=GRID, linewidth=0.8))
    for side in (-1, 1):
        start = -length / 2 if side == -1 else length / 2 - 16.5
        axis.add_patch(
            Rectangle((start, -20.16), 16.5, 40.32, fill=False, edgecolor=GRID, linewidth=0.8)
        )
    axis.set(
        xlim=(-length / 2 - 1, length / 2 + 1),
        ylim=(-width / 2 - 1, width / 2 + 1),
        aspect="equal",
    )
    axis.set_axis_off()


def short_name(name: str) -> str:
    return name.split(". ", 1)[-1]


def place_labels(
    axis: plt.Axes,
    players: list[dict[str, Any]],
    colors: dict[str, str],
    skip: set[str],
) -> None:
    """Label each dot with its role acronym, choosing the first free side."""
    occupied: list[tuple[float, float, float, float]] = [
        (p["x"] - 1.1, p["x"] + 1.1, p["y"] - 1.1, p["y"] + 1.1) for p in players
    ]
    candidates = (
        (1.4, 0.0, "left", "center"),
        (-1.4, 0.0, "right", "center"),
        (0.0, 1.5, "center", "bottom"),
        (0.0, -1.5, "center", "top"),
    )
    for player in sorted(players, key=lambda p: (p["y"], p["x"])):
        if player["name"] in skip:
            continue
        text = player["acronym"]
        half_w = 0.95 * len(text)
        for dx, dy, ha, va in candidates:
            cx = player["x"] + dx + (half_w if ha == "left" else -half_w if ha == "right" else 0)
            cy = player["y"] + dy + (0.9 if va == "bottom" else -0.9 if va == "top" else 0)
            box = (cx - half_w, cx + half_w, cy - 0.9, cy + 0.9)
            if not any(
                box[0] < o[1] and box[1] > o[0] and box[2] < o[3] and box[3] > o[2]
                for o in occupied
            ):
                break
        occupied.append(box)
        axis.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontsize=6.6,
            color=colors[player["name"]],
            zorder=5,
        )


def draw_game_context(context: dict[str, Any], output: Path) -> list[Path]:
    reference = context["reference_sample"]
    length = reference["pitch_length_m"]
    width = reference["pitch_width_m"]
    ball = reference["ball"]
    holder = context["holder"]
    caller = context["caller"]
    target = context["observed_target"]
    lineup = context["lineup_at_reference"]
    team_id = next(p["team_id"] for p in lineup if p["name"] == holder["name"])
    chain = context["possession_chain"]
    onset_frame = reference["frame"] - round(reference["relative_s"] * 10)
    passes: list[tuple[int, dict[str, Any], dict[str, Any], float]] = []
    for index, (possession, following) in enumerate(pairwise(chain)):
        if possession["pass_target"] != following["player"]:
            break
        passes.append(
            (
                index + 1,
                possession,
                following,
                (possession["frame_end"] - onset_frame) / 10,
            )
        )
        if following["player"] == caller["name"]:
            break
    final_seconds = passes[-1][3]
    yull = caller["nearest_opponent"]
    yull_point = next(p for p in lineup if p["name"] == yull["name"])
    chosen = next(o for o in chain[0]["provider_passing_options"] if o["name"] == target["name"])

    figure, (now, later) = plt.subplots(1, 2, figsize=(12.3, 6.1))
    figure.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.17, wspace=0.08)
    figure.suptitle(
        f"The safe pass, then the same pass {final_seconds:.0f} seconds later",
        x=0.075,
        ha="left",
        y=0.97,
        fontsize=18,
        fontweight="bold",
    )
    figure.text(
        0.075,
        0.905,
        f"Sydney FC attack left to right in the provider’s “{context['phase_of_play']['team_in_possession_phase_type']}” phase "
        f"against an Adelaide “{context['phase_of_play']['team_out_of_possession_phase_type'].replace('_', ' ')}”. "
        "Roles are the provider’s lineup roles; positions are source positions at +0.4 s.",
        color=STEEL,
        fontsize=10,
    )

    draw_pitch(now, length, width)
    now.add_patch(
        Rectangle(
            (-4.8, -width / 2),
            9.6,
            width,
            facecolor=CORAL,
            alpha=0.1,
            edgecolor="none",
            zorder=1,
        )
    )
    now.text(
        0,
        width / 2 - 1.6,
        "Adelaide’s first line: five players",
        ha="center",
        va="top",
        fontsize=8,
        color=CORAL,
    )
    colors = {p["name"]: (TEAL if p["team_id"] == team_id else CORAL) for p in lineup}
    for player in lineup:
        now.scatter(
            player["x"],
            player["y"],
            s=52,
            color=colors[player["name"]],
            alpha=0.9 if player["detected"] else 0.36,
            zorder=4,
        )
    named = {holder["name"], caller["name"], target["name"], yull["name"]}
    place_labels(now, lineup, colors, named)
    now.scatter(ball["x"], ball["y"], s=28, color=INK, zorder=6)
    for destination, color, distance in (
        (caller, AMBER, caller["distance_from_ball_m"]),
        (target, STEEL, target["distance_from_holder_m"]),
    ):
        now.annotate(
            "",
            xy=(destination["x"], destination["y"]),
            xytext=(ball["x"], ball["y"]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.2,
                "linestyle": "--",
                "shrinkA": 4,
                "shrinkB": 8,
            },
            zorder=3,
        )
        fraction = 0.5 if color == STEEL else 0.7
        mid = (
            ball["x"] + fraction * (destination["x"] - ball["x"]),
            ball["y"] + fraction * (destination["y"] - ball["y"]),
        )
        now.annotate(
            f"{distance:.0f} m",
            mid,
            xytext=(-7, 4) if color == STEEL else (7, -2),
            textcoords="offset points",
            ha="right" if color == STEEL else "left",
            fontsize=7.6,
            color=color,
            fontweight="bold",
            zorder=7,
        )
    now.scatter(holder["x"], holder["y"], color=TEAL, s=115, zorder=5)
    now.scatter(caller["x"], caller["y"], color=AMBER, marker="*", s=185, zorder=6)
    now.annotate(
        f"{holder['name']} #{holder['number']}, {holder['role'].lower()}, on the ball",
        (holder["x"], holder["y"]),
        xytext=(-9, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.2,
    )
    now.annotate(
        f"{target['name']} #{target['number']}, {target['role'].lower()}",
        (target["x"], target["y"]),
        xytext=(-9, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=8.2,
    )
    now.annotate(
        f"{caller['name']} #{caller['number']}, {caller['role'].lower()}, arm raised",
        (caller["x"], caller["y"]),
        xytext=(0, -11),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.2,
        color=AMBER,
    )
    now.annotate(
        f"{yull['name']} #{yull['number']}, {yull['distance_m']:.0f} m from {short_name(caller['name'])}",
        (yull_point["x"], yull_point["y"]),
        xytext=(9, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.2,
        color=CORAL,
    )
    now.set_title(
        "At confirmation, +0.4 s: two options of different kinds",
        loc="left",
        fontsize=11,
    )
    now.text(
        -length / 2,
        -width / 2 - 9.0,
        f"Amber: {caller['distance_from_ball_m']:.0f} m forward and diagonal past the line to {short_name(caller['name'])}, not in the provider’s option list.\n"
        f"Grey: {target['distance_from_holder_m']:.0f} m sideways to {short_name(target['name'])}, the pass played; provider completion estimate {chosen['xpass_completion']:.3f}.",
        fontsize=7.9,
        color=STEEL,
        va="top",
    )

    draw_pitch(later, length, width)
    for player in lineup:
        later.scatter(
            player["x"],
            player["y"],
            s=30,
            color=colors[player["name"]],
            alpha=0.18,
            zorder=2,
        )
    for number, possession, following, seconds in passes:
        start = (possession["x_start"], possession["y_start"])
        end = (following["x_start"], following["y_start"])
        is_last = following["player"] == caller["name"]
        color = AMBER if is_last else TEAL
        later.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "lw": 2.6 if is_last else 1.9,
                "shrinkA": 5,
                "shrinkB": 5,
            },
            zorder=4,
        )
        later.scatter(*start, s=60, color=color, zorder=5)
        later.annotate(
            f"{number}",
            start,
            xytext=(-9, -3),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=PAPER,
            bbox={
                "boxstyle": "circle,pad=0.25",
                "facecolor": color,
                "edgecolor": "none",
            },
            zorder=6,
        )
        if is_last:
            later.scatter(*end, color=AMBER, marker="*", s=185, zorder=6)
            later.annotate(
                f"{number}. {possession['player']} → {following['player']}\nforward, long, successful, +{seconds:.1f} s",
                end,
                xytext=(0, -11),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8.2,
                color=AMBER,
            )
    later.set_title(
        f"The next {final_seconds:.0f} seconds: six passes, the last one to the raised arm",
        loc="left",
        fontsize=11,
    )
    labels = [
        f"{number}. {short_name(possession['player'])} → {short_name(following['player'])} ({possession['pass_direction'].replace('_', ' ')}, +{seconds:.1f} s)"
        for number, possession, following, seconds in passes
    ]
    later.text(
        -length / 2,
        -width / 2 - 9.0,
        "   ".join(labels[:3]) + "\n" + "   ".join(labels[3:]),
        fontsize=7.9,
        color=STEEL,
        va="top",
    )
    add_footer(
        figure,
        "Dashed arrows are straight lines from the ball, not modelled passes. Completion is the provider’s own estimate, not computed here. "
        "Right: arrows join the provider’s possession start points; times are the release of each pass after the eligible onset.",
    )
    return write_figure(
        figure,
        output,
        "06-game-context",
        f"The safe pass, then the same pass {final_seconds:.0f} seconds later",
        "Two pitch maps: the positions, roles and two candidate passes at the confirmation instant, and the chain of six passes that ended with the ball carrier playing long to the requester about nineteen seconds after the onset.",
    )


FIGURES = {
    "01-integrated-case-clock": "Figure 2",
    "02-destination-route-snapshot": "Figure 3",
    "03-scenario-sensitivity": "Figure 4",
    "06-game-context": "Figure 5",
    "04-evidence-selection": "Figure 6",
}


def build_figures(
    case_path: Path, context_path: Path, real_transfer_path: Path | None, output: Path
) -> dict[str, Any]:
    """Draw every figure to ``output`` (PNG and SVG) and return a digest manifest."""
    setup_style()
    output.mkdir(parents=True, exist_ok=True)
    case_data = json.loads(case_path.read_text())
    context = json.loads(context_path.read_text())
    generated: dict[str, list[Path]] = {
        "01-integrated-case-clock": draw_case_clock(case_data, output),
        "02-destination-route-snapshot": draw_destination_route(case_data, output),
        "03-scenario-sensitivity": draw_scenarios(case_data, output),
        "06-game-context": draw_game_context(context, output),
    }
    inputs = {"case": file_hash(case_path), "context": file_hash(context_path)}
    if real_transfer_path is not None and real_transfer_path.exists():
        real_data = json.loads(real_transfer_path.read_text())
        generated["04-evidence-selection"] = draw_evidence_selection(real_data, output)
        inputs["real_transfer"] = file_hash(real_transfer_path)
    return {
        "inputs_sha256": inputs,
        "matplotlib": matplotlib.__version__,
        "figures": {
            stem: {
                "article_label": FIGURES[stem],
                "sha256": {path.name: file_hash(path) for path in paths},
            }
            for stem, paths in generated.items()
        },
    }
