"""Check every statement of the article that can be checked from regenerated outputs.

Each check pairs one value stated in the article with the value recomputed by this
repository, at the precision the article uses. Statements that depend on the
broadcast video or on a visual judgement cannot be recomputed from
public data; they are reported as ``frozen input`` and verified only for
consistency with the input file that records them.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import platform
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ball_request_lab.case import requester_head_bearings
from ball_request_lab.data import JOINTS, MatchData
from ball_request_lab.download import is_verified
from ball_request_lab.gestures import GestureConfig, arm_states
from ball_request_lab.util import atomic_json, file_hash

Observed = Any


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        group: str,
        statement: str,
        expected: Observed,
        observed: Observed,
        ok: bool | None = None,
        decimals: int | None = None,
        status: str | None = None,
    ) -> None:
        if ok is None:
            if decimals is not None:
                ok = bool(abs(float(observed) - float(expected)) < 0.5 * 10**-decimals + 1e-9)
            else:
                ok = bool(observed == expected)
        self.rows.append(
            {
                "group": group,
                "statement": statement,
                "article": expected,
                "recomputed": observed,
                "status": status or ("PASS" if ok else "FAIL"),
            }
        )

    def skip(self, group: str, why: str) -> None:
        self.rows.append(
            {
                "group": group,
                "statement": why,
                "article": None,
                "recomputed": None,
                "status": "SKIPPED",
            }
        )


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _round(value: float, digits: int) -> float:
    return round(float(value), digits)


def check_inputs(report: Report, args: argparse.Namespace) -> None:
    manifest = _load(args.sources)
    for entry in manifest["files"]:
        if int(Path(entry["path"]).name.split("_")[0].split(".")[0]) not in args.matches:
            continue
        path = args.data_dir / entry["path"]
        report.add(
            "inputs",
            f"{entry['path']} matches the pinned size and SHA-256",
            True,
            is_verified(path, entry),
        )
    report.add("inputs", "body pose has 29 joints per player", 29, len(JOINTS))


def check_detector(report: Report, args: argparse.Namespace) -> None:
    run = args.out / "detector"
    if not run.exists():
        report.skip("detector", "detector outputs missing")
        return
    audits = [_load(run / f"{m}-detection-audit.json") for m in args.matches]
    report.add(
        "detector",
        "the frozen detector returned 16 candidate fragments",
        16,
        sum(a["candidate_episodes"] for a in audits),
    )
    config = audits[0]["config"]
    report.add("detector", "wrist above shoulder by 0.08 m", 0.08, config["wrist_above_shoulder_m"])
    report.add("detector", "wrist above elbow by 0.02 m", 0.02, config["wrist_above_elbow_m"])
    report.add("detector", "joint error at most 20 cm", 20.0, config["max_joint_error_cm"])
    report.add("detector", "three shared samples, 0.4 s at 5 Hz", 0.4, config["min_duration_s"])
    ledger = _load(args.inputs / "review-002-adjudications.json")
    requests = [r for r in ledger if r["label"] == "probable_request"]
    sustained = [r for r in requests if r["evaluable_identity"] is True]
    report.add(
        "review ledger (frozen input)",
        "four candidate fragments were read as probable requests",
        4,
        len(requests),
    )
    report.add(
        "review ledger (frozen input)",
        "only one had identity evidence: H. Van Der Saag, number 17",
        ["H. Van Der Saag #17"],
        [f"{r['player_name']} #{r['shirt_number']}" for r in sustained],
    )
    event_id = sustained[0]["event_id"]
    analyzed = run / "1925299-analyzed.json"
    if not (args.data_dir / "prepared" / "1925299.npz").exists():
        report.skip("goalkeeper angle", "prepared data for match 1925299 missing")
        return
    angles = requester_head_bearings(args.data_dir, analyzed, event_id, args.detector_config)
    at = angles["confirmation_sample"]
    report.add(
        "goalkeeper angle",
        "Van Der Saag's head direction is within about 4 degrees of the ball carrier at confirmation (5 Hz grid)",
        3,
        _round(at["to_holder_deg"], 0),
        ok=abs(at["to_holder_deg"] - 3) < 1.5,
    )
    report.add(
        "goalkeeper angle",
        "his goalkeeper stands about 25 degrees from the same direction (5 Hz grid)",
        25,
        _round(at["to_goalkeeper_deg"], 0),
    )
    report.add(
        "goalkeeper angle",
        "every sample of the episode keeps the keeper off the head direction by 20 to 35 degrees",
        True,
        all(20 <= s["to_goalkeeper_deg"] <= 35 for s in angles["samples"]),
    )


def check_case(report: Report, args: argparse.Namespace) -> None:
    run = args.out / "case-005"
    if not (run / "case" / "case.json").exists():
        report.skip("case", "case outputs missing")
        return
    case = _load(run / "case" / "case.json")
    summary = _load(run / "case" / "summary.json")
    analyzed = _load(run / "1996435-analyzed.json")[0]
    info = case["case"]
    g = "case"
    report.add(
        g,
        "event: de Jesus (795547), carrier Grant (51644), match 1996435",
        [795547, 51644, 1996435],
        [info["caller_id"], info["holder_id"], analyzed["match_id"]],
    )
    samples = {s["relative_s"]: s for s in case["samples"]}
    report.add(
        g,
        "onset at 513.8 s, confirmation at 514.2 s of source clock",
        [513.8, 514.2],
        [samples[0.0]["source_clock_s"], samples[0.4]["source_clock_s"]],
    )
    context = _load(run / "context.json")
    first_pass = context["possession_chain"][0]
    report.add(
        g,
        "Grant passes to K. Shaw (#42) at +1.2 s",
        ["K. Shaw", 1.2],
        [first_pass["pass_target"], info["action_relative_s"]],
    )
    report.add(
        g,
        "the context second is rendered without possession or sector",
        ["unknown"],
        sorted({s["view"]["status"] for s in case["samples"] if s["relative_s"] < 0}),
    )
    report.add(
        g,
        "nearest opponent to de Jesus at +0.4 s is 11.29 m",
        11.29,
        analyzed["nearest_opponent_m"],
        decimals=2,
    )
    control = samples[0.4]["control"]
    report.add(
        g,
        "local-control index at +0.4 s is 0.965 (4 m disk)",
        0.965,
        control["4"]["mean"],
        decimals=3,
    )
    report.add(
        g,
        "local-control index at +0.4 s is 0.930 (5 m disk)",
        0.930,
        control["5"]["mean"],
        decimals=3,
    )
    intercept = tomllib.loads(args.detector_config.read_text())["intercept"]
    report.add(
        g,
        "assumptions: 0.2 s reaction, 3 m/s^2, 5 m/s ceiling",
        [0.2, 3.0, 5.0],
        [intercept[k] for k in ("reaction_time_s", "max_acceleration_mps2", "max_speed_mps")],
    )
    ecc = {
        s["relative_s"]: s["view"]["eccentricity_deg"]
        for s in case["samples"]
        if s["view"].get("eccentricity_deg") is not None
    }
    report.add(
        g,
        "de Jesus is about 15 degrees off Grant's head direction at onset",
        15,
        _round(ecc[0.0], 0),
    )
    report.add(g, "and about 28 degrees at confirmation", 28, _round(ecc[0.4], 0))
    counts = {k: (v["in_view"], v["total"]) for k, v in summary["view_by_angle"].items()}
    report.add(
        g,
        "inside the sector in 3/6, 5/6, 5/6 samples at 60, 90, 110 degrees",
        {"60": (3, 6), "90": (5, 6), "110": (5, 6)},
        counts,
    )
    near = sorted(round(e, 1) for e in ecc.values() if any(abs(e - a) < 3 for a in (60, 90, 110)))
    report.add(
        g,
        "two samples sit within three degrees of a threshold: 87.6 and 112.8",
        [87.6, 112.8],
        near,
    )
    passes = samples[0.4]["passes"]
    report.add(g, "route length at +0.4 s is 24.36 m", 24.36, passes["15"]["length_m"], decimals=2)
    for speed, value in (("10", -0.488), ("15", -0.073), ("20", 0.134)):
        report.add(
            g,
            f"minimum margin at {speed} m/s is {value:+.3f} s",
            value,
            passes[speed]["minimum_margin_s"],
            decimals=3,
        )
    report.add(
        g,
        "J. Yull (#12) is the critical opponent at 15 m/s, 12.43 m from the origin",
        [51646, 12.43],
        [int(passes["15"]["critical_player_id"]), _round(passes["15"]["critical_distance_m"], 2)],
    )
    margins15 = [s["passes"]["15"]["minimum_margin_s"] for s in case["samples"] if s["passes"]]
    margins20 = {
        s["relative_s"]: s["passes"]["20"]["minimum_margin_s"]
        for s in case["samples"]
        if s["passes"]
    }
    report.add(
        g,
        "15 m/s margin is +0.012 s at onset and falls at every sample",
        [0.012, True],
        [_round(margins15[0], 3), all(a > b for a, b in itertools.pairwise(margins15))],
    )
    report.add(
        g,
        "20 m/s margin stays positive until +0.6 s and is negative afterwards",
        True,
        all(v > 0 for k, v in margins20.items() if k <= 0.6)
        and all(v < 0 for k, v in margins20.items() if k > 0.6),
    )
    quality = samples[0.4]["quality"]
    report.add(
        g,
        "two of 22 players have extrapolated positions at confirmation",
        [2, 22],
        [len(quality["extrapolated_ids"]), quality["active_players"]],
    )
    report.add(
        g, "Yull is a detected player", True, context["caller"]["nearest_opponent"]["detected"]
    )
    people = {p["name"]: p for p in context["lineup_at_reference"]}
    report.add(
        g,
        "shirt numbers: de Jesus 21, Grant 23, Shaw 42, Yull 12",
        [21, 23, 42, 12],
        [people[n]["number"] for n in ("Z. de Jesus", "R. Grant", "K. Shaw", "J. Yull")],
    )
    ball = case["samples"][5]["ball"]

    def toward_ball(player_name: str) -> bool:
        xs = []
        for s in case["samples"]:
            if s["relative_s"] in (0.0, 0.4):
                q = next(p for p in s["players"] if p["name"] == player_name)
                xs.append(math.hypot(q["x"] - ball["x"], q["y"] - ball["y"]))
        return xs[1] < xs[0]

    report.add(
        g,
        "Yull is moving towards the ball between onset and confirmation",
        True,
        toward_ball("J. Yull"),
    )
    names = [o["name"] for o in context["possession_chain"][0]["provider_passing_options"]]
    report.add(
        g,
        "the provider's options for Grant name Cáceres, Shaw and Léo Sena, not de Jesus",
        ["A. Cáceres", "K. Shaw", "Léo Sena"],
        sorted(names, key=str.casefold),
    )
    sync = _load(args.config_dir / "video-sync.json")
    first = next(p for p in sync["periods"] if p["match_id"] == 1996435 and p["period"] == 1)
    report.add(
        "frozen input",
        "ball-event checks had residuals between -0.117 and +0.283 s (recorded in video-sync.json, not recomputable without the broadcast)",
        [-0.117, 0.283],
        [_round(v, 3) for v in first["observed_residual_envelope_s"]],
        status="PASS"
        if [round(v, 3) for v in first["observed_residual_envelope_s"]] == [-0.117, 0.283]
        else "FAIL",
    )


def check_gate(report: Report, args: argparse.Namespace) -> None:
    """The geometric rule is blind to de Jesus because of the 20 cm quality gate."""
    path = args.data_dir / "prepared" / "1996435.npz"
    if not path.exists():
        report.skip("quality gate", "prepared data for match 1996435 missing")
        return
    data = MatchData.load(path)
    p = int(np.flatnonzero(data.players == 795547)[0])
    window = range(2574, 2584)
    above = [
        float(
            data.joints[t, p, JOINTS.index("rWrist"), 2]
            - data.joints[t, p, JOINTS.index("rShoulder"), 2]
        )
        for t in window
    ]
    error = [float(data.errors[t, p, JOINTS.index("rWrist")]) for t in window]
    report.add(
        "quality gate",
        "raw right wrist is 0.2 to 0.4 m above the shoulder",
        [0.2, 0.4],
        [round(min(above), 1), round(max(above), 1)],
    )
    report.add(
        "quality gate",
        "predicted error of that wrist is 26 to 34 cm",
        [26, 34],
        [round(min(error)), round(max(error))],
    )
    holder = [t for t in range(2560, 2575) if data.owner[t] == 51644]
    report.add(
        "quality gate",
        "Grant is recorded in possession from 0.6 s before the onset",
        0.6,
        round(float(data.clock[2574] - data.clock[holder[0]]), 1),
        decimals=1,
    )
    state, _ = arm_states(data, GestureConfig())
    report.add(
        "quality gate",
        "the frozen rule returns no raised-arm state for him in those samples",
        True,
        bool(np.all(state[2574:2584, p] == 0)),
    )


def back_four_x(opponents: dict[str, dict[str, Any]]) -> list[float]:
    return [p["x"] for p in opponents.values() if p["acronym"] in ("LB", "RB", "LCB", "RCB")]


def check_context(report: Report, args: argparse.Namespace) -> None:
    run = args.out / "case-005"
    if not (run / "context.json").exists():
        report.skip("game context", "context output missing")
        return
    c = _load(run / "context.json")
    g = "game context"
    phase = c["phase_of_play"]
    clock = c["reference_sample"]["source_clock_s"]
    start = sum(float(x) * m for x, m in zip(phase["time_start"].split(":"), (60, 1), strict=False))
    report.add(
        g,
        "provider phase: Sydney 'create' against Adelaide 'medium block', attacking left to right",
        ["create", "medium_block", "left_to_right"],
        [
            phase["team_in_possession_phase_type"],
            phase["team_out_of_possession_phase_type"],
            phase["attacking_side"],
        ],
    )
    report.add(
        g,
        "the phase began thirteen seconds earlier, in Sydney's defensive third",
        [13, "defensive_third"],
        [round(clock - start), phase["third_start"]],
    )
    report.add(g, "ninth minute", 9, int(clock // 60) + 1)
    phases = args.data_dir / "matches" / "1996435" / "1996435_phases_of_play.csv"
    with phases.open(newline="") as handle:
        goals_before = sum(
            1
            for r in csv.DictReader(handle)
            if r["team_possession_lead_to_goal"] == "True" and int(r["frame_start"]) < 5152
        )
    report.add(g, "score 0-0 (no earlier phase led to a goal)", 0, goals_before)
    holder, caller, target = c["holder"], c["caller"], c["observed_target"]
    lineup = {p["name"]: p for p in c["lineup_at_reference"]}
    report.add(
        g,
        "roles: Grant RCB, Shaw LCB, de Jesus RB, Courtney-Perkins LB, Hollman RDM, Léo Sena LDM",
        ["RCB", "LCB", "RB", "LB", "RDM", "LDM"],
        [
            lineup[n]["acronym"]
            for n in (
                "R. Grant",
                "K. Shaw",
                "Z. de Jesus",
                "J. Courtney-Perkins",
                "C. Hollman",
                "Léo Sena",
            )
        ],
    )
    report.add(
        g,
        "Grant about eight metres inside his own half, right of centre",
        [-8, True],
        [round(holder["x"]), holder["y"] < 0],
    )
    report.add(
        g,
        "Shaw stands 18 metres from Grant, goalkeeper behind them",
        [18, True],
        [
            round(target["distance_from_holder_m"]),
            lineup["H. Devenish-Meares"]["x"] < min(holder["x"], target["x"]),
        ],
    )
    report.add(
        g,
        "Hollman and Léo Sena at the halfway line (within 3 m)",
        True,
        all(abs(lineup[n]["x"]) < 3 for n in ("C. Hollman", "Léo Sena")),
    )
    report.add(
        g,
        "de Jesus about twelve metres (11.6 m) inside the Adelaide half, a step (< 2 m) from the touchline",
        [12, True],
        [round(caller["x"]), 34 - abs(caller["y"]) < 2],
    )
    report.add(
        g,
        "Courtney-Perkins about ten metres inside the Adelaide half",
        10,
        round(lineup["J. Courtney-Perkins"]["x"]),
    )
    back_four = [
        p["x"]
        for p in c["lineup_at_reference"]
        if p["team_id"] != lineup["R. Grant"]["team_id"]
        and p["acronym"] in ("LB", "RB", "LCB", "RCB")
    ]
    report.add(
        g,
        "Cáceres is dropping wide on the right (y < -20 m, attacking left to right)",
        True,
        lineup["A. Cáceres"]["y"] < -20,
    )
    opponents = {
        p["name"]: p
        for p in c["lineup_at_reference"]
        if p["team_id"] != lineup["R. Grant"]["team_id"]
    }
    report.add(
        g,
        "Ouahim stands between Adelaide's midfield (Barnett) and its back line",
        True,
        opponents["J. Barnett"]["x"] < lineup["A. Ouahim"]["x"] < min(back_four_x(opponents)),
    )
    ball_xy = (c["reference_sample"]["ball"]["x"], c["reference_sample"]["ball"]["y"])
    report.add(
        g,
        "a pass to de Jesus goes forward (>15 m) and sideways (>10 m): diagonal",
        True,
        caller["x"] - ball_xy[0] > 15 and abs(caller["y"] - ball_xy[1]) > 10,
    )
    report.add(
        g,
        "Lolley and Klimala are level with the Adelaide back four",
        True,
        all(
            min(back_four) - 1 <= lineup[n]["x"] <= max(back_four) + 1
            for n in ("J. Lolley", "P. Klimala")
        ),
    )
    first_line = {"A. Goodwin", "Z. Clough", "E. Alagich", "S. Mauk", "J. Yull"}
    between = {p["name"]: p for p in c["opponents_between_ball_and_caller_x"]}
    report.add(
        g,
        "five Adelaide players (Goodwin, Clough, Alagich, Mauk, Yull) stand between the ball and de Jesus",
        True,
        first_line <= set(between),
    )
    report.add(
        g,
        "all five within about four metres of the halfway line (largest distance 4.06 m)",
        4.06,
        _round(max(abs(between[n]["x"]) for n in first_line), 2),
        decimals=2,
    )
    report.add(
        g,
        "Yull, the widest of them, is 11 metres from de Jesus",
        [11, True],
        [
            round(caller["nearest_opponent"]["distance_m"]),
            caller["nearest_opponent"]["name"] == "J. Yull",
        ],
    )
    report.add(
        g,
        "a pass to de Jesus is 24 metres, to Shaw 18 metres with Goodwin 11 metres from him",
        [24, 18, "A. Goodwin", 11],
        [
            round(caller["distance_from_ball_m"]),
            round(target["distance_from_holder_m"]),
            target["nearest_opponent"]["name"],
            round(target["nearest_opponent"]["distance_m"]),
        ],
    )
    options = {o["name"]: o for o in c["possession_chain"][0]["provider_passing_options"]}
    report.add(
        g,
        "provider completion estimates 0.998, 0.995, 0.985; threat at most 0.002",
        [0.998, 0.995, 0.985, True],
        [
            _round(options["K. Shaw"]["xpass_completion"], 3),
            _round(options["A. Cáceres"]["xpass_completion"], 3),
            _round(options["Léo Sena"]["xpass_completion"], 3),
            max(o["xthreat"] for o in options.values()) <= 0.002,
        ],
    )
    keeper = [o for o in c["possession_chain"][2]["provider_passing_options"]]
    jesus = next(o for o in keeper if o["name"] == "Z. de Jesus")
    report.add(
        g,
        "four seconds later de Jesus is listed at 0.70, the lowest of three options",
        [0.70, True, 3],
        [
            _round(jesus["xpass_completion"], 2),
            jesus["xpass_completion"] == min(o["xpass_completion"] for o in keeper),
            len(keeper),
        ],
    )
    chain = c["possession_chain"]
    report.add(
        g,
        "chain: Grant, Shaw, goalkeeper, Courtney-Perkins, Hollman, Grant, de Jesus",
        [
            "R. Grant",
            "K. Shaw",
            "H. Devenish-Meares",
            "J. Courtney-Perkins",
            "C. Hollman",
            "R. Grant",
            "Z. de Jesus",
        ],
        [x["player"] for x in chain[:7]],
    )
    last = chain[5]
    report.add(
        g,
        "Grant's long forward pass to de Jesus is successful",
        ["successful", "Z. de Jesus"],
        [last["pass_outcome"], last["pass_target"]],
    )
    report.add(
        g,
        "it is released 18.6 s after the onset",
        18.6,
        (last["frame_end"] - 5148) / 10,
        decimals=1,
    )
    report.add(
        g, "at most 19 seconds ('almost nineteen')", True, (last["frame_end"] - 5148) / 10 < 19
    )
    with phases.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    def phase_at(frame: int) -> dict[str, str]:
        return next(r for r in rows if int(r["frame_start"]) <= frame <= int(r["frame_end"]))

    arrives, leaves = phase_at(chain[6]["frame_start"]), phase_at(chain[6]["frame_end"])
    report.add(
        g,
        "while de Jesus has the ball the phase changes from 'create' to 'finish' and Adelaide's "
        "from 'medium block' to 'low block'; the attacking third is reached, no shot",
        [
            "Z. de Jesus",
            "create",
            "medium_block",
            "finish",
            "low_block",
            "attacking_third",
            "False",
        ],
        [
            chain[6]["player"],
            arrives["team_in_possession_phase_type"],
            arrives["team_out_of_possession_phase_type"],
            leaves["team_in_possession_phase_type"],
            leaves["team_out_of_possession_phase_type"],
            leaves["third_end"],
            leaves["team_possession_lead_to_shot"],
        ],
    )


def check_ml(report: Report, args: argparse.Namespace) -> None:
    path = args.out / "ml" / "metrics.json"
    if not path.exists():
        report.skip("synthetic model", "ML outputs missing (stage skipped or not run)")
        return
    m = _load(path)
    g = "synthetic model"
    chosen = next(t for t in m["tcn"] if f"seed-{t['seed']}" in m["chosen_checkpoint"])
    binary = [t["binary"]["overall"]["balanced_accuracy"] for t in m["tcn"]]
    report.add(
        g,
        "five classes, 29 joints, 11 frames (two seconds at 5 Hz)",
        [5, "11 frames; 2-second trailing window at 5 Hz"],
        [len(m["protocol"]["classes"]), m["protocol"]["window"]],
    )
    report.add(
        g,
        "rule balanced accuracy 0.731",
        0.731,
        m["raised_arm_baseline_binary"]["overall"]["balanced_accuracy"],
        decimals=3,
    )
    report.add(
        g,
        "ridge balanced accuracy 0.584",
        0.584,
        m["ridge"]["binary"]["overall"]["balanced_accuracy"],
        decimals=3,
    )
    report.add(
        g,
        "validation-selected TCN balanced accuracy 0.797",
        0.797,
        chosen["binary"]["overall"]["balanced_accuracy"],
        decimals=3,
    )
    report.add(
        g,
        "TCN range across three seeds 0.714 to 0.839",
        [0.714, 0.839],
        [_round(min(binary), 3), _round(max(binary), 3)],
    )
    report.add(
        g,
        "selected TCN five-class macro F1 0.492",
        0.492,
        chosen["overall"]["macro_f1"],
        decimals=3,
    )
    report.add(
        g,
        "67 of 90 two-arm examples called one-arm",
        [67, 90],
        [
            chosen["overall"]["confusion"][2][1],
            chosen["overall"]["per_class"]["two_arm_raise"]["support"],
        ],
    )
    report.add(
        g,
        "shuffled-label control 0.244 balanced accuracy",
        0.244,
        m["shuffled_label_control"]["overall"]["balanced_accuracy"],
        decimals=3,
    )
    families = m["protocol"]["temporal_families_by_split"]
    report.add(
        g,
        "late and pulse movement families are held out for testing",
        ["late", "pulse"],
        families["test"],
    )
    counts = m["protocol"]["base_group_counts"]
    report.add(
        g,
        "bases split before augmentation: 400 train, 100 validation, 150 test",
        [400, 100, 150],
        [counts["train"], counts["validation"], counts["test"]],
    )


def check_real(report: Report, args: argparse.Namespace) -> None:
    path = args.out / "real" / "transfer.json"
    if not path.exists():
        report.skip("real windows", "real-window outputs missing")
        return
    t = _load(path)
    g = "real windows"
    s = t["summary"]
    report.add(
        g,
        "16 windows selected before annotation, 176 frames (16 x 11 samples)",
        [16, 176],
        [s["selected"], s["selected"] * 11],
    )
    evaluable = [r for r in t["cases"] if r["evaluable_identity"]]
    report.add(
        g,
        "two evaluable windows, both no signal: J. Hingert and J. Courtney-Perkins",
        [2, ["no_signal", "no_signal"]],
        [s["identity_and_label_evaluable"], [r["gesture_type"] for r in evaluable]],
    )
    sel = _load(args.out / "real" / "selection.json")
    names = {c["case_id"]: c["player_name"] for c in sel["cases"]}
    report.add(
        g,
        "their names",
        ["J. Courtney-Perkins", "J. Hingert"],
        sorted(names[r["case_id"]] for r in evaluable),
    )
    report.add(
        g,
        "14 windows remain exclusions; no verified raised-arm label, so no recall",
        [14, False],
        [s["excluded"], s["can_estimate_raised_recall"]],
    )
    report.add(
        g,
        "correct on 2, 2 and 1 of the 2 cases (geometry, ridge, TCN)",
        [2, 2, 1],
        [s["models"][k]["correct"] for k in ("geometry", "ridge", "tcn")],
    )
    inc = _load(args.out / "incidental" / "transfer.json")["cases"][0]
    report.add(
        g,
        "on the de Jesus window: rule 'other', ridge 'one_arm_raise', TCN 'directional_point'",
        ["other", "one_arm_raise", "directional_point"],
        [inc["geometry"], inc["ridge"], inc["tcn"]],
    )


def check_retrain(report: Report, args: argparse.Namespace) -> None:
    """Optional: did training from scratch land on the article's numbers on this machine?"""
    path = args.out / "ml-retrained" / "metrics.json"
    if not path.exists():
        return
    out: list[tuple[str, float | str]] = []
    _walk(
        _load(path), _load(args.expected / "ml/metrics.json"), "", args.ml_tolerance, IGNORED, out
    )
    report.add(
        "retraining",
        "training from scratch reproduces the article run on this machine "
        f"(tolerance {args.ml_tolerance:g}); otherwise the frozen models carry the result",
        0,
        len(out),
        status="PASS" if not out else "WARN",
    )
    if out:
        report.rows[-1]["detail"] = {"differences": len(out), "first": out[:5]}


def check_figures(report: Report, args: argparse.Namespace) -> None:
    """Each figure must equal the published one, byte for byte or pixel for pixel.

    Another platform may encode the PNG differently or anti-alias a few pixels by one
    or two intensity levels; that is still the same figure. Anything larger fails.
    """
    folder = args.out / "figures"
    for reference in sorted((args.expected / "figures").glob("*.png")):
        path = folder / reference.name
        statement = f"{reference.name} equals the figure used in the article"
        if not path.exists():
            report.skip("figures", f"{reference.name} was not generated")
            continue
        if file_hash(path) == file_hash(reference):
            report.add("figures", statement + " (byte-identical)", True, True)
            continue
        with Image.open(path) as made, Image.open(reference) as published:
            if made.size != published.size:
                report.add("figures", statement, str(published.size), str(made.size), ok=False)
                continue
            gap = np.abs(
                np.asarray(made.convert("RGB"), dtype=np.int16)
                - np.asarray(published.convert("RGB"), dtype=np.int16)
            )
        report.add(
            "figures",
            statement + f" (pixel-identical within {PIXEL_TOLERANCE} intensity levels)",
            True,
            int(gap.max()) <= PIXEL_TOLERANCE,
            ok=int(gap.max()) <= PIXEL_TOLERANCE,
        )
        report.rows[-1]["detail"] = {
            "largest_pixel_difference": int(gap.max()),
            "pixels_differing": int((gap.max(axis=2) > 0).sum()),
        }


def _walk(
    a: Any,
    b: Any,
    path: str,
    tolerance: float,
    ignore: set[str],
    out: list[tuple[str, float | str]],
) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in ignore:
                continue
            if key not in a or key not in b:
                out.append((f"{path}/{key}", "missing"))
            else:
                _walk(a[key], b[key], f"{path}/{key}", tolerance, ignore, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} != {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b, strict=False)):
            _walk(x, y, f"{path}[{i}]", tolerance, ignore, out)
    elif isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        diff = abs(float(a) - float(b))
        if diff > tolerance:
            out.append((path, diff))
    elif a != b:
        out.append((path, "different"))


SCORE_DRIFT_LIMIT = 0.05
PIXEL_TOLERANCE = 2
IGNORED = {
    "python",
    "platform",
    "torch",
    "code_sha256",
    "config_sha256",
    "selector_sha256",
    "evaluator_sha256",
    "model_files_sha256",
    "selection_sha256",
    "annotations_sha256",
    "windows_sha256",
    "prepared_sha256",
    "previous_selection_sha256",
    "output",
    "evidence_path",
    "history",
    "best_validation_loss",
    "best_epoch",
}


def check_reference(report: Report, args: argparse.Namespace) -> None:
    """Compare regenerated outputs with the outputs of the run behind the article."""
    pairs = [
        ("detector/1925299-candidates.json", "detector/1925299-candidates.json"),
        ("detector/1996435-candidates.json", "detector/1996435-candidates.json"),
        ("detector/1925299-analyzed.json", "detector/1925299-analyzed.json"),
        ("detector/1996435-analyzed.json", "detector/1996435-analyzed.json"),
        ("case-005/1996435-analyzed.json", "case-005/1996435-analyzed.json"),
        ("case-005/case/summary.json", "case-005/summary.json"),
        ("case-005/context.json", "case-005/context.json"),
        ("ml/metrics.json", "ml/metrics.json"),
        ("real/transfer.json", "real/transfer.json"),
        ("incidental/transfer.json", "incidental/transfer.json"),
    ]
    for generated, reference in pairs:
        path = args.out / generated
        if not path.exists():
            continue
        out: list[tuple[str, float | str]] = []
        tolerance = (
            1e-9 if not generated.startswith(("ml/", "real/", "incidental/")) else args.ml_tolerance
        )
        _walk(_load(path), _load(args.expected / reference), "", tolerance, IGNORED, out)
        scores = [d for d in out if "softmax_scores" in d[0]]
        others = [d for d in out if "softmax_scores" not in d[0]]
        drift = max((v for _, v in scores if isinstance(v, float)), default=0.0)
        # Uncalibrated model scores may drift across platforms (different BLAS); every
        # discrete result and every other number must still match.
        score_only = bool(scores) and not others and drift <= SCORE_DRIFT_LIMIT
        status = "PASS" if not out else "WARN" if score_only else "FAIL"
        report.add(
            "reference outputs",
            f"{generated} equals the output of the article run (tolerance {tolerance:g})",
            0,
            len(others) if score_only else len(out),
            status=status,
        )
        if out:
            report.rows[-1]["detail"] = {
                "differences": len(out),
                "model_score_differences": len(scores),
                "largest_model_score_difference": drift,
                "other_differences": others[:5],
            }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    args.ml_tolerance = getattr(args, "ml_tolerance", 1e-6)
    report = Report()
    steps: list[Callable[[Report, argparse.Namespace], None]] = [
        check_inputs,
        check_detector,
        check_case,
        check_gate,
        check_context,
        check_ml,
        check_real,
        check_retrain,
        check_figures,
        check_reference,
    ]
    for step in steps:
        step(report, args)
    failed = [r for r in report.rows if r["status"] == "FAIL"]
    warned = [r for r in report.rows if r["status"] == "WARN"]
    skipped = [r for r in report.rows if r["status"] == "SKIPPED"]
    result = {
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "passed": sum(r["status"] == "PASS" for r in report.rows),
        "failed": len(failed),
        "warned": len(warned),
        "skipped": len(skipped),
        "checks": report.rows,
    }
    atomic_json(args.out / "verification.json", result)
    for row in report.rows:
        marker = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn ", "SKIPPED": " skip "}[
            row["status"]
        ]
        print(f"[{marker}] {row['group']}: {row['statement']}")
        if row["status"] == "FAIL":
            print(
                f"          article: {row['article']!r}\n          recomputed: {row['recomputed']!r}"
            )
    print(
        f"\n{result['passed']} passed, {result['failed']} failed, "
        f"{result['warned']} warnings, {result['skipped']} skipped"
    )
    sys.stdout.flush()
    return result
