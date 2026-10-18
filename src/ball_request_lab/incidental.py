"""Apply the frozen models to the one separately discovered de Jesus window.

There is no selector: the window is fixed (Sydney FC number 21, Z. de Jesus,
match 1996435, period 1, source clock 513.2 to 515.2 s). It is evaluated outside
the reserved 16-window transfer audit and tunes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ball_request_lab.data import HZ, MatchData, metadata
from ball_request_lab.real_evaluation import evaluate_real_windows
from ball_request_lab.real_windows import STEPS
from ball_request_lab.util import atomic_json, file_hash

CASE_ID = "incidental-1996435-1-01"
MATCH_ID = 1996435
PERIOD = 1
PLAYER_ID = 795547
TEAM_ID = 869
SHIRT_NUMBER = 21
WINDOW_START_S = 513.2
WINDOW_END_S = 515.2


def run_incidental(data_dir: Path, ml_run: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    prepared = data_dir / "prepared" / f"{MATCH_ID}.npz"
    data = MatchData.load(prepared)
    player_indices = np.flatnonzero(data.players == PLAYER_ID)
    if len(player_indices) != 1:
        raise ValueError("Expected exactly one prepared-data player index")
    ends = np.flatnonzero((data.period == PERIOD) & np.isclose(data.clock, WINDOW_END_S))
    if len(ends) != 1:
        raise ValueError("Expected exactly one source-clock terminal sample")
    end = int(ends[0])
    start = end - STEPS + 1
    if start < 0 or not np.isclose(data.clock[start], WINDOW_START_S):
        raise ValueError("Prepared samples no longer reproduce the 2 s window")
    player_index = int(player_indices[0])
    if int(data.teams[player_index]) != TEAM_ID:
        raise ValueError("Prepared-data team does not match the roster record")
    person = next(p for p in metadata(data_dir, MATCH_ID)["players"] if int(p["id"]) == PLAYER_ID)
    if int(person["number"]) != SHIRT_NUMBER:
        raise ValueError("Roster shirt number does not match")

    output_dir.mkdir(parents=True)
    windows = output_dir / "windows.npz"
    np.savez_compressed(
        windows,
        raw_windows_m=data.joints[start : end + 1, player_index][None, ...].copy(),
        errors_cm=data.errors[start : end + 1, player_index][None, ...].copy(),
        case_ids=np.array([CASE_ID]),
    )
    case = {
        "case_id": CASE_ID,
        "match_id": MATCH_ID,
        "period": PERIOD,
        "player_id": PLAYER_ID,
        "player_name": person["short_name"],
        "shirt_number": SHIRT_NUMBER,
        "team_id": TEAM_ID,
        "start_index": start,
        "analysis_index": end,
        "source_clock_s": float(data.clock[end]),
        "source_start_s": float(data.clock[start]),
        "cohort": "incidental_exploratory_not_random",
        "use": "exploratory qualitative model application; not training or test score",
    }
    atomic_json(
        output_dir / "selection.json",
        {
            "schema_version": 1,
            "cases": [case],
            "steps": STEPS,
            "sample_hz": HZ,
            "windows_sha256": file_hash(windows),
            "method": "Fixed post-review window; no sampling, detector, outcome or "
            "model-based selection.",
        },
    )
    atomic_json(
        output_dir / "annotations.json",
        [
            {
                "case_id": CASE_ID,
                "gesture_type": "one_arm_raise",
                "evaluable_identity": True,
                "identity_confidence": "high",
                "communicative_function": "uncertain",
                "reviewer": "Visual review; not human expert adjudication",
                "label_window": "only source clock 513.2 through 515.2 s",
            }
        ],
    )
    return evaluate_real_windows(ml_run, output_dir, output_dir / "transfer.json")
