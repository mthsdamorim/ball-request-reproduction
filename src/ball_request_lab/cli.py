"""Command-line entry point: one command per stage of the reproduction.

Every stage reads the downloaded public data (``--data-dir``) and the frozen
inputs under ``inputs/`` and ``config/``, and writes to its own directory under
``--out``. Re-running a stage replaces that stage's directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

from ball_request_lab import __version__
from ball_request_lab.analysis import analyze_match
from ball_request_lab.case import build_examined_case
from ball_request_lab.context import describe
from ball_request_lab.data import MatchData, prepare
from ball_request_lab.download import download
from ball_request_lab.figures import build_figures
from ball_request_lab.gestures import GestureConfig, detect
from ball_request_lab.incidental import run_incidental
from ball_request_lab.ml_experiment import run_experiment
from ball_request_lab.real_evaluation import evaluate_real_windows
from ball_request_lab.real_windows import build_real_windows
from ball_request_lab.util import atomic_json
from ball_request_lab.verify import verify

MATCHES = [1925299, 1996435]
EXAMINED_MATCH = 1996435
EXAMINED_EVENT = "1996435-795547-5148"

Stage = Callable[[argparse.Namespace], None]


def _fresh(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def stage_download(args: argparse.Namespace) -> None:
    report = download(args.sources, args.data_dir, args.matches)
    print(json.dumps(report, indent=2))


def stage_prepare(args: argparse.Namespace) -> None:
    for match_id in args.matches:
        audit = prepare(args.data_dir, match_id, args.data_dir / "prepared" / f"{match_id}.npz")
        print(json.dumps(audit, indent=2))


def stage_detect(args: argparse.Namespace) -> None:
    """Run the frozen geometric detector, then the region/view/response analysis."""
    run = _fresh(args.out / "detector")
    config = tomllib.loads(args.detector_config.read_text())
    shutil.copyfile(args.detector_config, run / "config.toml")
    for match_id in args.matches:
        data = MatchData.load(args.data_dir / "prepared" / f"{match_id}.npz")
        events, audit = detect(data, match_id, GestureConfig(**config["gesture"]))
        atomic_json(run / f"{match_id}-candidates.json", events)
        atomic_json(run / f"{match_id}-detection-audit.json", audit)
        episodes = analyze_match(args.data_dir, run, match_id, config)
        atomic_json(run / f"{match_id}-analyzed.json", episodes)
        print(f"{match_id}: {len(events)} candidates, {len(episodes)} analysed")


def stage_case(args: argparse.Namespace) -> None:
    """Analyse the visually defined de Jesus candidate and build its case."""
    run = _fresh(args.out / "case-005")
    config = tomllib.loads(args.detector_config.read_text())
    shutil.copyfile(args.inputs / "candidate-005.json", run / f"{EXAMINED_MATCH}-candidates.json")
    episodes = analyze_match(args.data_dir, run, EXAMINED_MATCH, config)
    analyzed = run / f"{EXAMINED_MATCH}-analyzed.json"
    atomic_json(analyzed, episodes)
    build_examined_case(
        data_dir=args.data_dir,
        analyzed=analyzed,
        ledger=args.inputs / "adjudication-005.json",
        case_config=args.config_dir / "case-005.toml",
        frozen_config=args.detector_config,
        output=run / "case",
    )
    describe(run / "case" / "case.json", args.data_dir, EXAMINED_MATCH, 9, run / "context.json")
    print(json.dumps({"event_id": EXAMINED_EVENT, "output": str(run)}))


def stage_ml(args: argparse.Namespace) -> None:
    """Evaluate the synthetic experiment with the frozen models, or retrain them."""
    name = "ml-retrained" if args.retrain else "ml"
    run = args.out / name
    if run.exists():
        shutil.rmtree(run)
    frozen = None if args.retrain else args.inputs / "frozen-models"
    result = run_experiment(args.config_dir / "ml-experiment.toml", run, frozen)
    print(json.dumps({"run_dir": str(run), "chosen": result["chosen_checkpoint"]}))


def stage_real(args: argparse.Namespace) -> None:
    """Select the 16 windows, attach the frozen labels, apply the frozen models."""
    real = args.out / "real"
    if real.exists():
        shutil.rmtree(real)
    selection = build_real_windows(
        args.data_dir,
        real,
        args.inputs / "review-002-anchors.json",
        per_half=4,
        seed=20261004,
    )
    shutil.copyfile(args.inputs / "real-adjudications.json", real / "annotations.json")
    result = evaluate_real_windows(args.out / "ml", real, real / "transfer.json")
    print(json.dumps({"selected": len(selection["cases"]), **result["summary"]}))
    incidental = args.out / "incidental"
    if incidental.exists():
        shutil.rmtree(incidental)
    row = run_incidental(args.data_dir, args.out / "ml", incidental)["cases"][0]
    print(json.dumps({k: row[k] for k in ("geometry", "ridge", "tcn")}))


def stage_figures(args: argparse.Namespace) -> None:
    manifest = build_figures(
        args.out / "case-005" / "case" / "case.json",
        args.out / "case-005" / "context.json",
        args.out / "real" / "transfer.json",
        args.out / "figures",
    )
    atomic_json(args.out / "figures" / "manifest.json", manifest)
    print(json.dumps(list(manifest["figures"])))


def stage_verify(args: argparse.Namespace) -> None:
    report = verify(args)
    sys.exit(0 if report["failed"] == 0 else 1)


STAGES: dict[str, tuple[Stage, str]] = {
    "download": (stage_download, "download and checksum the public inputs (~1.4 GB)"),
    "prepare": (stage_prepare, "align tracking, events and pose on the 5 Hz grid"),
    "detect": (stage_detect, "frozen geometric detector and analysis (16 candidates)"),
    "case": (stage_case, "the de Jesus case, its measurements and game context"),
    "ml": (stage_ml, "synthetic recognition experiment with the frozen models (needs PyTorch)"),
    "real": (stage_real, "16 real windows and the de Jesus window, frozen models"),
    "figures": (stage_figures, "Figures 2 to 6"),
    "verify": (stage_verify, "check every checkable statement of the article"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the article's results")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--inputs", type=Path, default=Path("inputs"))
    parser.add_argument("--config-dir", type=Path, default=Path("config"))
    parser.add_argument("--expected", type=Path, default=Path("expected"))
    parser.add_argument(
        "--matches",
        default=",".join(str(m) for m in MATCHES),
        help="comma-separated match ids (default: both)",
    )
    parser.add_argument(
        "--ml-tolerance",
        type=float,
        default=1e-5,
        help="largest numeric difference tolerated when comparing PyTorch-dependent "
        "outputs with the article run (default 1e-5, float32 rounding)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in STAGES.items():
        stage = sub.add_parser(name, help=help_text)
        if name == "ml":
            stage.add_argument(
                "--retrain",
                action="store_true",
                help="train the ridge and the networks from scratch into outputs/ml-retrained "
                "(not bit-reproducible across CPU architectures)",
            )
    every = sub.add_parser("all", help="run every stage in order, then verify")
    every.add_argument("--skip-ml", action="store_true", help="skip stages that need PyTorch")
    args = parser.parse_args()
    args.matches = [int(m) for m in args.matches.split(",")]
    args.sources = args.config_dir / "sources.json"
    args.detector_config = args.config_dir / "detector.toml"
    if args.command == "all":
        order = ["download", "prepare", "detect", "case"]
        order += [] if args.skip_ml else ["ml", "real"]
        order += ["figures", "verify"]
        args.retrain = False
        for name in order:
            print(f"\n=== {name} ===", flush=True)
            STAGES[name][0](args)
        return
    STAGES[args.command][0](args)
