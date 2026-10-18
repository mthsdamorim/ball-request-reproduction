# Ball Request Lab: reproduction package

Code, frozen inputs and checks that regenerate every number and figure of the article
*He raised his hand. Could the ball reach him?* from the public SkillCorner Open Data.

One command downloads the data, reruns the analysis and compares the result with
what the article states:

```sh
git clone https://github.com/mthsdamorim/ball-request-reproduction
cd ball-request-reproduction
scripts/reproduce.sh
```

It needs [uv](https://docs.astral.sh/uv/), network access, about 1.4 GB for the
download and about 4 GB of free disk. It takes a few minutes. The run ends with a
report of what was checked and exits non-zero if any statement of the article is
not reproduced.

## What the article claims, and where it is checked

| Article | Stage | What is recomputed |
| --- | --- | --- |
| Data and detector | `prepare`, `detect` | 29-joint pose on a shared 5 Hz grid; the frozen wrist-above-shoulder rule returns 16 candidate fragments in two matches |
| The arm the detector never saw | `case` | De Jesus's raw wrist is 0.2 to 0.4 m above the shoulder, but its predicted error is 26 to 34 cm, above the 20 cm gate, so the rule returns no state |
| Returning to one sequence, Figures 2 to 4 | `case`, `figures` | Nearest opponent, local-arrival indices, head-direction angles and sector counts, route margins at three ball speeds |
| Reading the game, Figure 5 | `case`, `figures` | Phase labels, roles, positions, distances, the provider's completion estimates and the six passes that follow |
| A small recognition model | `ml` | Synthetic five-class experiment: rule, ridge and a small temporal convolutional network, three seeds, shuffled-label control, evaluated with the frozen article weights |
| Sixteen real windows, Figure 6 | `real` | Re-selection of the 16 windows, frozen models applied, correct counts, the de Jesus window |
| The remark on Van Der Saag and his goalkeeper | `detect` | Angle between his head direction and the carrier and the goalkeeper |

`ball-request-lab verify` lists the checks one by one. It also compares each
regenerated result file with the file from the run behind the article
(`expected/`), and the five PNG figures with the published ones.

## Stages

Each stage can be run alone. Re-running one replaces its output directory.

```sh
uv run ball-request-lab download   # ~1.4 GB, resumable, SHA-256 verified
uv run ball-request-lab prepare    # align tracking, events and pose (about 1 minute)
uv run ball-request-lab detect     # frozen detector and analysis
uv run ball-request-lab case       # the de Jesus case and its game context
uv run ball-request-lab ml         # synthetic experiment with the frozen weights; --retrain trains anew
uv run ball-request-lab real       # 16 real windows and the de Jesus window
uv run ball-request-lab figures    # Figures 2 to 6 in outputs/figures
uv run ball-request-lab verify     # every checkable statement of the article
```

`--data-dir` and `--out` change where data and results go (`./data`, `./outputs`).
`scripts/download_data.sh --matches 1996435` downloads only the examined match,
enough for the `case` stage. `scripts/reproduce.sh --skip-ml` leaves out the
PyTorch stages. The `Dockerfile` runs everything on a clean Linux image.

## What cannot be reproduced from public data

The article contains material that depends on the broadcast video or on a visual
judgement. These are shipped as frozen inputs, not recomputed, and `verify`
checks them only for consistency with the file that records them.

- **Videos 1 and 2, and the broadcast frame in Figure 1.** They use A-Leagues
  broadcast footage, which is not redistributed here. The 3D reconstruction is
  illustrative and feeds no number.
- **Visual adjudications** (`inputs/`). Whether a raised arm is a probable request,
  and whose arm it is, was judged from broadcast frames and examined by the author.
  They are not independent expert labels.
  The files record each label and its reasoning, so a reader with access to the
  broadcast can check them against the frames.
- **Video synchronisation residuals** (`config/video-sync.json`). Measured against
  the broadcast in an earlier study; the file carries its own provenance.
- **Trained weights** (`inputs/frozen-models/`). The ridge and network weights from the run
  behind the article; see Determinism for why they ship instead of being retrained.
- **Clip anchors of the first review** (`inputs/review-002-anchors.json`). The
  independent window selection excludes players near them; the file keeps only the
  clock anchors the selector reads.

## Caveats

What this repository does not establish, stated plainly. The audit is in `docs/audit.md`.

1. **Training is not portable.** Training a neural network from scratch is not bit-for-bit
   reproducible across CPU architectures. `ml --retrain` lands on the article's numbers on
   macOS ARM and Linux ARM. On Linux x86 it does not (the selected network scores 0.808
   instead of 0.797), and `verify` warns. The article's numbers are reproduced from the
   trained weights of the original run in `inputs/frozen-models/`, which the repository
   evaluates but, on x86, cannot regenerate.
2. **Broadcast material is excluded.** The videos, the broadcast frame in Figure 1, the
   visual labels judged from broadcast frames, and the video synchronisation
   residuals depend on A-Leagues footage that cannot be redistributed. They are frozen
   inputs, checked only for internal consistency, and nothing here can recompute them. The
   labels are not independent expert agreement.
3. **Platforms.** Tested from a clean clone on macOS Apple silicon, Linux ARM and Linux x86
   (emulated). Not tested: Windows, macOS on Intel, native x86 hardware, other library
   versions than the locked ones.

The reproduction therefore covers every number and figure the article derives from public
data, on the platforms above, and not the whole article.

## Determinism

The data processing, the detector, the case measurements and the real-window selection
are deterministic and reproduced exactly. Training a neural network is not bit-for-bit
reproducible across CPU architectures: on ARM it lands on the article's numbers, on x86
it does not (for example 0.808 instead of 0.797 for the selected network). So the
weights of the ridge classifier and of the networks that produced the article's
numbers ship in `inputs/frozen-models/`, and `ball-request-lab ml` evaluates those, as
the article's own real-window evaluation did. With them every statement of the article
passes on macOS ARM, Linux ARM and Linux x86. `ball-request-lab ml --retrain` trains
from scratch into `outputs/ml-retrained`; `verify` then reports whether your machine
lands on the article's numbers and warns if it does not.

Differences that remain are float32 rounding: model scores differ from the reference by
at most 2e-6 (`--ml-tolerance`, default 1e-5), and the figures from the published PNGs by at
most two intensity levels on a few hundred pixels on Linux, which `verify` accepts as the
same figure. `docs/audit.md` has the runs.

## Layout

```
config/       pinned sources, detector and analysis parameters, case and experiment settings
inputs/       frozen judgements and anchors (see above)
expected/     result files and figure hashes from the run behind the article
src/          the analysis code, one module per step
scripts/      download_data.sh and reproduce.sh
docs/         method.md (what each stage computes and assumes), audit.md
```

## Method and limits

`docs/method.md` states what each stage computes and assumes. In short: the
local-arrival index is a relative index under stated movement assumptions, not a
probability of receiving; a head-direction sector is not gaze or attention; route
margins are mechanical scenarios, not pass advice; the synthetic experiment says
nothing about real football; and the 16 real windows contain no verified
raised-arm label, so no real-world recall can be reported.

## Data, licence and credit

Match data and body pose: [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
(MIT), pinned to immutable revisions in `config/sources.json` and downloaded at run
time; none is stored here. Code: MIT, see `LICENSE`. Analysis by Matheus de Amorim.
