# Audit of the reproduction

Premise: someone who clones this repository can regenerate every number and figure of
the article from public data. Tested on 17 October 2026.

## Runs

Each run started from the tracked files only, with no data and no outputs, downloaded
the 1.4 GB of public inputs itself, ran every stage, then ran `ml --retrain` and `verify`.

| Environment | Result |
| --- | --- |
| macOS, Apple silicon, Python 3.13.5, locked dependencies | 107 passed, 0 failed, 0 warnings |
| Linux ARM (Docker, aarch64), Python 3.13, locked dependencies | 107 passed, 0 failed, 0 warnings |
| Linux x86 (Docker, emulated amd64), Python 3.13, locked dependencies | 106 passed, 0 failed, 1 warning |

A further run used a real `git clone` of the published repository on macOS ARM and `scripts/reproduce.sh` alone: 106 passed, 0 failed, 0 warnings, in 119 s (the retraining check is not part of that command).

Each full run takes about two minutes on ARM. The single warning is the optional
retraining check on x86, described below. Every statement of the article, every result
file and every figure passes on all three.

## What the checks cover

- The ten input files match their pinned size and SHA-256.
- Every number the article states about the detector, the de Jesus case, the game
  context, the synthetic experiment and the real windows, at the precision the article
  uses (`ball-request-lab verify` prints each one).
- Ten result files equal the files from the run behind the article, field by field.
- The five PNG figures equal the published ones, byte for byte on macOS, and within two
  intensity levels on a few hundred pixels on Linux, which is the same figure.

The check is not vacuous. Changing one movement assumption in a copy of the configuration
makes `verify` fail on the affected statement and exit non-zero, and a corrupted input
file is detected and downloaded again.

## What the audit found and fixed

**Wording.** Three statements did not survive recomputation as worded. None changes a
result. The article, the evidence ledger and the Substack draft were corrected.

| Article said | Recomputed | Now says |
| --- | --- | --- |
| de Jesus "eleven metres" inside the Adelaide half | 11.57 m | twelve metres |
| Van Der Saag's goalkeeper "within about 22 degrees" of the head direction | 22 degrees in one native frame, 24.7 degrees on the 5 Hz grid at confirmation, 18 to 32 across single frames | about 25 degrees off the head direction that pointed at the carrier |
| Five Adelaide players "within four metres" of the halfway line | largest distance 4.06 m | within about four metres |

**Training is not portable.** A first version retrained the ridge classifier and the
networks on every run. That reproduced the article on ARM but not on x86: the selected
network scored 0.808 instead of 0.797, five synthetic-experiment numbers of the article
failed, and the ridge weights differed across BLAS builds by up to 1.3e-3, enough to move
ridge scores on real windows by 0.02. The cause is floating-point order in training and in a
32-bit normal-equation solve, which no seed fixes. The weights that produced the article's
numbers now ship in `inputs/frozen-models/`, and `ml` evaluates them, as the article's own
real-window evaluation did. `ml --retrain` still trains from scratch, and `verify` reports
whether the machine lands on the article's numbers: it does on macOS and Linux ARM, and
does not on x86, where it warns.

## What is not reproduced

By design, and stated in the README: the broadcast video, the three media built from it,
the visual labels that were judged from its frames, the synchronisation residuals measured
against it, and the clip anchors of the first review. They are frozen inputs, checked for
internal consistency only. The visual labels have no independent human agreement
behind them, and this repository cannot supply that.

## Not tested

Windows and macOS on Intel. The x86 run used emulation, not native hardware. Other
PyTorch or NumPy versions than the locked ones may differ in the last digits.
