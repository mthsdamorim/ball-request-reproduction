# Method

What each stage computes and assumes. The article gives the reasoning; this page
says where the numbers come from. Parameters live in `config/`.

## `prepare`: one clock for tracking and pose

Tracking is sampled at 10 Hz and pose at 25 Hz. Every second tracking frame and
every fifth pose frame form a shared 5 Hz grid. Period and clock must agree
between the two sources or the run stops. Where they disagree on who has the
ball, the holder is left unknown. Missing joints stay missing: nothing is
interpolated or upsampled.

## `detect`: the frozen geometric rule

A player is a candidate while, for three shared samples (0.4 s) in a row, one
wrist is at least 0.08 m above the shoulder and 0.02 m above the elbow, with the
predicted error of shoulder, elbow and wrist at most 20 cm and plausible arm
segment lengths. Only outfield teammates of a known holder, while the ball is
in play, are eligible. The rule was fixed before the results were read
(`config/detector.toml`).

Each candidate is then described by three separate measurements at the
confirmation sample:

- **Space.** At each point of a disk of 4 or 5 m around the player, a logistic
  transform (temperature 0.45 s) of how much earlier the team's closest player than the
  closest opponent could arrive, averaged over the disk. Movement assumptions:
  0.2 s reaction, 3 m/s² acceleration, 5 m/s speed ceiling. A relative index,
  not a probability.
- **Potential view.** A head direction from the ear baseline, signed by the nose.
  Whether the player's position falls within a half-angle of 60, 90 or 110
  degrees of it. This is a geometric sector, not gaze or attention.
- **Response.** The ball holder's next action, read only after the fact and never
  used as an input.

## `case`: the examined sequence

The rule returns no state for Z. de Jesus (match 1996435, 513.8 s), because the
predicted error of his raised wrist is above 20 cm. The event is therefore
defined by eye (`inputs/candidate-005.json`) with the detector's own timing rule
and passed through the same analysis. `inputs/adjudication-005.json` records the
visual judgement behind it.

`case` also builds the time-aligned samples from one second before the onset to
the pass, with the context second left without possession or sector, and a
mechanical ground-pass scenario for each sample: a straight segment from the
ball to the player's current position, a ball at a constant 10, 15 or 20 m/s,
released at once. The margin at a point is the earliest opponent arrival minus
the ball arrival; negative means an opponent is first. Offside, body
orientation, execution, receiver movement, first touch, deceleration and
height are not modelled. Re-evaluating the route at 0.25 m instead of 0.5 m
spacing changes no sign.

`context.json` adds descriptive facts from the provider's own files: lineup
roles, phase of play, positions and distances, the options the provider lists
with its completion and threat estimates (which this project neither computed
nor validated), and the possessions that follow.

## `ml`: the synthetic positive control

Procedural 29-joint movements in five classes (no signal, one arm raise, two arm
raises, directional point, hand to head), 11 frames at 5 Hz. Bone lengths are
fixed within a base movement, and noise and occlusion are added afterwards.
Bases are split before augmentation (400 train, 100 validation, 150 test); the
late and pulse temporal families are held out for test. Models: the geometric
rule, a ridge classifier on temporal features and a two-layer temporal
convolutional network (three seeds, selected by validation loss), plus a
shuffled-label control. Class balance and rates are not football rates.

## `real`: sixteen real windows

A deterministic selector draws four player-windows per match half, uniformly
within chronological strata, using only eligibility and pose quality: no detector
output, model prediction or pass outcome. Players within 15 s of the clips of the
first review are excluded (`inputs/review-002-anchors.json`). The windows are
frozen before any label is attached. `inputs/real-adjudications.json` holds the
visual labels, made from broadcast frames; only two windows had identity and gesture
evidence sufficient for evaluation, both labelled no signal. The frozen models are
applied without fitting. The same models are applied to the de Jesus window, which
sits outside this evaluation.

## `verify`

Each statement of the article that can be recomputed is compared with the
recomputed value at the precision the article uses. Result files are also
compared with those of the run behind the article, and the five PNG figures with
the published ones.

## Limits

Tracking and pose are generated separately and can be slightly misaligned.
Positions flagged as extrapolated stay in, and results that use them are
conditional. The visual labels are not independent expert
agreement. One sequence, a small real sample and a synthetic experiment support
descriptions and caution, not recognition performance or decision quality.
