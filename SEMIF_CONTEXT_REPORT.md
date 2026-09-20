# SemIf context and Pac-Man trial

Tested September 20, 2026. This report compares SemIf and DiffusionGemma,
not TypeSafe's proprietary Jev model.

Status: completed and verified. **Do not replace the live game with this SemIf
deployment or claim the adapter improved grounding.** No replacement deployed.

## Model and scope

Actual SemIf source at `ca3ba65f142967030ecb453346e94d6f476a69df`, using
`Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Native upstream MLX direct readout, original checkpoint precision, no generated
answer tokens. Upstream tests: 41 passed; 69 published claims verified; raw
artifact checksums passed. The production game and GPU deployments were not
modified.

## Context diagnostics

| Backend | Relevant | Distractor | Missing evidence | Conflicting non-authoritative advice | Total |
|---|---:|---:|---:|---:|---:|
| DiffusionGemma | 8/8 | 8/8 | 4/8 | 8/8 | 28/32 |
| SemIf / Qwen3.5-4B | 5/8 | 6/8 | 0/8 | 4/8 | 15/32 |

These are authored one-step counterfactual arena questions, not general
accuracy estimates. They test whether supplied fictional protection rules
change the preferred action. The missing-evidence diagnostic offers abstention;
the live game does not. Four missing-evidence pairs are duplicate prompts, so
there are 28 distinct prompts, not 32 independent scenarios. Rotations and
reversed option orders are related cases too. SemIf often continued selecting
ordinary food when the supplied rule made contact safe and more rewarding.

Median diagnostic latency: DiffusionGemma 300 ms, SemIf 589 ms. Different
hardware: remote H100 service versus local Apple Silicon, not an architecture
or GPU-speed comparison. No model training used these diagnostic rows.

## Actual game engine at unchanged speed

Three fresh seeds per condition; up to 300 ticks at 200 ms/tick. Real requests,
no cached moves, no frozen ghosts, no scripted move fallback. This is an engine
test rather than a rendered browser test. Network timing means seeds do not
guarantee identical trajectories. DiffusionGemma conditions ran concurrently
against its multi-replica service; SemIf conditions ran sequentially against one
local model. Previous pending requests can finish after a game is over.

| Backend | Corpus | Scores, seeds 1423 / 1543 / 1663 | Lives left | Per-run median latency |
|---|---|---|---|---|
| DiffusionGemma | Off | 2080 / 1480 / 2120 | 1 / 0 / 3 | 185 / 182 / 181 ms |
| DiffusionGemma | On | 1170 / 1530 / 1470 | 0 / 1 / 2 | 243 / 240 / 240 ms |
| SemIf local | Off | 180 / 160 / 440 | 0 / 0 / 0 | 1738 / 1694 / 1697 ms |
| SemIf local | On | 60 / 30 / 30 | 0 / 0 / 0 | 3181 / 2918 / 3006 ms |

The local SemIf deployment is unsuitable for this game's decision deadline:
ghosts continue moving while it waits seconds for each move. Its game scores
cannot isolate policy quality from latency. It is not a drop-in improvement.
Corpus retrieval also did not improve aggregate DiffusionGemma score in these
three runs; this small result is not a general statement about retrieval.

## Fine-tuning experiment

Training initiated because the frozen context diagnostics exposed failures.
240 training, 48 validation, 48 test synthetic examples; entity names are
disjoint but template families are shared. Four families: explicit attributes,
permission, fictional reward/protection, and queue assignments. Missing evidence,
irrelevant facts, obsolete advice, and shuffled choices are included.

Train only rank-8 q/v LoRA adapters in attention layers among the last four
decoder blocks: 114,688 parameters. Base weights remain frozen. Adam 1e-4,
120 single-example steps, gradient clipping 1.0, seed 31415. Loss is cross-entropy
over SemIf's declared answer-slot logits using its exact prompt, not generated
prose. Select checkpoint by validation loss every 30 steps; retain the initial
zero adapter if none improves validation.

Original model on synthetic splits: validation 41/48, test 40/48. Those scores
do not establish transfer. Paired evaluation reran the original context probes
and a separately frozen 16-case RFC3986 evidence check, whose source document is
absent from the training data. These existing RFC questions were used previously
in another model experiment, so they are not a new blind benchmark.

### Training outcome: reject the trained checkpoints

| Checkpoint | Validation correct | Validation cross-entropy |
|---|---:|---:|
| Original | 41/48 | 0.28547 |
| Step 30 | 43/48 | 1.32096 |
| Step 60 | 41/48 | 1.17345 |
| Step 90 | 42/48 | 0.42769 |
| Step 120 | 42/48 | 0.29853 |

The declared validation-loss selection rule retained **step 0**, not a trained
checkpoint. A few additional correct answers did not compensate for worse
probability assignments on the remaining examples. This does not establish
calibration in general; cross-entropy was the selection metric for this run.
120 single-example optimizer steps saw 120 of the 240 available training rows,
not 120 full epochs. Final selected-model synthetic test remained 40/48.

The saved adapter's LoRA B matrices were independently checked and are all
zero. Loading that selected adapter leaves the original model unchanged. It
must not be advertised as a successfully improved fine-tuned model. Nonzero
intermediate checkpoints were not retained because they failed validation;
therefore the subsequent transfer comparison checks the retained original,
not the generalization of an unsaved trained checkpoint.

## Final paired verification

The evaluation process exited successfully with all 96 expected records:
48 inputs evaluated with the original model and with the selected zero adapter.

| Model | Game-context diagnostics | RFC3986 evidence questions |
|---|---:|---:|
| Original SemIf | 15/32 | 15/16 |
| Selected zero adapter | 15/32 | 15/16 |

There were zero changed choices and a maximum probability difference of zero.
This is an identity check, not a claim of successful fine-tuning. The original
model's strong document result alongside weak counterfactual action results
suggests that extracting a documented fact and applying a conditional rule to
an action are distinct weaknesses to measure; neither score is broad proof of
grounding or planning ability.

## Recommendation

Keep the current deployment rather than switching based on a general leaderboard.
The attempted adapter failed its predeclared validation criterion. A next training
experiment would need broader counterfactual state/rule/action examples and
outcome-labelled game trajectories, with unseen rule structures and mazes held
out. A GPU run would be required before attributing this Mac deployment's
latency failure to SemIf itself. These are recommendations, not completed work.

Row-level predictions, game outcomes, training metrics, and the paired identity
check were retained for this experiment. This publication contains the report,
not the full reproduction bundle or model weights.

## Sources

- [SemIf implementation at the tested revision](https://github.com/TheoLeeCJ/SemIf/tree/ca3ba65f142967030ecb453346e94d6f476a69df)
- [Qwen3.5-4B checkpoint at the tested revision](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
- [URI syntax: RFC 3986](https://www.rfc-editor.org/rfc/rfc3986)
- [StrategyWiki Pac-Man guide](https://strategywiki.org/wiki/Pac-Man), used alongside game-specific authored passages in the game corpus. Arcade-specific mechanics were identified as distinct from this mini-game.
