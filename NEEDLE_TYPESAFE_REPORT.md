# Needle 3 vs. TypeSafe classification benchmark

**Run date:** September 17, 2026

**Scope:** Zero-shot typed classification and routing, not free-form reasoning

## Executive summary

TypeSafe was substantially more accurate and faster on this small synthetic
classification suite. Across the 40 cases with an unambiguous expected answer,
TypeSafe answered 38 correctly (95.0%) and Needle 3 answered 18 correctly
(45.0%).

| System | Accuracy | Mean latency | Median latency | P95 latency |
|---|---:|---:|---:|---:|
| TypeSafe | 38/40 (95.0%) | 299 ms | 280 ms | 408 ms |
| Needle 3 | 18/40 (45.0%) | 11,826 ms | 4,738 ms | 48,235 ms |

In this run, TypeSafe was about 39.5x faster by mean latency, 16.9x faster by
median, and 118x faster at P95. This is not a hardware-normalized throughput
comparison: TypeSafe was a hosted service, while Needle used its released local
CPU inference engine. The latency result describes these tested deployments,
not the maximum performance either system could achieve.

Needle successfully exposed the basic behaviors needed for a router: selecting
one tool, returning no tool when none applied, extracting typed arguments, and
selecting multiple tools. However, the base model was not a drop-in replacement
for TypeSafe's classification primitives on this workload. Its performance was
best for binary decisions and weakest for multi-class routing.

## What we tested

The benchmark contains 46 English cases covering three decision types:

- **Binary decision:** determine whether a user explicitly requested a refund.
- **Multi-class route:** select billing, account access, technical support, or
  general support.
- **Ordinal score:** assign incident severity from cosmetic through critical.

Forty cases have clear expected answers and are scored. Six deliberately
ambiguous cases are retained for behavioral inspection but excluded from
accuracy. The scored set includes 28 ordinary cases and 12 adversarial cases,
with six long-context cases embedded across those groups.

Adversarial examples include quoted instructions, negation, misleading labels,
prompt-injection-like text, user-supplied severity claims, and attempts to
downplay serious incidents. Long-context pairs test whether relevant facts
survive substantial irrelevant context.

TypeSafe used its native Noul, Choice, and Score outputs. Needle used its native
tool-selection interface:

- A binary action tool, where selecting it means true and returning no call
  means false.
- One named tool for each routing option.
- One named tool for each severity level.

An initial generic adapter was rejected before the final run because it did not
represent Needle naturally. The reported run uses the native action/refusal
shape and explicit tools for each class. This improved the fairness of the
comparison, though tool wording remains an important experimental variable for
Needle.

## Results by decision type

| Decision type | Cases | TypeSafe | Needle 3 | Needle mean latency |
|---|---:|---:|---:|---:|
| Binary | 12 | 11/12 (91.7%) | 9/12 (75.0%) | 10,122 ms |
| Multi-class route | 14 | 14/14 (100%) | 4/14 (28.6%) | 9,740 ms |
| Ordinal severity | 14 | 13/14 (92.9%) | 5/14 (35.7%) | 15,403 ms |

Binary classification was Needle's strongest category. Its larger gap was in
distinguishing semantically adjacent tools and in mapping incident descriptions
to an ordered severity scale. Common Needle errors included routing technical
failures to billing, assigning low severity to authentication bypasses, and
following misleading wording embedded in the input.

## Ordinary, adversarial, and long context

| Slice | Cases | TypeSafe | Needle 3 |
|---|---:|---:|---:|
| Ordinary | 28 | 27/28 (96.4%) | 16/28 (57.1%) |
| Adversarial | 12 | 11/12 (91.7%) | 2/12 (16.7%) |
| Long context | 6 | 6/6 (100%) | 4/6 (66.7%) |

Needle's largest accuracy decline was adversarial: from 57.1% on ordinary cases
to 16.7% on adversarial cases. TypeSafe declined from 96.4% to 91.7%.

Needle handled four of the six long-context cases correctly, but its mean
latency for those cases was 59.8 seconds because the released engine performs
prefill on CPU. TypeSafe answered all six in a mean of 288 ms. With only six
long-context cases, this slice is directional rather than conclusive.

## Confidence behavior

TypeSafe's mean certainty was 0.920 on correct answers and 0.450 on incorrect
answers. Needle's mean confidence was 0.894 on correct answers and 0.954 on
incorrect answers. On this suite, Needle was therefore more confident when it
was wrong than when it was right.

The values are not identical constructs. TypeSafe derives confidence from its
classification output, while Needle reports aggregate tool-call confidence.
Even with that qualification, Needle's value was not safe to use as an
accuracy gate in this run. It would require task-specific calibration on a
held-out set.

## Functionality smoke tests

Separate from the scored suite, Needle completed four native tool-use checks:

| Capability | Observed behavior |
|---|---|
| Single-tool routing | Correctly routed a duplicate charge to billing |
| Typed extraction | Produced a structured support ticket with category, priority, channel, and summary |
| Refusal/no action | Returned no call for an unrelated poetry request |
| Multiple actions | Selected both lighting and thermostat actions from one request |

Peak observed memory was approximately 147 MB. This supports Needle's stated
edge-oriented design, even though its zero-shot classification accuracy was not
sufficient for this particular router.

## Interpretation

Needle can implement the mechanics of a TypeSafe-like router, but the base
Needle 3 model did not reproduce TypeSafe's reliability in this experiment.
The distinction is important:

- TypeSafe directly supplies typed decision primitives and distributions.
- Needle is a compact tool-calling model whose behavior depends heavily on tool
  descriptions and is designed to be customized for specific applications.
- Needle does not provide TypeSafe's full class distributions through the
  tested interface.

The most promising Needle use here is a narrow, locally deployed binary gate or
a task-specific fine-tuned router—not an immediate zero-shot replacement for a
general multi-class classifier. A fair next experiment would fine-tune Needle
on the routing taxonomy, freeze the tool schemas, calibrate confidence on a
validation set, and then rerun this exact held-out suite.

## Limitations

- This is one run over 40 scored synthetic English examples.
- The suite measures typed classification and routing, not general reasoning.
- The systems used different serving stacks and hardware, so latency is not a
  pure model-speed comparison.
- Needle was tested zero-shot; its documented customization path was not used.
- Schema wording can materially affect tool-selection models.
- The suite was not designed to measure every safety, multilingual, or
  domain-specific behavior.

## Reproduction and public references

Run `compare_typesafe_needle.py` with service locations and credentials supplied
only through environment variables. The script imports the same frozen cases as
the original TypeSafe comparison and does not write credentials or service
locations into its output.

- [Needle source and documentation](https://github.com/cactus-compute/needle)
- [Needle 3 model card](https://huggingface.co/Cactus-Compute/needle3)
- [Needle Apache 2.0 license](https://github.com/cactus-compute/needle/blob/main/LICENSE)
- [TypeSafe confidence documentation](https://docs.typesafe.ai/confidence)
