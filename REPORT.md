# TypeSafe vs. SOL classification comparison

Date: 2026-09-16

## Executive summary

This exploratory benchmark compared TypeSafe typed judgments with SOL-generated
structured classifications across binary, multiclass, and ordinal tasks. It also
tested long context and deliberately adversarial inputs.

| System | Scored cases correct | Accuracy | Mean latency | Median latency | Observed p95 |
|---|---:|---:|---:|---:|---:|
| TypeSafe | 38/40 | 95.0% | 237 ms | 231 ms | 278 ms |
| SOL | 40/40 | 100.0% | 2,464 ms | 2,134 ms | 4,009 ms |

TypeSafe was 10.4x faster by mean latency and 9.2x faster by median latency in
this run. SOL scored two additional cases correctly under the authored labels.
The sample is too small and curated to support a general quality claim.

TypeSafe and SOL expose different products:

- TypeSafe returns constrained typed values and probability distributions for
  code to act on directly.
- SOL generates a classification, self-reported confidence, and short evidence
  that must be prompted and parsed.
- Neither output is a private reasoning trace. SOL's evidence is a concise
  explanation of its answer, not proof of how the model reached it.

## Method

The benchmark contained 46 total cases:

- 28 ordinary clear cases with authored expected labels;
- 12 adversarial clear cases containing misleading instructions, fake labels,
  keyword stuffing, quotations, or negation;
- 6 intentionally ambiguous cases used only to inspect uncertainty;
- 6 short/long pairs included in the ordinary clear set to test context-length
  stability.

Each case was evaluated once by each system. Wall time covered the classification
request only. Calls were sequential, and network conditions, backend load, and
cold starts were not controlled.

### Classification shapes

1. **Binary:** whether a customer explicitly requested a refund or reversal.
2. **Multiclass:** routing among billing, technical, account access, and other.
3. **Ordinal:** incident severity from cosmetic through critical.

TypeSafe used its native binary, choice, and score primitives. SOL was instructed
to return a JSON object containing a label, confidence, and evidence of at most
20 words. All 46 SOL responses were valid and parseable in this run.

## Accuracy and latency by classification type

| Classification | TypeSafe correct | SOL correct | TypeSafe mean | SOL mean | Speed ratio |
|---|---:|---:|---:|---:|---:|
| Binary | 11/12 | 12/12 | 246 ms | 2,279 ms | 9.3x |
| Multiclass | 14/14 | 14/14 | 240 ms | 2,629 ms | 10.9x |
| Ordinal | 13/14 | 14/14 | 225 ms | 2,460 ms | 10.9x |

The ambiguous cases are excluded from correctness denominators but included in
the latency averages. Long-context pairs count as separate ordinary clear cases.

## Adversarial classification

The 12 adversarial cases attempted to redirect the classifier with text such as
fake output labels, commands to choose the wrong category, repeated irrelevant
keywords, or dramatic/downplaying language that conflicted with the facts.

| System | Ordinary correct | Adversarial correct | Adversarial mean latency |
|---|---:|---:|---:|
| TypeSafe | 27/28 | 11/12 | 237 ms |
| SOL | 28/28 | 12/12 | 2,582 ms |

Both systems handled quoted refund language, negation, fake JSON, keyword
stuffing, and attempts to inflate or suppress severity. TypeSafe's one adversarial
miss was a binary case that said to output “yes” while describing only a shipping
status question. It returned 0.54 for “yes,” just over the 0.50 decision threshold;
the derived certainty was only 0.08. SOL returned “false” with self-reported
confidence 0.99.

This failure is a useful policy example: a hard threshold forced a low-margin
TypeSafe result into a label, while a modest abstention band around 0.50 would
have escalated it. The run does not establish adversarial robustness against
adaptive attackers; it only measures these 12 hand-authored attempts.

## The ordinary ordinal disagreement

The other scored disagreement involved a browser-specific export failure with a
working alternative browser. The authored expected level was 1, “minor degradation
with a practical workaround.”

- SOL selected level 1 with self-reported confidence 0.98.
- TypeSafe returned a fractional score of 1.65: probability 0.35 on level 1 and
  0.65 on level 2, with confidence 0.65. Rounding selected level 2.

The authored level 2 also described a major function as unavailable while the
broader service still operates, so the levels overlap for a browser-specific
failure. TypeSafe's distribution exposed that boundary conflict. A follow-up
should tighten the rubric and evaluate revised wording on held-out cases.

## Ambiguity and uncertainty

For TypeSafe binary answers, certainty was derived as `2 × |p(yes) - 0.5|`.
For TypeSafe choice and score answers, the returned confidence was used. SOL's
confidence is self-reported, so its absolute values are not directly calibrated
against TypeSafe's values.

| Group | TypeSafe mean certainty | SOL mean confidence |
|---|---:|---:|
| Ordinary clear | 0.951 | 0.990 |
| Adversarial clear | 0.756 | 0.992 |
| Ambiguous | 0.673 | 0.845 |

TypeSafe's certainty fell substantially on adversarial and ambiguous inputs,
including the near-threshold adversarial miss. SOL remained highly self-confident
on the adversarial group. This does not by itself mean one confidence signal is
better calibrated; calibration requires repeated representative data and proper
scoring rules.

## Short versus long context

Six clear cases were paired. The short versions averaged 50 characters. The long
versions averaged 11,755 characters and placed the same decisive evidence after
neutral distractor context.

| System | Short correct | Long correct | Stable pairs | Short mean | Long mean | Long/short |
|---|---:|---:|---:|---:|---:|---:|
| TypeSafe | 6/6 | 6/6 | 6/6 | 242 ms | 250 ms | 1.03x |
| SOL | 6/6 | 6/6 | 6/6 | 2,099 ms | 2,213 ms | 1.05x |

Both systems preserved every paired decision. Long-context latency was about 3%
higher for TypeSafe and 5% higher for SOL in this run. One sample per pair is
insufficient to characterize scaling; future work should repeat randomized calls
across several token-length bands.

## How the classification approaches differ

### Binary decisions

TypeSafe's binary primitive directly returns the probability of “yes.” This makes
threshold and abstention policy explicit in code. SOL can return the same hard
label, but its confidence is generated text and needs application-specific
validation before it is treated as a probability.

### Multiclass decisions

TypeSafe returns the selected class plus probability for every candidate. Competing
classes remain visible, which supports secondary routing, escalation, and abstention.
SOL produced useful short evidence, but the test contract exposed only its selected
class and self-reported confidence.

### Ordinal decisions

TypeSafe returns a probability-weighted position that can fall between levels. This
preserves boundary information but requires the application to decide whether to
round, threshold, rank, or retain the continuous value. SOL emitted a single discrete
level, which was convenient for exact-match scoring but discarded boundary shape.

## Limitations

- The cases are synthetic, English-only, and intentionally compact.
- Expected labels were authored by one evaluator and were not independently reviewed.
- Each case was called once; latency percentiles are descriptive observations only.
- Six long cases do not characterize maximum context behavior or scaling.
- Twelve static adversarial cases do not represent adaptive attacks or prove safety.
- SOL's self-reported confidence and TypeSafe's distribution-derived confidence have
  different meanings and should not be compared as if identically calibrated.
- The benchmark measures classification, not downstream task quality, cost, or safety.

## Recommended next tests

1. Repeat every case several times in randomized order and report confidence intervals.
2. Add independently labeled real examples and adjudicate disagreements blind to system.
3. Separate rubric-development data from a held-out evaluation set.
4. Test several context bands with relevant contradictions and evidence at the
   beginning, middle, and end.
5. Generate adaptive adversarial inputs against a held-out classifier specification.
6. Evaluate calibration with reliability diagrams and proper scoring rules.
7. Measure abstention policies, schema-failure rates, token usage, and end-to-end cost.
