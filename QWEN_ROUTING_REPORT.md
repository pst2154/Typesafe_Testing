# Jev vs. Qwen3.8-27B for request routing

Date: 2026-09-16

## Result

Jev and `nvidia/qwen/qwen3.8-27b` achieved identical accuracy on this routing
set. Jev had substantially lower end-to-end latency.

| System | Correct | Accuracy | Mean latency | Median latency | Observed p95 | Valid schema |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 64/64 | 100.0% | 228 ms | 223 ms | 260 ms | 72/72 |
| Qwen3.8-27B | 64/64 | 100.0% | 1,887 ms | 1,195 ms | 5,642 ms | 72/72 |

Jev was 8.3x faster by mean latency and 5.4x faster by median latency. Qwen's
observed p95 was 21.7x Jev's p95. These are hosted end-to-end measurements, so
queueing and serving configuration are included and the result is not a comparison
of isolated model compute.

## Method

The benchmark reused the frozen routing set from the earlier comparison:

- 24 ordinary scored cases;
- 8 adversarial scored cases containing misleading route instructions;
- 4 ambiguous cases excluded from exact-match accuracy;
- 4 routes: deterministic tool, small model, reasoning model, and human review.

Every case ran twice in shuffled order, producing 72 requests per system and 64
scored decisions per system. Call order alternated between systems. Jev received a
native Choice question. Qwen received the same policy, four examples, a JSON
contract, and JSON mode. No labels or prompts were changed after observing the
earlier model results.

## Accuracy breakdown

| Group | Jev | Qwen3.8-27B |
|---|---:|---:|
| Ordinary | 48/48 | 48/48 |
| Adversarial | 16/16 | 16/16 |
| Deterministic tool | 16/16 | 16/16 |
| Small model | 16/16 | 16/16 |
| Reasoning model | 16/16 | 16/16 |
| Human review | 16/16 | 16/16 |

Both systems also had 100% coverage and 100% accuracy at the benchmark's 0.60
confidence threshold. Jev confidence comes from its probability distribution;
Qwen confidence is self-reported generated output, so equal threshold results do
not establish equal calibration.

## Ambiguous cases

The systems disagreed only on cases intentionally excluded from accuracy. Jev
routed a brief suspicious-login question to the reasoning model with low confidence,
while Qwen chose the small model. Jev routed a disputed-records summary intended
for a legal filing to human review, while Qwen chose the small model. These examples
show policy boundaries worth clarifying before production deployment.

## Interpretation

Qwen3.8-27B solved the frozen routing set and corrected the severe accuracy and
schema-compliance problems observed with the 1B baseline. It is a viable quality
baseline for this policy. Jev delivered the same measured routing accuracy with a
much lower and tighter latency distribution and returned native typed probabilities.

For a front-door router, the observed result favors Jev. Qwen remains useful when
the routing decision must be combined with generated analysis, or when its broader
capabilities justify the additional latency.

## Limitations

- The cases are synthetic and labeled by one evaluator.
- Two repetitions improve on a single pass but are still too few for stable tail
  latency estimates or confidence intervals.
- Perfect accuracy on 32 unique scored cases does not establish production accuracy.
- The adversarial cases are static and not adaptive attacks.
- Confidence values from the two systems have different semantics.
- Cost, concurrency, throughput, token use, and hardware efficiency were not measured.

## Recommended next test

Expand the frozen set with independently labeled production-like requests near the
route boundaries, then run at least 10 randomized repetitions. Keep exact-match
accuracy, schema compliance, latency, confidence calibration, and abstention quality
as separate metrics.
