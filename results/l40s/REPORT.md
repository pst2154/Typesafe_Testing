# L40S FP8 test report

September 18, 2026.

## Validation scope

The service completed typed classification requests on one NVIDIA L40S.
The measured trial used the pinned local runtime image with the fixed adapter
and a separately launched gateway. The GHCR registry manifest was checked, but
the exact all-in-one GHCR launch has not been end-to-end retested on L40S.
These results do not establish production equivalence or general accuracy.

Model loading reported 24.75 GiB, KV cache availability 13.63 GiB, and an observed
post-start total GPU allocation of 41,609 MiB. Runtime warned that no tuned L40S
MoE configuration was available; performance may be suboptimal.

## Measured checks

Four short classification cases: 4/4 correct. First request: 10.41 seconds;
next three: 51–54 ms. This is a smoke test, not broad accuracy evidence.

Warm benchmark: exactly **1,000 state tokens**, excluding schema/instructions,
random numeric padding followed by a refund request, one binary Choice. A fresh
random seed was used for the warm repeat to avoid exact-prefix cache reuse.
Latencies include the local gateway and structured adapter, not external networking.

| Concurrent requests | Measured requests | Requests/sec | Median | p95 |
|---:|---:|---:|---:|---:|
| 1 | 8 | 10.42 | 96 ms | 98 ms |
| 4 | 24 | 14.85 | 269 ms | 273 ms |
| 8 | 24 | 16.49 | 475 ms | 538 ms |

All 56 measured warm requests were correct, with no HTTP errors. This homogeneous
synthetic task does not establish general accuracy or a latency SLA.
The first concurrency-4 pass included approximately 13.84-second first-batch
latencies and achieved only 1.58 requests/sec. Warm results must not hide that
cold-shape overhead; warm representative batch shapes before evaluating capacity.

The 12-question Noul check passed response-shape validation but failed semantic
consistency (4/12 positive at threshold 0.5). Cause is not isolated: checkpoint,
quantization, runtime and one-step structured decoding remain possible factors.
Do not attribute this specifically to L40S hardware or FP8 without controlled tests.

Cold setup included 32 seconds downloading 25.33 GiB of weights, approximately
156 seconds loading weights, and 104 seconds of torch compilation, plus additional
profiling and graph capture. Preserve caches. No inference performance comparison
against H100 with identical settings was run in this trial.

## Raw results

- [Initial benchmark](benchmark-results.jsonl)
- [Warm benchmark](benchmark-warm-results.jsonl)
- [Smoke tests](smoke-results.jsonl)
