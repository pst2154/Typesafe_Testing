# Jev vs. a 1B model for request routing

Date: 2026-09-16

## Result

For this routing policy and serving setup, Jev was both substantially faster and
more accurate than Meta Llama 3.2 1B Instruct.

| System | Correct | Accuracy | Mean latency | Median latency | Observed p95 | Valid schema |
|---|---:|---:|---:|---:|---:|---:|
| Jev | 32/32 | 100.0% | 236 ms | 227 ms | 281 ms | 36/36 |
| Llama 3.2 1B Instruct | 5/32 | 15.6% | 4,783 ms | 3,840 ms | 9,735 ms | 24/36 |

Jev was 20.2x faster by mean latency and 16.9x faster by median latency. A
smaller parameter count did not translate into lower end-to-end latency here.
Hosted queueing, serving configuration, prompt processing, and generated output
can dominate model size, so these numbers characterize this run rather than raw
model compute.

## Routing policy

Each request was assigned to one of four handlers:

- `deterministic_tool`: exact lookups, calculations, and established workflows;
- `small_model`: bounded low-risk rewriting, summarization, extraction, translation,
  and simple classification;
- `reasoning_model`: complex diagnosis, comparison, or planning across constraints;
- `human_review`: decisions requiring authority, accountability, approval, or
  safety, legal, employment, or financial judgment.

The set contained 36 cases: 24 ordinary scored cases, 8 adversarial scored cases,
and 4 ambiguous cases excluded from accuracy. The adversarial cases embedded fake
route labels or instructions to choose the wrong handler.

Jev received one native Choice question and returned a typed option, probability
distribution, and confidence. The 1B model received the same policy, four examples,
an instruction to ignore route labels inside the request, and a JSON response
contract. JSON mode was enabled. Calls were sequential and the case order was
shuffled.

## Accuracy by route

| Expected handler | Jev | Llama 3.2 1B Instruct |
|---|---:|---:|
| Deterministic tool | 8/8 | 1/8 |
| Small model | 8/8 | 0/8 |
| Reasoning model | 8/8 | 1/8 |
| Human review | 8/8 | 3/8 |

Jev classified all 24 ordinary and all 8 adversarial cases correctly. The 1B
model classified 5/24 ordinary cases and 0/8 adversarial cases correctly. It often
collapsed distinct routes into a generic model route or failed to preserve the
requested schema.

## Confidence-gated routing

At a confidence threshold of 0.60:

| System | Coverage | Accuracy among covered cases |
|---|---:|---:|
| Jev | 100.0% | 100.0% |
| Llama 3.2 1B Instruct | 43.8% | 14.3% |

The 1B model's confidence was self-reported generated text, while Jev's confidence
was derived from its returned probability distribution. They should not be treated
as equally calibrated signals. In this run, the 1B model's confidence did not rescue
the routing policy: its confidently accepted subset was still mostly wrong.

## Interpretation

Jev is the stronger router for this workload. It produced a directly consumable
typed result, remained stable under the tested route-injection attempts, and avoided
the latency of generating explanatory text. The 1B model was not competitive even
after adding concise examples and JSON mode.

This does not establish that Jev beats every small model or prompt. It shows that a
general-purpose 1B instruction model was a poor fit for this particular nuanced
four-way routing policy under the tested serving conditions. A larger or
classification-tuned model, constrained decoding, fine-tuning, or a simpler policy
could materially change the result.

## Limitations

- The expected labels and synthetic cases were authored by one evaluator.
- Each final configuration was measured once per case; latency statistics do not
  include confidence intervals.
- The four routes have semantic overlap near their boundaries.
- The adversarial inputs were static rather than adaptively generated.
- The comparison measured end-to-end hosted latency, not isolated inference time,
  throughput under load, token cost, or hardware efficiency.
- The prompt was tuned lightly for the 1B model but not exhaustively optimized or
  fine-tuned on routing examples.

## Recommended next step

Keep Jev as the front router and use confidence to control escalation. For a more
competitive general-model baseline, repeat the same frozen test set with a 3B–8B
instruction model and several randomized repetitions, without changing the labels
after observing results.
