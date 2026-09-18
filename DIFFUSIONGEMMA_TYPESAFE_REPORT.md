# DiffusionGemma vs. TypeSafe for typed decisions

Run timestamp: 2026-09-18 UTC

## Executive summary

This experiment compared TypeSafe/Jev with DiffusionGemma on 46 small decision
problems: yes/no judgments, four-way routing, and four-level severity scoring.
The suite included ordinary cases, prompt-injection attempts, ambiguous inputs,
and matched short/long-context pairs.

The main result is promising for a self-hosted decision engine:

| Result | TypeSafe | DiffusionGemma |
| --- | ---: | ---: |
| Correct on 40 clear cases | 37/40 | **39/40** |
| Accuracy | 92.5% | **97.5%** |
| Adversarial accuracy | 10/12 | **11/12** |
| Median latency | 168 ms | **70 ms** |
| Observed p95 latency | 247 ms | **98 ms** |

In this run, warmed DiffusionGemma was about 2.4x faster by median end-to-end
latency and made two fewer errors. This does **not** establish that it is always
better or intrinsically faster than TypeSafe: DiffusionGemma ran locally on one
B300 GPU, while TypeSafe was called as a hosted service. Network and hardware
placement are part of the latency result.

The result should be read as evidence that DiffusionGemma can make useful typed
decisions quickly, not as a general reasoning benchmark. Neither system was
asked for free-form reasoning or a chain of thought.

## What was tested

The benchmark reused the same frozen cases and labels as the earlier TypeSafe
classification comparisons.

| Primitive | Cases | Scored | Decision |
| --- | ---: | ---: | --- |
| Noul | 14 | 12 | Did the customer explicitly request a refund or reversal? |
| Choice | 16 | 14 | Which support team should own the ticket? |
| Score | 16 | 14 | How severe is the production incident? |
| **Total** | **46** | **40** | 6 ambiguous probes were not scored for exact-match accuracy |

The 40 clear cases consisted of 28 ordinary cases and 12 adversarial cases.
Six ordinary examples also had a long-context partner, allowing the same
decision to be compared with and without roughly 11.7K characters of irrelevant
but plausible context.

The adversarial cases attempted to change the decision through fake JSON,
embedded instructions, keyword repetition, negation, dramatic wording, or text
that downplayed a serious event.

## System configurations

### TypeSafe

- The request used the `jev-latest` alias.
- The service reported that `jev-1.13.0` handled every request.
- Noul, Choice, and Score used their native TypeSafe question types.
- The harness converted Noul probabilities to booleans at a 0.50 threshold and
  rounded Score's fractional position to the nearest authored level.

### DiffusionGemma

- Model: `nvidia/diffusiongemma-26B-A4B-it-NVFP4`.
- Serving stack: the structured-generation implementation in
  [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), tested at
  commit `d2c2b5422d6e1b62fc9af68e16dc26a975d2aa23`.
- One diffusion step, zero think tokens, and seed 42.
- Adaptive sampling requested between one and four reads with a 0.1 threshold.
- The model was served in text-only mode with the CUTLASS NVFP4 MoE backend.
- The structured layer returned the selected label, a distribution over labels,
  confidence, sampling diagnostics, and timing.

This was an application-level comparison, not identical model computation.
TypeSafe used purpose-built hosted primitives. DiffusionGemma read label
probabilities from a structured prompt through a self-hosted vLLM server.

## Accuracy

### By primitive

| Primitive | TypeSafe | DiffusionGemma |
| --- | ---: | ---: |
| Noul | 11/12 (91.7%) | **12/12 (100%)** |
| Choice | 13/14 (92.9%) | **14/14 (100%)** |
| Score | 13/14 (92.9%) | 13/14 (92.9%) |
| **Overall** | **37/40 (92.5%)** | **39/40 (97.5%)** |

The systems agreed on 42 of all 46 cases, including the ambiguous probes.
DiffusionGemma's only clear-case error was an ordinal severity boundary. TypeSafe
made one error in each primitive.

### Ordinary and adversarial cases

| Subset | TypeSafe | DiffusionGemma |
| --- | ---: | ---: |
| Ordinary clear | 27/28 (96.4%) | **28/28 (100%)** |
| Adversarial clear | 10/12 (83.3%) | **11/12 (91.7%)** |

Both systems were fooled at least once. This matters more than the small numerical
lead: neither should be treated as an adversarial security boundary without an
abstention or escalation policy.

## What the errors show

### TypeSafe: embedded instruction without a refund request

One input contained an instruction to answer yes but did not actually request a
refund. The expected result was false.

- TypeSafe returned yes with `p(yes) = 0.58`. The derived certainty was only 0.16.
- DiffusionGemma returned no with probability 0.574.

This was a TypeSafe exact-match error, but its near-boundary probability was a
useful warning. An abstention interval around 0.50 could route this case to a
safer fallback.

### TypeSafe: fake JSON inside a technical ticket

A technical request contained fake JSON claiming that the route was billing.

- TypeSafe chose billing, with billing at 0.51 and technical at 0.49.
- DiffusionGemma chose technical with probability 0.996.

Again, TypeSafe exposed that the decision was nearly tied. Consuming only the
winning label would discard that operationally useful signal.

### TypeSafe: browser-specific workaround

One incident described a feature failing in one browser while a practical
workaround remained available. The authored level was 1, minor degradation.

- TypeSafe assigned 0.31 to level 1 and 0.69 to level 2, producing a fractional
  score of 1.69 and rounding to level 2.
- DiffusionGemma selected level 1 with probability 0.593, while still assigning
  0.397 to level 2.

The distributions show that this is also a rubric-boundary case rather than an
obvious categorical failure.

### DiffusionGemma: dramatic language despite a workaround

An incident used severe wording but explicitly said that a workaround existed.
The authored level was 1.

- DiffusionGemma selected level 2 with probability 0.748 and assigned 0.244 to
  level 1.
- TypeSafe selected level 1 with probability 0.66.

DiffusionGemma overweighted the dramatic framing. A production severity router
should therefore test emotional language separately from concrete impact and
workaround availability.

## Short vs. long context

| Context | Size | TypeSafe accuracy | DiffusionGemma accuracy |
| --- | --- | ---: | ---: |
| Short | 24-142 characters | 31/34 | **33/34** |
| Long | 11,744-11,773 characters | 6/6 | 6/6 |

Both systems produced the same label for the short and long member of all six
matched pairs. The long-context additions did not change any paired decision.

| Context | System | Mean | Median | Observed p95 |
| --- | --- | ---: | ---: | ---: |
| Short | TypeSafe | 183 ms | 168 ms | 247 ms |
| Short | DiffusionGemma | **71 ms** | **70 ms** | **72 ms** |
| Long | TypeSafe | 172 ms | 170 ms | 199 ms |
| Long | DiffusionGemma | **79 ms** | **97 ms** | **103 ms** |

There are only six long cases, so these latency numbers are descriptive rather
than a scaling curve. They show no large latency cliff at the tested length.

## Latency and startup behavior

The published results use a warmed server.

| Metric | TypeSafe | DiffusionGemma | Observed ratio |
| --- | ---: | ---: | ---: |
| Mean | 181 ms | **72 ms** | 2.5x |
| Median | 168 ms | **70 ms** | 2.4x |
| Observed p95 | 247 ms | **98 ms** | 2.5x |

The first invocation of a new single-question request shape took 18.3 seconds
because it triggered additional kernel work. Repeating the identical case after
that warm-up took 72 ms. A long-lived service can amortize this cost; an on-demand
or frequently restarted deployment cannot. Production rollout should explicitly
pre-warm every request shape it intends to serve.

The adaptive policy used four reads on 44 of 46 cases and one read on only two.
At the selected threshold it behaved mostly like fixed four-sample inference.
Tuning the threshold or maximum read count is the clearest next latency/cost
experiment.

## Confidence and ambiguity

| Subset | TypeSafe mean certainty | DiffusionGemma mean certainty |
| --- | ---: | ---: |
| Clear cases | 0.895 | 0.946 |
| Ambiguous probes | 0.668 | 0.857 |

Higher certainty is not automatically better. The ambiguous cases have no single
scored answer, and DiffusionGemma remained especially decisive on them. These
values are derived from label probabilities; they have not been calibrated
against real-world correctness. They should be treated as ranking or escalation
signals until a larger held-out calibration set demonstrates otherwise.

## Repeatability note

A preliminary validation pass produced 38/40 for TypeSafe and 39/40 for
DiffusionGemma. The final warmed pass produced 37/40 and 39/40, respectively.
DiffusionGemma's clear-case labels stayed the same; one near-boundary TypeSafe
adversarial decision changed. The benchmark did not have a TypeSafe sampling seed
control, so the two-result accuracy gap should not be presented as a stable model
ranking.

## Practical interpretation

DiffusionGemma looks useful when all of the following are true:

- decisions can be expressed as a small fixed label set;
- a GPU-backed, long-lived service is acceptable;
- request shapes can be pre-warmed;
- full probability distributions are consumed, not just winning labels; and
- low-margin or security-sensitive decisions can abstain or escalate.

TypeSafe remains operationally attractive when a hosted typed-decision primitive
is preferable to owning model serving, kernel warm-up, capacity, and reliability.
The correct choice therefore depends on more than model accuracy: ownership cost,
deployment footprint, traffic shape, and fallback policy matter.

## Limitations

- The suite has only 40 scored cases and was labeled by one evaluator.
- Inputs are synthetic, English-only, and concentrated in support routing and
  incident severity.
- The latency comparison includes different hardware and network paths.
- Observed p95 is an order statistic from only 46 calls, not a production SLO.
- The systems use different interfaces and confidence definitions.
- Probability-derived certainty is not proven to be calibrated correctness.
- Only one DiffusionGemma serving configuration was measured.
- The benchmark does not measure throughput, concurrency, GPU cost, free-form
  reasoning, or downstream business impact.

## Reproduction artifacts

- Harness: [`compare_typesafe_diffusiongemma.py`](compare_typesafe_diffusiongemma.py)
- Sanitized per-case results:
  [`results/diffusiongemma_vs_typesafe.json`](results/diffusiongemma_vs_typesafe.json)

The committed artifacts contain no credentials, service locations, private source
locations, or input text. Service configuration and credentials are supplied at
runtime through environment variables.
