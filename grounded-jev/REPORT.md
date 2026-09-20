# Grounded typed decisions: Laya and DiffusionGemma

Tested September 20, 2026.

## Outcome

A working RAG decision prototype retrieves public technical documentation, selects a typed answer, and returns source excerpts or abstains. It does not generate prose. Use **DiffusionGemma with the `diverse` one-pass variant** for the tuned prototype. The original two-pass baseline remains available. Laya was faster locally but made more mistakes, including unsupported answers that its own verifier accepted.

This is a **Jev-style interface**, not a deployment or benchmark of TypeSafe's proprietary Jev model. Laya and DiffusionGemma are the actual inference backends.

## Tuning update: faster context use and a real fine-tuning attempt

The original baseline below was frozen and published before tuning. We then tested shorter evidence-first prompts, batched per-option entailment, half-precision Laya, candidate-expanded retrieval, and two-query retrieval diversification. The original held-out set became a **regression set** once its results informed follow-up work; do not treat the tuned scores on it as a new blind test.

| DiffusionGemma path | Original 20-query set, three repeats | p50 / p95 | URI document set, 16 queries, three repeats | p50 / p95 |
|---|---:|---:|---:|---:|
| Original two-pass baseline | 57/60 | 526 / 694 ms | Not measured | — |
| Short evidence-first, one pass | 60/60 | 278 / 360 ms | 39/48 | 300 / 418 ms |
| One pass, diversified retrieval | 60/60 | 271 / 376 ms | 42/48 | 260 / 364 ms |

The tuned path removes the sequential verifier call, reduces instruction/JSON overhead, and fits more source text into the same 512-token budget. It searches both the query alone and the query plus **all** candidate descriptions, interleaving their results. It does not know the expected answer. This avoids relying exclusively on candidate-expanded retrieval, which improved some cases but lost the URI port passage. One-pass results have `verification: null`; they do not claim a second check occurred.

On the regression set, median latency fell about **48%** (526 to 271 ms). This is a workload-level improvement, not a controlled kernel-speed claim. The URI document, RFC 3986, was absent from the training corpus and its 16 questions were written before reading the fine-tuned model's results. Retrieval changes were subsequently tested on it, so the final 42/48 is a **post-tuning validation result**, not a pristine held-out estimate. Its six errors were abstentions, with no wrong non-abstaining answers in this sample. A new blind test remains necessary before claiming general improvement.

### Does the model actually follow changed evidence?

Twelve synthetic context interventions cover fictional retention periods, ports, frame masking and numeric formats: two opposing documented values and one missing-information case per scenario. These are not official RFC statements. DiffusionGemma scored 11/12 with the longer baseline question, 12/12 with the short evidence-first question, and 12/12 with batched entailment. Laya scored 8/12, 9/12 and 9/12 respectively. These are small behavioral probes, not independent general-accuracy estimates.

Explicit entailment was not a universal fix: on the development documentation questions, Laya's entailment variant fell to 11/20; DiffusionGemma's was 18/20 versus 20/20 for its plain evidence-first variant. Laya float16 plus the plain prompt scored 18/20 at 190 ms median on the development set, but still produced wrong answers. Because precision and prompt changed together in that experiment, no isolated float16 speedup is claimed.

### Laya fine-tuning: performed, evaluated, and rejected for serving

We trained **1,052,673 parameters in Laya's decision scorer**, freezing the encoder, contextual transformer head, type embeddings, and auxiliary action head. Frozen contextual features were collected once; AdamW optimized the scorer using those features. This is genuine head-only fine-tuning, not full-model training, LoRA, retrieval caching, or a changed prompt presented as training.

The deterministic synthetic dataset contains 648 training, 216 validation, and 216 test examples. Entity names are disjoint across splits. Nine technical-attribute templates supply supported alternatives, explicit negations, distractor entities, and missing information. Splits share template families, so synthetic accuracy alone cannot establish transfer.

- Initial learning rate `1e-5`, 15 epochs: no validation accuracy improvement; selection retained the original head.
- Second run `1e-4`, 100 epochs, selected by validation accuracy: validation rose from 62.0% to 73.1%; the synthetic test rose from **63.9% to 75.5%**. Seed 31415; batch size 32; clipping 1.0; weight decay 0.01.
- On the new URI document with the same plain RAG interface: untuned Laya scored **24/48 (50%)**, fine-tuned Laya **21/48 (43.8%)**. Wrong non-abstaining answers increased from 18 to 21. The trained head also did worse on the context-intervention probes.

**Do not deploy this adapter as an improvement.** It overfit the synthetic task and failed the document-transfer test. The experimental scorer checkpoint, generated dataset, training histories, and external results are included to make this failure reproducible. Fine-tuned probabilities were not recalibrated. The local service uses the faster DiffusionGemma path, not these weights. Fine-tuned and untuned local timing is not used to claim a speed gain; one tuning evaluation overlapped another local inference process.

## Original baseline measured results

The final comparison uses 20 held-out questions: 15 answerable technical questions and five questions absent from the corpus. Each was run three times, giving 60 requests per backend, not 60 independent questions. One separate warmup was excluded. Accuracy counts expected abstention as correct and abstention on an answerable question as incorrect.

| Backend and deployment | Correct outcomes | Wrong non-abstaining answers | Total abstentions | End-to-end p50 | End-to-end p95 |
|---|---:|---:|---:|---:|---:|
| Laya English, local Apple M3 Pro, MPS | 45/60 (75%) | 3/60 | 27/60 | 276 ms | 406 ms |
| DiffusionGemma NVFP4, remote H100 service | 57/60 (95%) | 0/60 | 18/60 | 526 ms | 694 ms |

Both correctly abstained on all 15 absent-information requests. On the 45 answerable requests, Laya answered 30 correctly, answered three incorrectly, and abstained on 12. DiffusionGemma answered 42 correctly and abstained on three. Median lexical retrieval was approximately 3 ms for both; inference and remote round trips dominate.

The speed figures compare **deployments, not identical hardware**. Laya used local MPS float32 inference on an M3 Pro with 36 GB unified memory. DiffusionGemma used an existing four-replica H100 service; each request was handled by one replica, without tensor or pipeline parallelism. Its timings include network transit. No answer cache was used, but the existing service can reuse prefix caches. No concurrent-client throughput or saturation claim is made.

We also reversed choice descriptions' labels and positions on all 20 held-out questions. Laya remained 15/20 (75%), with 307 ms p50 and 435 ms p95; DiffusionGemma remained 19/20 (95%), with 556 ms p50 and 680 ms p95. This checks one simple source of option bias, not arbitrary adversarial robustness.

## Grounding is distinct from getting the label right

Laya's JWT issuer answer was factually correct but cited an audience-claim passage, which did not support it. It repeated this three times and accepted the same unsupported answer in the label-swapped test. This is a grounding failure even though ordinary answer accuracy counts it as correct. Removing those three unsupported successes lowers Laya's held-out outcome count to **42/60 (70%)**. This adjustment is a targeted manual audit, not a complete blinded citation audit.

Laya also incorrectly said servers must mask outgoing WebSocket frames, and its verifier accepted the mistake. DiffusionGemma instead abstained on the issuer question because retrieval had selected the wrong passage. No wrong non-abstaining answer appeared in its small held-out sample, but that does not establish zero error in production.

The service guarantees that returned excerpts come from checksum-verified corpus files. It **does not guarantee that an excerpt semantically supports the answer**. The second inference pass helps, but correlated errors remain possible. Probabilities are returned as model outputs, not as independently calibrated guarantees.

## Pipeline

1. Index five immutable public RFCs with BM25: JSON (8259), JWT (7519), HTTP semantics (9110), HTTP caching (9111), and WebSocket (6455).
2. Retrieve four candidates and consider the top two for context. Preserve paragraph boundaries where possible, with a bounded chunk size. The resulting index contains 1,119 chunks.
3. Pack evidence under the English Laya checkpoint's 512-token budget. Both backends use the same budget and passages in the final comparison. Drop complete lower-ranked chunks rather than silently truncate evidence.
4. Ask a Choice question with caller-defined options and `insufficient_evidence`.
5. For a non-abstaining decision, make a second narrow support/contradiction/unknown judgment against the same evidence. Failed verification produces abstention.
6. Return the selected option, raw model distributions, verification result, exact source excerpts, document checksums, source-span identifiers, and timing breakdown.

The source-span line numbers follow Python `splitlines`, including form-feed page breaks in RFC text; use the accompanying exact excerpt and document checksum when locating evidence, rather than assuming a browser's line numbering matches.

The API currently accepts a natural-language query and 2–8 short choice descriptions. It is not a complete replacement for TypeSafe's multi-question Choice/Noul/Score API. It has no free-form generation, multimodal support, or multilingual evaluation. Long option descriptions can still be shortened by the upstream Laya SDK's head-budget logic; this prototype is tested only with the short descriptions in the benchmark.

## Development and reproducibility

An initial 20-question development screen exposed cut-off passages. Before the held-out evaluation, chunking was changed to prefer full paragraphs. Early results are retained but should not be pooled with the final comparison: the first DiffusionGemma screen scored 18/20, the paragraph screen 19/20, and Laya's paragraph screen 15/20. Early screens did not use matched context budgets or excluded warmup consistently. No pipeline tuning followed the held-out results.

Laya checkpoint: `convaiinnovations/laya`, English root, revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. Runtime: Laya 0.3.4, Transformers 4.57.6, PyTorch 2.8.0. DiffusionGemma: `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, through the existing patched typed-classification service. Its model revision was not independently pinned in this experiment; reproduction requires a compatible structured service.

Raw results include the prompts, options, expected labels, model outputs, retrieved/used source identifiers, citations, and per-request timings. The repeat index identifies repeats of the same question. p50 uses the median; p95 uses nearest rank. The included tests check exact excerpt provenance, failed-verifier abstention, and invalid-input rejection. An HTTP smoke test returned the correct JSON NaN decision with citations.

## Limits and next steps

- Only 20 held-out questions; these results are a screening experiment, not a general accuracy estimate.
- Most answerable questions name the relevant technical terms directly. Hard paraphrases, cross-document reasoning, conflicting revisions, and large-scale retrieval are not tested.
- The original held-out choices placed the correct technical answer first; the separate label/order-swapped run checks that weakness explicitly.
- Prompt injection is instructed against but not adversarially validated. Do not ingest untrusted documents and assume the verifier makes them safe.
- The original baseline used no fine-tuning, semantic embeddings, GPU Laya comparison, or independent verifier. The later head-only fine-tuning experiment is documented above; GPU Laya, richer training data, and encoder adaptation remain untested.
- A single-request admission limit returns HTTP 429 under overlap; there is no tested production concurrency claim.

The original comparison favored DiffusionGemma for accuracy. The subsequent one-pass/context-packing changes reduced its measured median to roughly **260–271 ms** on the two validation workloads. These results support the prototype configuration, not a universal production accuracy or latency guarantee.

## Sources

- [Laya model card](https://huggingface.co/convaiinnovations/laya)
- [Laya source](https://github.com/NandhaKishorM/laya)
- [TypeSafe citation-checking pattern](https://docs.typesafe.ai/cookbooks/citation_check)
- [JSON RFC 8259](https://www.rfc-editor.org/rfc/rfc8259)
- [JWT RFC 7519](https://www.rfc-editor.org/rfc/rfc7519)
- [HTTP semantics RFC 9110](https://www.rfc-editor.org/rfc/rfc9110)
- [HTTP caching RFC 9111](https://www.rfc-editor.org/rfc/rfc9111)
- [WebSocket RFC 6455](https://www.rfc-editor.org/rfc/rfc6455)
- [URI RFC 3986](https://www.rfc-editor.org/rfc/rfc3986)
