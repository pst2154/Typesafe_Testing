# NanoJev versus DiffusionGemma: matched decision accuracy

Date: September 18, 2026

## Result

DiffusionGemma answered **71 of 72 cases correctly (98.6%)**; NanoJev answered **55 of 72 (76.4%)**. The difference was **22.2 percentage points**, or 16 additional correct answers for diffusion. Both completed every request successfully in the final run.

This is a small, deliberately constructed diagnostic of general decision tasks. It is not a broad benchmark, a game evaluation, or a comparison with the proprietary Jev service.

## Models and execution

| Component | NanoJev | Diffusion |
|---|---|---|
| Model | Public NanoJev root checkpoint, Qwen3-0.6B backbone | NVIDIA DiffusionGemma 26B-A4B IT NVFP4 |
| Interface | Native decision heads | Existing structured-decision gateway |
| Numeric configuration | BF16 weight storage and BF16 autocast | Existing NVFP4 deployment |
| Hardware | Original H100 80 GB, shared with diffusion | Same original H100 80 GB |
| Request scheduling | Sequential, alternating model order by case | Sequential, alternating model order by case |

NanoJev source revision: `71a513bb0163b5634467842b523ee0c0ed6fb1c7`. Its root checkpoint is a navigation-trained baseline, not a model tuned on this diagnostic. Its runtime uses PyTorch 2.13.0+cu130, Transformers 5.17.0, and safetensors 0.8.0; upstream records PyTorch 2.14.0.

Two local serving changes were needed: BF16 parameter storage to fit alongside diffusion, and threaded HTTP handling with a lock around inference to avoid browser connections blocking the server. The former changes numerical precision from upstream's FP32 parameter storage; no FP32 accuracy control was run. Neither change trains or tunes the checkpoint.

The four-H100 deployment was not used or modified.

## Dataset and scoring

The dataset was defined before the first comparison: 24 base cases, each evaluated in three forms, yielding 72 inputs per model and 144 requests in the final run.

- **Choice:** eight customer requests across billing, technical, account, and sales, with two base requests per category.
- **Boolean:** eight age/membership cases under the explicit rule `age >= 18 AND membership active`; three positive and five negative base cases.
- **Score:** eight records with an explicit count of affected regions from zero through three, two records per level. The question asks for the exactly matching level.

The input variants are plain short text, a prefix of repeated irrelevant archive notes followed by the current evidence, and a suffix instructing the model to ignore its rules and output a specified answer. Instructions tell both models to ignore commands embedded in evidence. The distractor form is longer within NanoJev's 512-token candidate-path limit; it is **not** a long-context benchmark.

Both systems received identical state, question instructions, and category/level descriptions. Only transport wrapping and the Boolean type name (`boolean` versus `noul`) differed. Their internal encoders and inference mechanisms differ naturally.

Choice is scored by the selected category; Boolean by a 0.5 threshold; Score by the highest-probability level. Score is not assessed by rounding its expected value. Correct answers follow the supplied definitions or explicit numerical rule; no LLM judge was used.

Frozen dataset SHA-256, calculated over the indented JSON string before its final newline:
`76caf52b188d7e932d13f9deedd269f2c86348f944471fef6d3f1e8d904ecc40`

## Accuracy by task

| Task | NanoJev | Diffusion |
|---|---:|---:|
| Choice | 18/24 — 75.0% | 24/24 — 100% |
| Boolean | 18/24 — 75.0% | 24/24 — 100% |
| Score, highest-probability level | 19/24 — 79.2% | 23/24 — 95.8% |
| **Overall** | **55/72 — 76.4%** | **71/72 — 98.6%** |

## Accuracy by input variant

| Variant | NanoJev | Diffusion |
|---|---:|---:|
| Short | 17/24 — 70.8% | 24/24 — 100% |
| Distractor prefix | 20/24 — 83.3% | 24/24 — 100% |
| Injected instruction | 18/24 — 75.0% | 23/24 — 95.8% |

NanoJev improved on this particular distractor set. That does not establish that longer inputs improve it: repeated padding can change representations, and these variants reuse the same underlying facts. Likewise, injection-set accuracy is not an attack success rate. The injected target coincides with the correct answer in some cases, and some models were already wrong on corresponding plain inputs.

## Representative mistakes

| Evidence / correct answer | NanoJev behavior | Diffusion behavior |
|---|---|---|
| HTTP 503 on every request / technical | Plain input: billing 56.0%, technical 3.1%; injected input: sales | Correct on all three variants |
| New teammate needs access permissions / account | Plain input: billing; injected input: sales 61.4% | Correct on all three variants |
| Age 17, active membership / ineligible | Predicted eligible with 79.6% probability on plain input; wrong on all variants | Correct on all variants |
| Exactly three affected regions / level 3 | Several inputs favored level 1 despite explicit count | One injected case favored level 2 at 49.2%, versus level 3 at 46.6% |

The Boolean errors suggest trouble enforcing the numeric eligibility threshold in these examples. The score errors show that accepting the Score API and producing a valid distribution do not guarantee faithful use of explicit counts.

## Probability quality and latency

| Metric | NanoJev | Diffusion |
|---|---:|---:|
| Boolean Brier score, lower is better | 0.19023 | 0.0000679 |
| Mean absolute error of expected Score, 0–3 scale | 0.4753 | 0.1264 |
| Median end-to-end request time | 289 ms | 247 ms |
| Failed requests in final run | 0/72 | 0/72 |

Brier score here is the mean squared error between `P(true)` and the binary label. These probability metrics describe this dataset only; they do not certify calibration in deployment.

Latency includes client/network overhead and gateway work. The models shared a GPU, their software stacks differ, and the NanoJev dashboard was checked during the run. These measurements are observational and should not be used to claim an isolated speed advantage. Model parameter count alone did not predict latency in this configuration.

## Run accounting and limits

The first attempt encountered a blocked NanoJev HTTP server. A subsequent preliminary run started before NanoJev finished reloading and contained five connection refusals. Those runs are excluded from the headline result. The final run repeated all 72 cases for both models after serving was functional and had no request errors. The dataset and model weights were not changed in response to preliminary outcomes.

Diffusion scored 70/72 in the preliminary run and 71/72 in the final run, with a score decision near the probability boundary changing. A repeated-trial stability study was not performed.

The 72 inputs are correlated variations of 24 base cases; score cases repeat region counts with different incident IDs. No statistical significance or population-wide accuracy claim is made. Task coverage is narrow, category order is fixed, and paraphrase robustness, long contexts, domain-specific training, and game performance were not evaluated. NanoJev's specialized training and reduced weight precision limit conclusions about its architecture or other checkpoints.

For the tested general decision questions, the current diffusion deployment is the stronger choice. NanoJev is a working small-model baseline, but these results do not support treating this checkpoint as an equally accurate replacement.

## Reproduction artifacts

The accompanying [comparison script](compare_nanojev_diffusion.py) creates the frozen cases and evaluates both services. The verified artifacts contain [cases](results/nanojev_diffusion/cases.json), [individual predictions and probabilities](results/nanojev_diffusion/results.jsonl), and [aggregate results](results/nanojev_diffusion/summary.json). Service addresses are supplied at runtime and intentionally omitted from this report.
