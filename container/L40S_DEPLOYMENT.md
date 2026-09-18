# DiffusionGemma FP8 on one L40S — experimental deployment

Verified September 18, 2026. The service starts and completes typed classification
requests on one NVIDIA L40S. **Multi-question semantic correctness is not validated:**
12 identical positive Noul questions returned valid responses but only 4 scored
above 0.5. Do not treat this deployment as a production-equivalent replacement.

## Tested configuration

- GPU: one NVIDIA L40S, 46,068 MiB reported memory; driver 610.57.04.
- Runtime: vLLM PR 57250, commit `d2c2b5422d6e1b62fc9af68e16dc26a975d2aa23`.
- Local runtime image: `vllm-diffusiongemma-structured:pr57250-d2c2b54`.
- Checkpoint: [RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic).
- Model revision: `3b3dae4697494da5a290e9c0461954449e76c4f5`.
- FP8 weights and dynamic activations; BF16 residual computation and default KV cache.
- Text-only; canvas 32; TP=1, PP=1; context 16,384; 8 sequences;
  batch token budget 4,096; GPU memory utilization 0.90.
- Triton attention, automatic MoE selection (selected Triton FP8),
  Cutlass FP8 linear kernels; prefix caching and async scheduling enabled.
- Classification: samples=1, steps=1, think=0.

Model loading reported 24.75 GiB, KV cache availability 13.63 GiB, and an observed
post-start total GPU allocation of 41,609 MiB. Runtime warned that no tuned L40S
MoE configuration was available; performance may be suboptimal.

## Reproduce the service

Use the structured runtime and fixed adapter sources from
[Typesafe_Testing](https://github.com/pst2154/Typesafe_Testing).
The measured trial used the local runtime image above with the fixed
`container/structured_server.py` mounted over its original adapter. It did not
validate a fresh GHCR pull. Obtain/build that pinned runtime first using the
repository's runtime-container instructions. Run the following from the checkout:

```sh
docker volume create dgemma-l40s-models
docker volume create dgemma-l40s-cache
docker run -d --name dgemma-l40s --gpus 'device=0' \
  --network host --shm-size 8g \
  -v dgemma-l40s-models:/cache/huggingface \
  -v dgemma-l40s-cache:/tmp/.cache \
  -v "$PWD/container/structured_server.py:/opt/vllm/examples/features/diffusion_reads/structured_server.py:ro" \
  -e HOME=/tmp -e HF_HOME=/cache/huggingface -e HF_XET_HIGH_PERFORMANCE=1 \
  -e MODEL_ID=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
  -e MOE_BACKEND=auto -e UPSTREAM_PORT=8200 -e STRUCTURED_PORT=8110 \
  vllm-diffusiongemma-structured:pr57250-d2c2b54 \
  --revision 3b3dae4697494da5a290e9c0461954449e76c4f5 \
  --max-model-len 16384 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9
```

Use a GPU assigned to your allocation, not an arbitrary device on a shared host.
The runtime entrypoint already supplies canvas 32, language-model-only, Triton
attention, prefix caching and async scheduling. The later sequence-limit argument
overrides its built-in 32; vLLM logs a duplicate-argument warning but confirms 8.
Named-volume commands above adapt the trial's bind mounts for portability.

After `curl -f http://127.0.0.1:8110/health` succeeds, launch the UI/API gateway:

```sh
DIFFUSION_URLS=http://127.0.0.1:8110/v1/chat/completions \
RAW_URLS=http://127.0.0.1:8200/v1/chat/completions \
GATEWAY_PORT=8014 GATEWAY_MODEL_NAME=diffusiongemma-fp8-l40s \
DIFFUSION_SAMPLES=1 python3 container/gateway.py
```

Keep this process running under your service supervisor. The gateway serves the
adjacent `explorer.html` at `/` and TypeSafe-compatible requests at `/v1/systemone`.
Its health endpoint alone does not prove the model is ready. No API credentials
are required in this isolated trial; set `GATEWAY_API_KEY` and use network access
controls/TLS before wider exposure. Raw and structured ports also need protection.

```sh
curl http://YOUR_HOST:8014/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","state":"Please refund my duplicate invoice.","questions":{"route":{"type":"choice","instructions":"Which team handles this request?","criteria":{"billing":"Payments, invoices, refunds","technical":"Software bugs, infrastructure outages"}}}}'
```

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
