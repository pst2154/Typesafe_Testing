# Deploy DiffusionGemma FP8 on one L40S

Pull the container, launch the FP8 model, and open the bundled UI or call the API.

## Configuration

- GPU: one NVIDIA L40S, 46,068 MiB reported memory; driver 610.57.04.
- Runtime: vLLM PR 57250, commit `d2c2b5422d6e1b62fc9af68e16dc26a975d2aa23`.
- Container: `ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54-fix1`.
- Checkpoint: [RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic](https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic).
- Model revision: `3b3dae4697494da5a290e9c0461954449e76c4f5`.
- FP8 weights and dynamic activations; BF16 residual computation and default KV cache.
- Text-only; canvas 32; TP=1, PP=1; context 16,384; 8 sequences;
  batch token budget 4,096; GPU memory utilization 0.90.
- Triton attention, automatic MoE selection (selected Triton FP8),
  Cutlass FP8 linear kernels; prefix caching and async scheduling enabled.
- Classification: samples=1, steps=1, think=0.

## Pull and run

Pull the published all-in-one GHCR image. It includes the patched runtime,
fixed structured adapter, TypeSafe-compatible gateway, and Decision Lab UI.
**No repository checkout, source build, or separately launched gateway is needed.**

Requires Linux x86-64, Docker with NVIDIA Container Toolkit, a CUDA 13-compatible
driver, one L40S, and sufficient disk space for the image and model cache.

```sh
docker pull ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54-fix1
docker volume create dgemma-l40s-models
docker volume create dgemma-l40s-cache
docker run -d --name dgemma-l40s --gpus 'device=0' \
  --shm-size 8g -p 8014:8012 \
  -v dgemma-l40s-models:/cache/huggingface \
  -v dgemma-l40s-cache:/tmp/.cache \
  -e HOME=/tmp -e HF_HOME=/cache/huggingface -e HF_XET_HIGH_PERFORMANCE=1 \
  -e MODEL_ID=RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
  -e MOE_BACKEND=auto -e DIFFUSION_SAMPLES=1 \
  -e GATEWAY_MODEL_NAME=diffusiongemma-fp8-l40s \
  ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54-fix1 \
  --revision 3b3dae4697494da5a290e9c0461954449e76c4f5 \
  --max-model-len 16384 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.9
```

Use a GPU assigned to your allocation, not an arbitrary device on a shared host.
The runtime entrypoint already supplies canvas 32, language-model-only, Triton
attention, prefix caching and async scheduling. The later sequence-limit argument
overrides its built-in 32; vLLM logs a duplicate-argument warning but confirms 8.
`MOE_BACKEND=auto` overrides the image's NVFP4-oriented Marlin default.
The model weights are downloaded at first startup, not embedded in the image.
Named volumes preserve model downloads and compilation caches between runs.
If required, supply `HF_TOKEN` through the environment, never in a published file.

Watch startup and verify the service:

```sh
docker logs -f dgemma-l40s
# In another terminal, after initialization:
curl -f http://localhost:8014/health
```

Open **http://localhost:8014/** for the UI. The same port serves
`POST /v1/systemone` and `POST /v1/chat/completions`. For another computer, replace
`localhost` with the deployment host. The launcher starts the model, structured
adapter, and UI/API gateway together. Internal model ports are not published.
Use the real request below as well as the health check to verify inference.

This command exposes an unauthenticated service on host port 8014. Use network
access controls and an authenticated TLS reverse proxy for shared deployments.
For local-only access, use `-p 127.0.0.1:8014:8012`. The optional
`GATEWAY_API_KEY` protects API calls, but the bundled UI has no token-entry control.

```sh
curl http://YOUR_HOST:8014/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"diffusiongemma","state":"Please refund my duplicate invoice.","questions":{"route":{"type":"choice","instructions":"Which team handles this request?","criteria":{"billing":"Payments, invoices, refunds","technical":"Software bugs, infrastructure outages"}}}}'
```
