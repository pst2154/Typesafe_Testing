# DiffusionGemma decision container

Image: `ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54`

This community image packages vLLM PR #57250 at commit
`d2c2b5422d6e1b62fc9af68e16dc26a975d2aa23`, the structured decision server,
a TypeSafe-style HTTP adapter, and the Decision Lab explorer. It is not an
official NVIDIA image or the proprietary Jev model.

## Run on an H100

Requires Linux x86-64, Docker with NVIDIA Container Toolkit, a CUDA 13-compatible
driver, and enough GPU memory for the NVFP4 model and its KV cache. The runtime
has been exercised on H100 80 GB and H100 NVL. The newly combined launcher was
checked without starting a second full model on the occupied test GPU.

```bash
docker pull ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54
docker volume create diffusiongemma-model-cache
docker run --rm --name diffusiongemma --gpus '"device=0"' \
  --ipc=host \
  -p 8012:8012 \
  -e HF_HOME=/model-cache \
  -e MOE_BACKEND=marlin \
  -v diffusiongemma-model-cache:/model-cache \
  ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54
```

The first launch downloads `nvidia/diffusiongemma-26B-A4B-it-NVFP4` from Hugging
Face and initializes the model. Weights are not embedded in the image. The
named volume retains downloads between runs. If model access requires a token,
provide `HF_TOKEN` through your runtime environment, not a Dockerfile.

Open `http://localhost:8012/` for Decision Lab after initialization. The public
port provides:

- `POST /v1/systemone`: keyed Choice, Noul, and Score questions.
- `POST /v1/chat/completions`: native structured decisions.
- `POST /v1/raw/chat/completions`: raw model chat completions.
- `GET /health`: gateway status; a live inference request is the stronger check.

```bash
curl http://localhost:8012/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"dgemma-structured","state":"Customer payouts have failed for three days.","questions":{"route":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments, payouts, invoices, or refunds","technical":"Application bugs or infrastructure outages"}}}}'
```

The example exposes an unauthenticated service. For shared deployments, provide
your own network access controls or authenticated reverse proxy. The optional
`GATEWAY_API_KEY` protects API calls with a bearer token; the bundled browser
does not currently have a token-entry control. Internal model ports need not be
published. The gateway offers a supported subset of TypeSafe-style requests,
not a guarantee of complete TypeSafe API parity.

## Build

`Dockerfile` adds the application to the saved runtime image. To reconstruct the
runtime from source in this directory:

```bash
git clone https://github.com/vllm-project/vllm.git vllm
git -C vllm checkout d2c2b5422d6e1b62fc9af68e16dc26a975d2aa23
docker build -f Dockerfile.runtime \
  -t vllm-diffusiongemma-structured:pr57250-d2c2b54 .
docker build -t ghcr.io/pst2154/diffusiongemma-structured:pr57250-d2c2b54 .
```

These are the source recipe and pinned vLLM revision; dependency resolution uses
upstream package repositories, so a source rebuild is not guaranteed bitwise
identical to the published image. Prefer a published image digest for exact
deployment pinning.

## Terms

See [NOTICE](NOTICE). The NVIDIA container license is retained inside the image
at `/NGC-DL-CONTAINER-LICENSE`; vLLM and other components retain their own
licenses. Use and redistribution remain subject to those terms. The model
publisher's separate terms apply when downloading weights.
