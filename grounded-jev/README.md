# Grounded typed-decision prototype

Retrieve public technical documentation, answer a caller-defined Choice question, and return source excerpts or abstain. The default `diverse` variant makes one evidence-first inference call; `--variant baseline` retains the original two-pass verification path. See [measured results and limitations](REPORT.md).

Requires Python 3.11. Use a clean virtual environment; do not inherit unrelated system packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
USE_TF=0 python prepare.py
python -m unittest discover -p 'test_*.py'
```

Run Laya locally (Apple MPS is the default; set `LAYA_DEVICE=cpu` or `cuda` for a different device):

```bash
USE_TF=0 python server.py --backend laya
```

Or use a compatible DiffusionGemma typed-classification service:

```bash
export DIFFUSION_URL='https://YOUR-SERVICE/v1/systemone'
# Set DIFFUSION_API_KEY if the service requires authentication.
USE_TF=0 python server.py --backend diffusiongemma
```

Both use the downloaded Laya tokenizer for identical context budgeting. Model weights load only for the Laya backend.

```bash
curl http://127.0.0.1:8790/v1/grounded/decision \
  -H 'Content-Type: application/json' \
  -d '{"query":"Does JSON allow NaN as a number?","criteria":{"allowed":"NaN is allowed","forbidden":"NaN is not permitted in JSON"}}'
```

The response includes `choice`, `answer`, `abstained`, `citations`, `verification`, and `timing_ms`. This is a grounded decision API, not free-form chat or a drop-in implementation of every TypeSafe primitive. Only accept citations after considering the report's semantic-grounding limitations.

The default listener is local-only. Non-loopback binding requires `GROUNDED_API_KEY`; clients then supply `Authorization: Bearer ...`. Use a TLS reverse proxy before exposing it publicly. Overlapping requests receive HTTP 429 rather than an unbounded queue.

Re-run the held-out comparison:

```bash
USE_TF=0 python evaluate.py --backend laya --suite heldout --repeat 3 --output laya-heldout.json
USE_TF=0 python evaluate.py --backend diffusiongemma --suite heldout --repeat 3 --output diffusion-heldout.json
USE_TF=0 python evaluate.py --backend laya --suite heldout --swap-labels --output laya-swapped.json
USE_TF=0 python evaluate.py --backend diffusiongemma --suite heldout --swap-labels --output diffusion-swapped.json
```

Each run warms separately, then makes real inference requests. Results are written after each request. The corpus manifest records public source URLs and SHA-256 hashes; original RFC copyright notices remain in the downloaded documents.

Evaluate the optimized variant and the separate URI document:

```bash
python external_cases.py
USE_TF=0 python evaluate.py --backend diffusiongemma --variant diverse --suite heldout --repeat 3 --output tuned-regression.json
USE_TF=0 python evaluate.py --backend diffusiongemma --variant diverse --suite external --repeat 3 --output tuned-uri.json
```

Reproduce the head-only training experiment:

```bash
USE_TF=0 python finetune.py
USE_TF=0 python finetune.py --reuse-features --learning-rate 0.0001 --epochs 100 --output finetuned-head-v2
LAYA_SCORER=finetuned-head-v2/scorer.safetensors USE_TF=0 python evaluate.py --backend laya --variant plain --suite external --repeat 3 --output tuned-laya-uri.json
```

The supplied experimental adapter **regressed on real-document evaluation** and is not enabled by default. Synthetic training data, validation-selected weights and results are provided for reproducibility, not as a recommended model release.
