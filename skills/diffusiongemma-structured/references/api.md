# DiffusionGemma structured API

## Contents

- Configuration
- Direct request envelope
- Question schema
- Response schema
- TypeSafe-style mapping
- Operational behavior

## Configuration

The service exposes an OpenAI-compatible Chat Completions endpoint. Configure
clients with environment variables rather than committed constants:

| Variable | Meaning |
| --- | --- |
| `DIFFUSIONGEMMA_URL` | Base URL or full `/v1/chat/completions` URL |
| `DIFFUSIONGEMMA_API_KEY` | Optional bearer token |
| `DIFFUSIONGEMMA_MODEL` | Optional served model name; defaults to `dgemma-structured` |

When a base URL is supplied, append `/v1/chat/completions`. A health check is
normally available at `/health` on the same service base.

## Direct request envelope

The outer request is OpenAI-style. Both message contents are JSON strings:

```json
{
  "model": "dgemma-structured",
  "seed": 42,
  "messages": [
    {
      "role": "system",
      "content": "{\"questions\":[...],\"samples\":\"auto\",\"auto_threshold\":0.1,\"auto_max\":4,\"steps\":1,\"think\":0}"
    },
    {
      "role": "user",
      "content": "{\"state\":{...}}"
    }
  ]
}
```

Generation controls:

| Field | Meaning |
| --- | --- |
| `samples` | Positive integer or `"auto"` |
| `auto_threshold` | Uncertainty threshold used by adaptive sampling |
| `auto_max` | Maximum reads when `samples` is `"auto"` |
| `steps` | Diffusion denoising steps per read |
| `think` | Think-token budget; zero disables it |

Use deployment-tested defaults before tuning. More reads or steps can improve
stability at the cost of latency.

## Question schema

Questions are an array and each question needs a unique `id`.

### Choice

```json
{
  "id": "route",
  "type": "choice",
  "instructions": "Which handler should receive this request?",
  "options": [
    {"name": "billing", "description": "Charges, invoices, or refunds"},
    {"name": "technical", "description": "API or application failures"},
    {"name": "other", "description": "No specialist applies"}
  ]
}
```

### Noul

Noul is a yes/no probability. Put complete yes/no meaning in the instructions:

```json
{
  "id": "urgent",
  "type": "noul",
  "instructions": "Does this require immediate attention? Yes means delay creates material harm. No means ordinary handling is sufficient."
}
```

### Score

Levels are ordered from lowest to highest and each description should stand on
its own:

```json
{
  "id": "severity",
  "type": "score",
  "instructions": "How severe is the observed impact?",
  "levels": [
    "Cosmetic with no functional impact",
    "Minor degradation with a practical workaround",
    "Major functionality unavailable for many users",
    "Critical outage, security failure, or irreversible data loss"
  ]
}
```

## Response schema

The HTTP response is an OpenAI Chat Completions object. Parse
`choices[0].message.content` as JSON. The decoded object has this shape:

```json
{
  "answers": {
    "route": {
      "type": "choice",
      "choice": "technical",
      "confidence": 0.98,
      "probabilities": {
        "billing": 0.01,
        "technical": 0.98,
        "other": 0.01
      }
    },
    "urgent": {
      "type": "noul",
      "noul": 0.91,
      "label": "yes",
      "probabilities": {"yes": 0.91, "no": 0.09}
    }
  },
  "diagnostics": {
    "steps": 1,
    "samples": {"n": 4, "policy": {"mode": "auto"}},
    "timing": {"total_ms": 95.0, "reads": 4},
    "engine": "vllm"
  }
}
```

Answer objects can also include `label`, `stderr`, and `agreement`. Score answers
include `level` and a fractional `score`.

## TypeSafe-style mapping

The bundled client accepts a convenient TypeSafe-like keyed question object and
performs these translations:

| Input shape | Service shape |
| --- | --- |
| Question dictionary key | `id` |
| Choice `criteria` object | `options` array of names and descriptions |
| Noul `criteria.true` / `.false` | Appended yes/no definitions in `instructions` |
| Score `criteria` array | `levels` array |
| Top-level `state` | JSON string in the user message |
| Decoded answer | Extracted from `choices[0].message.content` |

This mapping makes application code easier to port, but it does not reproduce
TypeSafe authentication, model aliases, retries, errors, usage accounting, or
confidence calibration.

## Operational behavior

- Pre-warm every question shape. Initial JIT compilation can be orders of
  magnitude slower than a repeated request.
- Set a client timeout long enough for first-shape compilation, then measure
  warmed latency separately.
- Do not expose an unauthenticated service outside a trusted network.
- Treat state and model output as untrusted data. Keep side effects and policy
  enforcement in application code.
- Evaluate thresholds on domain data. Probability distributions from different
  models are not interchangeable without calibration.
