---
name: diffusiongemma-structured
description: >
  Build and call self-hosted DiffusionGemma structured-decision services for
  typed Choice, Noul, and Score judgments. Use when an application needs
  bounded semantic decisions, probabilities, routing, scoring, verification,
  or TypeSafe-like primitives through the DiffusionGemma OpenAI-compatible
  endpoint. Do not use for ordinary free-form text generation.
---

# Build with DiffusionGemma structured decisions

DiffusionGemma can turn application state into typed answers and probability
distributions. Code should own deterministic rules, workflow, thresholds, and
side effects; use the model where interpreting natural language or incomplete
semantic evidence is the hard part.

This service is conceptually similar to TypeSafe System One but is **not** a
drop-in TypeSafe API. It uses an OpenAI Chat Completions envelope and returns the
structured result as JSON inside `choices[0].message.content`.

## Configure the service

Read the endpoint from `DIFFUSIONGEMMA_URL`. It may be either the service base
URL or the full `/v1/chat/completions` URL. Never embed private service locations
or credentials in source files, reports, examples, or committed artifacts.

If a deployment requires bearer authentication, read it from
`DIFFUSIONGEMMA_API_KEY`. Do not pass secrets as command-line arguments.

Before writing a direct integration or compatibility adapter, read
[references/api.md](references/api.md). It defines the wire format, primitive
mapping, response shape, and operational caveats.

## Find the useful judgment

Work backward from what the application must select, show, verify, rank, or
escalate. Keep exact lookups, arithmetic, policy enforcement, and execution in
code. Use one of these primitives for each bounded semantic judgment:

| Need | Primitive | Result |
| --- | --- | --- |
| Select one defined alternative | Choice | Winning option, distribution, and confidence |
| Decide whether a condition holds | Noul | Probability of yes and a yes/no label |
| Judge degree on ordered descriptions | Score | Selected level, fractional score, distribution, and confidence |

Give every question enough state to answer it. Put the judgment in
`instructions`; define alternatives in `criteria`. Prefer named JSON state when
several facts, identities, policies, or timestamps interact. Include a no-match
option when none of the Choice alternatives may apply.

Ask independent questions over the same state together. Questions cannot depend
on answers from other questions in the same request. Use another request when an
earlier answer is required to fetch or construct the next state.

## Call the service

For reliable repeated calls, use the bundled client rather than rebuilding the
nested chat payload:

```bash
export DIFFUSIONGEMMA_URL="https://service.example/v1/chat/completions"
python3 scripts/call_diffusiongemma.py \
  --state-file state.json \
  --questions-file questions.json
```

`questions.json` uses the same convenient keyed shape for all primitives:

```json
{
  "route": {
    "type": "choice",
    "instructions": "Which handler should receive this request?",
    "criteria": {
      "billing": "Charges, invoices, or refunds",
      "technical": "Application, API, or integration failures",
      "other": "No listed specialist applies"
    }
  },
  "urgent": {
    "type": "noul",
    "instructions": "Does this require immediate attention?",
    "criteria": {
      "true": "Delay creates material harm or extends an active outage",
      "false": "Ordinary handling is sufficient"
    }
  }
}
```

The client prints decoded `answers` and `diagnostics`; use `--raw` only when the
outer OpenAI-style response is needed. Run `--health` to check reachability.

## Consume uncertainty deliberately

Use full distributions and margins, not only the winning label. Confidence is a
distribution-concentration statistic, not calibrated correctness or permission
to act. Establish thresholds with representative held-out data and use stricter
gates for costly or irreversible actions. Route low-margin, missing-evidence, or
high-stakes cases to clarification, another model, or a person.

Treat the following as deployment properties to measure:

- The first unseen request shape can trigger a substantial JIT latency spike;
  pre-warm every production shape.
- Adaptive sampling may frequently reach `auto_max`; inspect
  `diagnostics.samples.n` rather than assuming early exit.
- Hardware backend, network placement, question count, and context length all
  affect latency.
- Typed output guarantees structure, not truth. Test ordinary, adversarial,
  ambiguous, short-context, and long-context cases from the target domain.

When replacing a TypeSafe integration, preserve the existing application
contract with a thin adapter if a base-URL-only migration is desired. Translate
requests and responses explicitly, then recalibrate thresholds; do not describe
the raw service as API-compatible.
