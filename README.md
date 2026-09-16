# TypeSafe classification testing

Reproducible comparisons of TypeSafe typed classifications and SOL structured
classifications. The benchmark covers binary, multiclass, ordinal, ambiguous,
short-context, long-context, and adversarial cases.

The benchmark requires these environment variables:

- `TYPESAFE_API_KEY`
- `TYPESAFE_API_URL`
- `TYPESAFE_MODEL`
- `INFERENCE_API_KEY` or `NVIDIA_API_KEY`
- `SOL_API_URL`
- `SOL_MODEL`

Run:

```bash
python3 compare_typesafe_sol.py
python3 compare_jev_small_router.py
```

Set `COMPACT_RESULTS=1` to print aggregate and adversarial summaries without the
full per-case rows.

The routing comparison is summarized in [ROUTING_REPORT.md](ROUTING_REPORT.md).

The repository contains no credentials, endpoint values, internal paths, or raw
production data.
