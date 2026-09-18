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
- `DIFFUSIONGEMMA_URL` for the DiffusionGemma comparison

Run:

```bash
python3 compare_typesafe_sol.py
python3 compare_jev_small_router.py
python3 compare_typesafe_needle.py
python3 compare_typesafe_diffusiongemma.py
```

Set `COMPACT_RESULTS=1` to print aggregate and adversarial summaries without the
full per-case rows.

Set `RESULT_PATH` when running the DiffusionGemma comparison to checkpoint each
completed case and resume an interrupted run.

Routing comparisons:

- [Jev vs. Llama 3.2 1B](ROUTING_REPORT.md)
- [Jev vs. Qwen3.8-27B](QWEN_ROUTING_REPORT.md)
- [Detailed Jev vs. SOL and Qwen benchmark report](DETAILED_BENCHMARK_REPORT.md)
- [TypeSafe vs. Needle 3 classification benchmark](NEEDLE_TYPESAFE_REPORT.md)
- [TypeSafe vs. DiffusionGemma typed-decision benchmark](DIFFUSIONGEMMA_TYPESAFE_REPORT.md)

The repository contains no credentials, endpoint values, internal paths, or raw
production data.
