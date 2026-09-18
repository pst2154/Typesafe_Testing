#!/usr/bin/env python3
"""Compare TypeSafe with DiffusionGemma structured reads.

The program prints sanitized JSON. Service locations and credentials are read
from environment variables and are never included in the output.
"""

from __future__ import annotations

import json
import math
import os
import ssl
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi
except ImportError:
    certifi = None

# Reuse the frozen cases without requiring the unrelated baseline at runtime.
for _name in ("TYPESAFE_API_URL", "SOL_API_URL", "TYPESAFE_MODEL", "SOL_MODEL"):
    os.environ.setdefault(_name, "unused")
from compare_typesafe_sol import TASKS  # noqa: E402


SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where() if certifi else None)
CHECKPOINT = os.environ.get("RESULT_PATH")


def post(url: str, payload: dict[str, Any], api_key: str | None = None) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600, context=SSL_CONTEXT) as response:
        body = json.load(response)
    return body, (time.perf_counter() - started) * 1000


def typesafe_result(kind: str, state: str) -> dict[str, Any]:
    body, latency = post(
        os.environ["TYPESAFE_API_URL"],
        {
            "state": state,
            "model": os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"),
            "questions": {"classification": TASKS[kind]["question"]},
        },
        os.environ["TYPESAFE_API_KEY"],
    )
    answer = body["answers"]["classification"]
    if kind == "noul":
        value = float(answer["noul"])
        prediction: bool | str | int = value >= 0.5
        certainty = abs(value - 0.5) * 2
    elif kind == "choice":
        prediction = answer["choice"]
        value = prediction
        certainty = float(answer["confidence"])
    else:
        value = float(answer["score"])
        prediction = min(3, max(0, math.floor(value + 0.5)))
        certainty = float(answer["confidence"])
    return {
        "model": body.get("model", os.environ.get("TYPESAFE_MODEL", "jev-1.13.0")),
        "prediction": prediction,
        "value": value,
        "certainty": certainty,
        "probabilities": answer.get("probabilities"),
        "latency_ms": latency,
    }


def diffusion_schema(kind: str) -> dict[str, Any]:
    question = TASKS[kind]["question"]
    schema_question: dict[str, Any] = {
        "id": "classification",
        "type": kind,
        "instructions": question["instructions"],
    }
    if kind == "noul":
        criteria = question["criteria"]
        schema_question["instructions"] += (
            f" Yes means: {criteria['true']} No means: {criteria['false']}"
        )
    elif kind == "choice":
        schema_question["options"] = [
            {"name": name, "description": description}
            for name, description in question["criteria"].items()
        ]
    else:
        schema_question["levels"] = question["criteria"]
    return {
        "questions": [schema_question],
        "samples": "auto",
        "auto_threshold": 0.1,
        "auto_max": 4,
        "steps": 1,
        "think": 0,
    }


def diffusion_result(kind: str, state: str) -> dict[str, Any]:
    body, latency = post(
        os.environ["DIFFUSIONGEMMA_URL"],
        {
            "model": "dgemma-structured",
            "seed": 42,
            "messages": [
                {"role": "system", "content": json.dumps(diffusion_schema(kind))},
                {"role": "user", "content": json.dumps({"state": state})},
            ],
        },
    )
    content = json.loads(body["choices"][0]["message"]["content"])
    answer = content["answers"]["classification"]
    if kind == "noul":
        value = float(answer["noul"])
        prediction: bool | str | int = value >= 0.5
    elif kind == "choice":
        prediction = answer["choice"]
        value = prediction
    else:
        prediction = TASKS[kind]["question"]["criteria"].index(answer["level"])
        value = float(answer["score"]) - 1
    return {
        "prediction": prediction,
        "value": value,
        "certainty": float(answer["confidence"]),
        "probabilities": answer.get("probabilities"),
        "samples": content["diagnostics"]["samples"]["n"],
        "engine_ms": content["diagnostics"]["timing"]["total_ms"],
        "latency_ms": latency,
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summary(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    clear = [row for row in rows if not row["ambiguous"]]
    ambiguous = [row for row in rows if row["ambiguous"]]
    latencies = [row[system]["latency_ms"] for row in rows]
    return {
        "correct": sum(row[system]["prediction"] == row["expected"] for row in clear),
        "scored": len(clear),
        "accuracy": sum(row[system]["prediction"] == row["expected"] for row in clear) / len(clear),
        "mean_ms": statistics.mean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "mean_clear_certainty": statistics.mean(row[system]["certainty"] for row in clear),
        "mean_ambiguous_certainty": (
            statistics.mean(row[system]["certainty"] for row in ambiguous)
            if ambiguous
            else None
        ),
    }


def result_document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(rows),
        "scored_case_count": sum(not row["ambiguous"] for row in rows),
        "configuration": {
            "typesafe_requested_model": os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"),
            "typesafe_reported_models": sorted(
                {row["typesafe"]["model"] for row in rows}
            ),
            "diffusion_steps": 1,
            "diffusion_samples": "adaptive, one to four",
            "diffusion_think_tokens": 0,
            "seed": 42,
        },
        "aggregate": {
            name: summary(rows, name) for name in ("typesafe", "diffusiongemma")
        },
        "by_primitive": {
            kind: {
                name: summary([row for row in rows if row["kind"] == kind], name)
                for name in ("typesafe", "diffusiongemma")
            }
            for kind in TASKS
        },
        "adversarial": {
            name: summary([row for row in rows if row["adversarial"]], name)
            for name in ("typesafe", "diffusiongemma")
        },
        "agreement": sum(
            row["typesafe"]["prediction"] == row["diffusiongemma"]["prediction"]
            for row in rows
        ) / len(rows),
        "rows": rows,
    }


def write_checkpoint(rows: list[dict[str, Any]]) -> None:
    if not CHECKPOINT:
        return
    path = Path(CHECKPOINT)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"rows": rows}, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    rows: list[dict[str, Any]] = []
    if CHECKPOINT and Path(CHECKPOINT).exists():
        rows = json.loads(Path(CHECKPOINT).read_text()).get("rows", [])
    completed = {(row["kind"], row["name"]) for row in rows}
    total = sum(len(task["cases"]) for task in TASKS.values())
    for kind, task in TASKS.items():
        for case in task["cases"]:
            if (kind, case.name) in completed:
                continue
            row = {
                "kind": kind,
                "name": case.name,
                "expected": case.expected,
                "ambiguous": case.ambiguous,
                "context_length": case.context_length,
                "pair_id": case.pair_id,
                "adversarial": case.adversarial,
                "char_count": len(case.state),
            }
            row["typesafe"] = typesafe_result(kind, case.state)
            row["diffusiongemma"] = diffusion_result(kind, case.state)
            rows.append(row)
            write_checkpoint(rows)
            print(f"completed {len(rows)}/{total}: {kind}/{case.name}", file=sys.stderr)

    result = result_document(rows)
    if CHECKPOINT:
        Path(CHECKPOINT).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
