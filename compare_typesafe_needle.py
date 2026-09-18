#!/usr/bin/env python3
"""Compare TypeSafe/Jev decision primitives with Needle 3 tool selection.

Service locations and credentials come from environment variables and are never
written to the result. The frozen case definitions are shared with the original
TypeSafe/SOL benchmark.
"""

from __future__ import annotations

import json
import math
import os
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    import certifi
except ImportError:
    certifi = None

for _name in ("TYPESAFE_API_URL", "SOL_API_URL", "TYPESAFE_MODEL", "SOL_MODEL"):
    os.environ.setdefault(_name, "unused")
from compare_typesafe_sol import TASKS  # noqa: E402


SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where() if certifi else None)
CHECKPOINT = os.environ.get("RESULT_PATH")


def post(
    url: str,
    payload: dict[str, Any],
    api_key: str | None = None,
    timeout: float = 600,
) -> tuple[dict[str, Any], float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=SSL_CONTEXT
        ) as response:
            return json.load(response), (time.perf_counter() - started) * 1000
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def typesafe_result(kind: str, state: str) -> dict[str, Any]:
    body, latency = post(
        os.environ["TYPESAFE_API_URL"],
        {
            "state": state,
            "model": os.environ.get("TYPESAFE_MODEL", "jev-1.13.0"),
            "questions": {"classification": TASKS[kind]["question"]},
        },
        os.environ["TYPESAFE_API_KEY"],
        timeout=120,
    )
    answer = body["answers"]["classification"]
    if kind == "noul":
        value = float(answer["noul"])
        prediction: bool | str | int = value >= 0.5
        certainty = abs(value - 0.5) * 2
    elif kind == "choice":
        value = answer["choice"]
        prediction = value
        certainty = float(answer["confidence"])
    else:
        value = float(answer["score"])
        prediction = min(3, max(0, math.floor(value + 0.5)))
        certainty = float(answer["confidence"])
    return {
        "prediction": prediction,
        "value": value,
        "certainty": certainty,
        "probabilities": answer.get("probabilities"),
        "latency_ms": latency,
    }


def needle_tools(kind: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question = TASKS[kind]["question"]
    if kind == "noul":
        criteria = question.get("criteria", {})
        # Needle's native binary shape is an action or an empty call, not two
        # competing yes/no tools. A call means true; refusal means false.
        options: list[tuple[str, Any, str]] = [
            (
                "refund_requested",
                True,
                str(criteria.get("true", "The condition is true.")),
            ),
        ]
    elif kind == "choice":
        options = [
            (f"route_{name}", name, str(description or name))
            for name, description in question["criteria"].items()
        ]
    else:
        severity_names = ("cosmetic", "minor", "major", "critical")
        options = [
            (f"severity_{severity_names[index]}", index, str(description))
            for index, description in enumerate(question["criteria"])
        ]

    tools = []
    predictions = {}
    for tool_suffix, prediction, meaning in options:
        name = tool_suffix
        predictions[name] = prediction
        if kind == "noul":
            description = (
                f"Use this tool only when the answer to "
                f"'{question['instructions']}' is yes: {meaning}"
            )
        elif kind == "choice":
            description = f"Route to {prediction}: {meaning}"
        else:
            description = f"Classify incident severity at this level: {meaning}"
        tools.append(
            {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            }
        )
    return tools, predictions


def needle_result(kind: str, state: str) -> dict[str, Any]:
    tools, predictions = needle_tools(kind)
    body, latency = post(
        os.environ["NEEDLE_URL"].rstrip("/") + "/v1/tools",
        {"query": state, "tools": tools},
    )
    calls = body.get("function_calls") or []
    suppressed = body.get("suppressed_calls") or []
    candidates = calls or suppressed
    tool_name = candidates[0]["name"] if candidates else None
    prediction = predictions.get(tool_name)
    if kind == "noul" and not candidates:
        prediction = False
    return {
        "prediction": prediction,
        "confidence": body.get("confidence"),
        "acted": bool(calls),
        "suppressed": bool(suppressed),
        "refused": not candidates,
        "latency_ms": latency,
        "prefill_tps": body.get("prefill_tps"),
        "decode_tps": body.get("decode_tps"),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def summarize(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    clear = [row for row in rows if not row["ambiguous"]]
    valid = [row for row in rows if "error" not in row[system]]
    scored_valid = [row for row in clear if "error" not in row[system]]
    latencies = [row[system]["latency_ms"] for row in valid]
    result = {
        "correct": sum(
            row[system].get("prediction") == row["expected"] for row in scored_valid
        ),
        "scored": len(clear),
        "valid_scored": len(scored_valid),
        "errors": len(rows) - len(valid),
        "mean_ms": statistics.mean(latencies) if latencies else None,
        "p50_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": percentile(latencies, 0.95),
    }
    if system == "needle":
        result.update(
            {
                "acted": sum(row[system].get("acted", False) for row in valid),
                "suppressed": sum(
                    row[system].get("suppressed", False) for row in valid
                ),
                "refused": sum(row[system].get("refused", False) for row in valid),
            }
        )
    return result


def result_document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case_count": len(rows),
        "aggregate": {
            system: summarize(rows, system) for system in ("typesafe", "needle")
        },
        "by_primitive": {
            kind: {
                system: summarize(
                    [row for row in rows if row["kind"] == kind], system
                )
                for system in ("typesafe", "needle")
            }
            for kind in TASKS
        },
        "rows": rows,
    }


def save_checkpoint(rows: list[dict[str, Any]]) -> None:
    if CHECKPOINT:
        path = Path(CHECKPOINT)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(result_document(rows), indent=2))
        temporary.replace(path)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for kind, task in TASKS.items():
        for case in task["cases"]:
            row: dict[str, Any] = {
                "kind": kind,
                "name": case.name,
                "expected": case.expected,
                "ambiguous": case.ambiguous,
                "context_length": case.context_length,
                "pair_id": case.pair_id,
                "adversarial": case.adversarial,
                "char_count": len(case.state),
            }
            for system, function in (
                ("typesafe", typesafe_result),
                ("needle", needle_result),
            ):
                try:
                    row[system] = function(kind, case.state)
                except Exception as exc:
                    row[system] = {
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            rows.append(row)
            save_checkpoint(rows)
            print(f"{len(rows):02d} {kind}/{case.name}", file=sys.stderr, flush=True)
    print(json.dumps(result_document(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
