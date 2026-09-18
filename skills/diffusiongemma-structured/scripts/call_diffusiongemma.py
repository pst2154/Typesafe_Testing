#!/usr/bin/env python3
"""Call a DiffusionGemma structured-decision service.

The input question file uses a convenient keyed Choice/Noul/Score format. The
script translates it to the nested OpenAI Chat Completions request and prints
the decoded structured result. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def endpoint_url(configured: str) -> str:
    """Normalize a base URL or full endpoint URL."""
    value = configured.rstrip("/")
    if value.endswith("/v1/chat/completions"):
        return value
    return f"{value}/v1/chat/completions"


def health_url(configured: str) -> str:
    """Return the health URL for a base or chat-completions URL."""
    value = configured.rstrip("/")
    suffix = "/v1/chat/completions"
    if value.endswith(suffix):
        value = value[: -len(suffix)]
    return f"{value}/health"


def load_json_or_text(path: Path) -> Any:
    """Load JSON when possible, otherwise preserve the file as text state."""
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_inline_state(value: str) -> Any:
    """Accept either a JSON value or ordinary state text."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_question(question_id: str, source: dict[str, Any]) -> dict[str, Any]:
    """Translate a keyed TypeSafe-style question to the service schema."""
    kind = str(source.get("type", "")).lower()
    instructions = source.get("instructions")
    if kind not in {"choice", "noul", "score"}:
        raise ValueError(
            f"question {question_id!r} has unsupported type {kind!r}; "
            "expected choice, noul, or score"
        )
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError(f"question {question_id!r} needs non-empty instructions")

    result: dict[str, Any] = {
        "id": question_id,
        "type": kind,
        "instructions": instructions,
    }
    criteria = source.get("criteria")

    if kind == "choice":
        options = source.get("options")
        if options is None:
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(
                    f"choice {question_id!r} needs at least two criteria entries"
                )
            options = [
                {"name": name, "description": description}
                for name, description in criteria.items()
            ]
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"choice {question_id!r} needs at least two options")
        result["options"] = options

    elif kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict) or not {
                "true",
                "false",
            }.issubset(criteria):
                raise ValueError(
                    f"noul {question_id!r} criteria must define true and false"
                )
            result["instructions"] += (
                f" Yes means: {criteria['true']} No means: {criteria['false']}"
            )

    else:
        levels = source.get("levels", criteria)
        if not isinstance(levels, list) or len(levels) < 2:
            raise ValueError(f"score {question_id!r} needs at least two levels")
        result["levels"] = levels

    return result


def normalize_questions(source: Any) -> list[dict[str, Any]]:
    """Accept keyed questions or an already normalized question array."""
    if isinstance(source, dict):
        return [normalize_question(str(key), value) for key, value in source.items()]
    if isinstance(source, list):
        normalized = []
        for index, value in enumerate(source):
            if not isinstance(value, dict) or not value.get("id"):
                raise ValueError(f"question at index {index} needs an id")
            question_id = str(value["id"])
            normalized.append(normalize_question(question_id, value))
        return normalized
    raise ValueError("questions must be a JSON object or array")


def request_json(
    url: str,
    payload: dict[str, Any] | None,
    *,
    timeout: float,
    api_key: str | None,
) -> Any:
    """Make one JSON HTTP request without leaking authorization details."""
    headers = {"Accept": "application/json"}
    data = None
    method = "GET"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url, data=data, headers=headers, method=method
    )
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request failed: {error.reason}") from error
    if not body:
        return {"ok": True}
    return json.loads(body)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Call a DiffusionGemma structured-decision endpoint."
    )
    state = result.add_mutually_exclusive_group()
    state.add_argument("--state", help="State as text or an inline JSON value")
    state.add_argument("--state-file", type=Path, help="JSON or text state file")
    result.add_argument("--questions-file", type=Path, help="JSON question file")
    result.add_argument(
        "--url",
        default=os.environ.get("DIFFUSIONGEMMA_URL"),
        help="Base or chat-completions URL; defaults to DIFFUSIONGEMMA_URL",
    )
    result.add_argument(
        "--model",
        default=os.environ.get("DIFFUSIONGEMMA_MODEL", "dgemma-structured"),
    )
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--samples", default="auto")
    result.add_argument("--auto-threshold", type=float, default=0.1)
    result.add_argument("--auto-max", type=int, default=4)
    result.add_argument("--steps", type=int, default=1)
    result.add_argument("--think", type=int, default=0)
    result.add_argument("--timeout", type=float, default=600)
    result.add_argument("--raw", action="store_true", help="Print the outer response")
    result.add_argument("--health", action="store_true", help="Check health and exit")
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.url:
        print("DIFFUSIONGEMMA_URL or --url is required", file=sys.stderr)
        return 2
    api_key = os.environ.get("DIFFUSIONGEMMA_API_KEY")

    try:
        if args.health:
            result = request_json(
                health_url(args.url),
                None,
                timeout=args.timeout,
                api_key=api_key,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

        if not args.questions_file:
            raise ValueError("--questions-file is required unless --health is used")
        if args.state_file:
            state = load_json_or_text(args.state_file)
        elif args.state is not None:
            state = parse_inline_state(args.state)
        else:
            raise ValueError("--state or --state-file is required")

        questions_source = json.loads(args.questions_file.read_text())
        questions = normalize_questions(questions_source)
        try:
            samples: str | int = int(args.samples)
        except ValueError:
            if args.samples != "auto":
                raise ValueError("--samples must be a positive integer or 'auto'")
            samples = "auto"
        if isinstance(samples, int) and samples < 1:
            raise ValueError("--samples must be positive")

        schema = {
            "questions": questions,
            "samples": samples,
            "auto_threshold": args.auto_threshold,
            "auto_max": args.auto_max,
            "steps": args.steps,
            "think": args.think,
        }
        payload = {
            "model": args.model,
            "seed": args.seed,
            "messages": [
                {"role": "system", "content": json.dumps(schema, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": json.dumps({"state": state}, ensure_ascii=False),
                },
            ],
        }
        outer = request_json(
            endpoint_url(args.url),
            payload,
            timeout=args.timeout,
            api_key=api_key,
        )
        if args.raw:
            output = outer
        else:
            content = outer["choices"][0]["message"]["content"]
            output = json.loads(content)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (KeyError, IndexError, json.JSONDecodeError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
