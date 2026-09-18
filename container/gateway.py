#!/usr/bin/env python3
"""TypeSafe HTTP compatibility layer for a structured DiffusionGemma backend."""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


_pool_lock = threading.Lock()
_pool_positions: dict[str, int] = {}


def configured_urls(name: str, fallback: str) -> list[str]:
    urls = [url.strip() for url in os.environ.get(name, fallback).split(",") if url.strip()]
    if not urls:
        raise ValueError(f"{name} contains no backend URLs")
    return urls


def ordered_urls(name: str, fallback: str) -> list[str]:
    urls = configured_urls(name, fallback)
    with _pool_lock:
        start = _pool_positions.get(name, 0) % len(urls)
        _pool_positions[name] = start + 1
    return urls[start:] + urls[:start]


def post_to_pool(name: str, fallback: str, body: bytes) -> tuple[bytes, str]:
    errors: list[str] = []
    timeout = float(os.environ.get("BACKEND_TIMEOUT_SECONDS", "300"))
    for url in ordered_urls(name, fallback):
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        backend_key = os.environ.get("DIFFUSION_API_KEY")
        if backend_key:
            request.add_header("Authorization", f"Bearer {backend_key}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.headers.get("Content-Type", "application/json")
        except urllib.error.HTTPError as error:
            body_text = error.read().decode(errors="replace")
            if error.code < 500:
                raise RuntimeError(f"upstream HTTP {error.code}: {body_text}") from error
            errors.append(f"{url}: HTTP {error.code}")
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("all backends failed: " + "; ".join(errors))


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_request(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    if "state" not in payload:
        raise ValueError("state is required")
    if not isinstance(payload.get("model"), str):
        raise ValueError("model is required and must be a string")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions is required and must be a non-empty object")
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question ids must be non-empty strings")
        if not isinstance(question, dict):
            raise ValueError(f"questions.{question_id} must be an object")
        kind = question.get("type")
        if kind not in {"choice", "noul", "score"}:
            raise ValueError(f"questions.{question_id}.type must be choice, noul, or score")
        if "instructions" not in question:
            raise ValueError(f"questions.{question_id}.instructions is required")
        criteria = question.get("criteria")
        if kind == "choice" and (not isinstance(criteria, dict) or len(criteria) < 2):
            raise ValueError(f"questions.{question_id}.criteria must contain at least two choices")
        if kind == "score" and (not isinstance(criteria, list) or len(criteria) < 2):
            raise ValueError(f"questions.{question_id}.criteria must contain at least two levels")
        if kind == "noul" and criteria is not None and not isinstance(criteria, dict):
            raise ValueError(f"questions.{question_id}.criteria must be an object when provided")


def diffusion_questions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for question_id, question in payload["questions"].items():
        kind = question["type"]
        item: dict[str, Any] = {
            "id": question_id,
            "type": kind,
            "instructions": as_text(question["instructions"]),
        }
        if kind == "choice":
            item["options"] = [
                {"name": name, "description": as_text(description)}
                for name, description in question["criteria"].items()
            ]
        elif kind == "noul":
            criteria = question.get("criteria") or {}
            if "true" in criteria:
                item["instructions"] += f" Yes means: {as_text(criteria['true'])}"
            if "false" in criteria:
                item["instructions"] += f" No means: {as_text(criteria['false'])}"
        else:
            item["levels"] = [as_text(level) for level in question["criteria"]]
        converted.append(item)
    return converted


def diffusion_payload(payload: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    samples_value = env.get("DIFFUSION_SAMPLES", "auto")
    samples: str | int = int(samples_value) if samples_value.isdigit() else samples_value
    schema = {
        "questions": diffusion_questions(payload),
        "samples": samples,
        "auto_threshold": float(env.get("DIFFUSION_AUTO_THRESHOLD", "0.1")),
        "auto_max": int(env.get("DIFFUSION_AUTO_MAX", "4")),
        "steps": int(env.get("DIFFUSION_STEPS", "1")),
        "think": int(env.get("DIFFUSION_THINK", "0")),
    }
    return {
        "model": env.get("DIFFUSION_MODEL", "dgemma-structured"),
        "seed": int(env.get("DIFFUSION_SEED", "42")),
        "messages": [
            {"role": "system", "content": json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": json.dumps({"state": payload["state"]}, ensure_ascii=False)},
        ],
    }


def typesafe_response(
    request_payload: dict[str, Any], outer: dict[str, Any], model: str = "dgemma-structured"
) -> dict[str, Any]:
    content = outer["choices"][0]["message"]["content"]
    decoded = json.loads(content)
    raw_answers = decoded["answers"]
    answers: dict[str, Any] = {}
    for question_id, question in request_payload["questions"].items():
        raw = raw_answers[question_id]
        kind = question["type"]
        if kind == "noul":
            answers[question_id] = {"type": "noul", "noul": float(raw["noul"])}
        elif kind == "choice":
            answers[question_id] = {
                "type": "choice",
                "choice": raw["choice"],
                "probabilities": raw["probabilities"],
                "confidence": float(raw["confidence"]),
            }
        else:
            levels = [as_text(level) for level in question["criteria"]]
            raw_probabilities = raw["probabilities"]
            probabilities = {
                str(index): float(
                    raw_probabilities.get(level, raw_probabilities.get(str(index + 1), 0.0))
                )
                for index, level in enumerate(levels)
            }
            score = max(0.0, min(float(len(levels) - 1), float(raw["score"]) - 1.0))
            answers[question_id] = {
                "type": "score",
                "score": score,
                "legend": {str(index): level for index, level in enumerate(levels)},
                "probabilities": probabilities,
                "confidence": float(raw["confidence"]),
            }

    usage = outer.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return {
        "model": model,
        "answers": answers,
        "usage": {
            "input_tokens": int(prompt_tokens or 0),
            "output_tokens": int(completion_tokens or 0),
        },
    }


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "DiffusionGateway/2.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, payload: Any) -> None:
        self.send_bytes(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json")

    def send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def send_explorer(self) -> None:
        explorer_path = Path(os.environ.get(
            "EXPLORER_HTML", str(Path(__file__).with_name("explorer.html"))
        ))
        try:
            body = explorer_path.read_bytes()
        except OSError as error:
            print(f"explorer error: {error!r}", file=sys.stderr, flush=True)
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "explorer unavailable"}})
            return
        self.send_bytes(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_json(HTTPStatus.NO_CONTENT, {})

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in {"/", "/explorer"}:
            self.send_explorer()
        elif path == "/health":
            self.send_json(HTTPStatus.OK, {
                "status": "ok",
                "service": "diffusion-unified-gateway",
                "structured_backends": len(configured_urls(
                    "DIFFUSION_URLS", "http://127.0.0.1:8011/v1/chat/completions"
                )),
                "raw_backends": len(configured_urls(
                    "RAW_URLS", "http://127.0.0.1:8000/v1/chat/completions"
                )),
            })
        elif path == "/v1/models":
            self.send_json(HTTPStatus.OK, {"object": "list", "data": [
                {"id": "dgemma-structured", "object": "model", "owned_by": "local"},
                {"id": "dgemma", "object": "model", "owned_by": "local"},
            ]})
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        path = self.path.rstrip("/")
        if path not in {"/v1/systemone", "/v1/chat/completions", "/v1/raw/chat/completions"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        required_key = os.environ.get("GATEWAY_API_KEY")
        if required_key:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {required_key}"
            if not secrets.compare_digest(supplied, expected):
                self.send_json(HTTPStatus.UNAUTHORIZED, {"error": {"message": "invalid API key"}})
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > int(os.environ.get("MAX_BODY_BYTES", "10485760")):
                raise ValueError("request body is empty or too large")
            request_body = self.rfile.read(length)
            if path == "/v1/systemone":
                payload = json.loads(request_body)
                validate_request(payload)
                translated = json.dumps(diffusion_payload(payload), ensure_ascii=False).encode()
                response_body, _ = post_to_pool(
                    "DIFFUSION_URLS",
                    "http://127.0.0.1:8011/v1/chat/completions",
                    translated,
                )
                outer = json.loads(response_body)
                self.send_json(
                    HTTPStatus.OK,
                    typesafe_response(
                        payload, outer, os.environ.get("GATEWAY_MODEL_NAME", "dgemma-structured")
                    ),
                )
            elif path == "/v1/chat/completions":
                response_body, content_type = post_to_pool(
                    "DIFFUSION_URLS",
                    "http://127.0.0.1:8011/v1/chat/completions",
                    request_body,
                )
                self.send_bytes(HTTPStatus.OK, response_body, content_type)
            else:
                response_body, content_type = post_to_pool(
                    "RAW_URLS",
                    "http://127.0.0.1:8000/v1/chat/completions",
                    request_body,
                )
                self.send_bytes(HTTPStatus.OK, response_body, content_type)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": {"message": str(error)}})
        except Exception as error:
            print(f"gateway error: {error!r}", file=sys.stderr, flush=True)
            self.send_json(HTTPStatus.BAD_GATEWAY, {"error": {"message": str(error)}})


class GatewayServer(ThreadingHTTPServer):
    """Threaded HTTP server sized for short, bursty classification traffic."""

    daemon_threads = True
    block_on_close = False
    request_queue_size = int(os.environ.get("GATEWAY_BACKLOG", "256"))


def main() -> None:
    host = os.environ.get("GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("GATEWAY_PORT", "8012"))
    server = GatewayServer((host, port), GatewayHandler)
    count = len(configured_urls(
        "DIFFUSION_URLS", "http://127.0.0.1:8011/v1/chat/completions"
    ))
    print(f"Unified diffusion gateway listening on {host}:{port} with {count} backends", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
