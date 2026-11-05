"""A small HTTP client for the inference service; it needs no torch."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .protocol import SCHEMA_VERSION, ActionChunk, Observation, strict_json

MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class Client:
    def __init__(self, url="http://127.0.0.1:8000", *, timeout=30.0):
        parts = urlsplit(url)
        if (
            parts.scheme not in ("http", "https")
            or not parts.hostname
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError("A valid HTTP service URL is required")
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not 0 < timeout < float("inf")
        ):
            raise ValueError("Timeout must be finite and positive")
        self.url = url.rstrip("/")
        self.timeout = float(timeout)

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            code = exc.code
            exc.close()
            raise RuntimeError(f"Inference service returned HTTP {code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("Unable to contact the inference service") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeds the supported size")
        try:
            result = strict_json(raw)
        except (ValueError, UnicodeError):
            raise ValueError("Invalid service response") from None
        if not isinstance(result, dict) or result.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported service response schema")
        return result

    def health(self):
        return self._call("/health")

    def metadata(self):
        return self._call("/metadata")

    def predict(self, observation: Observation) -> ActionChunk:
        result = ActionChunk.from_dict(self._call("/predict", observation.to_dict()))
        if (
            result.request_id != observation.request_id
            or result.observation_timestamp_s != observation.timestamp_s
        ):
            raise ValueError("Response identity differs from the observation")
        return result
