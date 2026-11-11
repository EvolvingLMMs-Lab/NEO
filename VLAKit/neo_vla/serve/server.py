"""The HTTP service: /health, /metadata and /predict over one backend.

Predictions are served one at a time. A failed request gets a short error and
its request id; the traceback goes to the server log only.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .backend import ModelInfo, load_backend
from .protocol import SCHEMA_VERSION, ActionChunk, Observation, strict_json

LOGGER = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 16 * 1024 * 1024
DEFAULT_BACKEND = "neo_vla.serve.neo_backend:create_backend"


class InferenceServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, backend):
        self.backend = backend
        self.model_info = backend.info()
        if not isinstance(self.model_info, ModelInfo):
            raise TypeError("Backend metadata must be ModelInfo")
        self.prediction_lock = threading.Lock()
        super().__init__(address, RequestHandler)

    def handle_error(self, request, client_address):
        LOGGER.exception("request handler failed")


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "VLAInference/1.0"
    sys_version = ""

    def log_message(self, *args):
        # Request lines and bodies stay out of the log; failures are logged below.
        return

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def send_json(self, status, value):
        data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, dict(status="ok", schema_version=SCHEMA_VERSION))
        elif self.path == "/metadata":
            self.send_json(
                200, dict(schema_version=SCHEMA_VERSION, **self.server.model_info.to_dict())
            )
        else:
            self.send_json(404, dict(error="Unknown endpoint"))

    def do_POST(self):
        if self.path != "/predict":
            self.send_json(404, dict(error="Unknown endpoint"))
            return
        try:
            if self.headers.get("Transfer-Encoding") is not None:
                raise ValueError("Unsupported transfer encoding")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("JSON content type is required")
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                self.send_json(413, dict(error="Request size is outside the supported range"))
                return
            data = self.rfile.read(length)
            if len(data) != length:
                raise ValueError("Incomplete request")
            observation = Observation.from_dict(strict_json(data))
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ):
            self.send_json(400, dict(error="Invalid observation request"))
            return
        started = time.monotonic()
        try:
            with self.server.prediction_lock:
                result = self.server.backend.predict(observation)
                if not isinstance(result, ActionChunk):
                    raise TypeError("Backend must return ActionChunk")
                if (
                    result.request_id != observation.request_id
                    or result.observation_timestamp_s != observation.timestamp_s
                    or result.model_id != self.server.model_info.model_id
                ):
                    raise ValueError("Backend response identity differs from the request or model")
                payload = result.to_dict()
                ActionChunk.from_dict(payload)
        except Exception:
            LOGGER.exception("prediction failed: request_id=%s", observation.request_id)
            self.send_json(500, dict(error="Prediction failed", request_id=observation.request_id))
            return
        LOGGER.info(
            "prediction done: request_id=%s elapsed_s=%.3f",
            observation.request_id,
            time.monotonic() - started,
        )
        self.send_json(200, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve a policy over HTTP")
    parser.add_argument("--backend", default=DEFAULT_BACKEND,
                        help="module:factory of the backend (default: the NEO-VLA backend)")
    parser.add_argument("--config", type=Path, help="backend JSON config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 0 <= args.port <= 65535:
        parser.error("Port must be in the range 0..65535")
    backend = None
    try:
        config = {} if args.config is None else strict_json(args.config.read_bytes())
        if not isinstance(config, dict):
            raise ValueError("Backend configuration must be an object")
        backend = load_backend(args.backend, config)
        with InferenceServer((args.host, args.port), backend) as server:
            LOGGER.info("serving on port %d", server.server_address[1])
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                LOGGER.info("stopped")
    except Exception:
        LOGGER.exception("the service could not start or stopped on an error")
        return 1
    finally:
        if backend is not None:
            backend.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
