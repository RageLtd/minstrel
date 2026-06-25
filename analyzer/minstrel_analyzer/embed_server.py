from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .clap import Clap, EMBED_DIM


def make_handler(clap: Clap) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to a loaded CLAP instance.

    POST /embed_text {"text": "..."} -> {"embedding": [512 floats]}
    GET  /healthz                    -> {"status": "ok", "dim": 512}
    """

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok", "dim": EMBED_DIM})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/embed_text":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, {"error": "invalid JSON"})
                return
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                self._send(400, {"error": "missing 'text'"})
                return
            embedding = clap.embed_text([text])[0]
            self._send(200, {"embedding": embedding.tolist()})

        def log_message(self, *args: object) -> None:
            pass  # quiet by default

    return Handler


def serve(clap: Clap | None = None, host: str = "0.0.0.0", port: int = 8001) -> None:
    clap = clap or Clap()
    server = ThreadingHTTPServer((host, port), make_handler(clap))
    print(f"clap embed server listening on {host}:{port}")
    server.serve_forever()


def main() -> None:
    serve(
        host=os.environ.get("MINSTREL_EMBED_HOST", "0.0.0.0"),
        port=int(os.environ.get("MINSTREL_EMBED_PORT", "8001")),
    )


if __name__ == "__main__":
    main()
