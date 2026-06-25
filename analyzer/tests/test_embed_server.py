import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from minstrel_analyzer.clap import Clap, EMBED_DIM
from minstrel_analyzer.embed_server import make_handler


def _post(port: int, path: str, payload: dict) -> tuple[int, dict]:
    req = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def test_embed_text_endpoint_returns_512_dims(clap: Clap):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(clap))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, body = _post(port, "/embed_text", {"text": "heavy doom metal"})
        assert status == 200
        assert len(body["embedding"]) == EMBED_DIM
    finally:
        server.shutdown()
