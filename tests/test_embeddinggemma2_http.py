"""Real-weight CUDA endpoint checks; opt in with GEWELL_EMBEDDINGGEMMA2_HTTP=1."""
import base64
import http.client
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/embeddinggemma-2"
pytestmark = pytest.mark.skipif(os.environ.get("GEWELL_EMBEDDINGGEMMA2_HTTP") != "1", reason="requires native BF16 bundle and SM120 GPU")


class Server:
    def __init__(self, directory, *, bundle=None, **options):
        bundle = bundle or Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_BUNDLE", ROOT / "artifacts/embeddinggemma2/native-bf16"))
        binary = Path(os.environ.get("GEWELL_BINARY", ROOT / "build/gewell"))
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.log = (directory / f"{self.port}.log").open("w+")
        args = [str(binary), "--log-format", "json", "serve-embeddings", str(bundle), "--port", str(self.port)]
        for key, value in options.items():
            args.append("--" + key.replace("_", "-"))
            if value is not True:
                args.append(str(value))
        self.process = subprocess.Popen(args, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    break
                try:
                    if self.request("GET", "/health")[0] == 200:
                        return
                except OSError:
                    pass
                time.sleep(.05)
            self.log.seek(0)
            raise AssertionError(self.log.read())
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=30)
        self.log.seek(0)
        log = self.log.read()
        self.log.close()
        assert self.process.returncode == 0, log

    def request(self, method, path, value=None, *, raw=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            connection.request(method, path, raw if raw is not None else None if value is None else json.dumps(value), {"Content-Type": "application/json"})
            response = connection.getresponse()
            data = response.read().decode()
            return response.status, data if path == "/metrics" else json.loads(data)
        finally:
            connection.close()

    def embed(self, text="hello", **fields):
        return self.request("POST", "/v1/embeddings", {"model": MODEL, "input": text, **fields})

    def start(self, text, **fields):
        body = json.dumps({"model": MODEL, "input": text, **fields}).encode()
        connection = socket.create_connection(("127.0.0.1", self.port), timeout=30)
        connection.sendall(f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        return connection

    def metric(self, name):
        status, body = self.request("GET", "/metrics")
        assert status == 200
        match = re.search(r"^gewell:embedding_" + name + r"\{[^\n]*\} ([^\n]+)$", body, re.M)
        assert match, body
        return float(match[1])

    def wait_metric(self, name, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            value = self.metric(name)
            if predicate(value):
                return value
            time.sleep(.005)
        raise AssertionError((name, value))


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("embedding-http")
    servers = []
    def create(**options):
        server = Server(directory, **options)
        servers.append(server)
        return server
    yield create
    for server in reversed(servers):
        server.close()


@pytest.fixture(scope="module")
def server(factory):
    return factory()


def test_models_and_dedicated_routes(server):
    assert server.request("GET", "/v1/models")[1]["data"][0]["id"] == MODEL
    assert server.request("GET", "/v1/models/google%2Fembeddinggemma-2")[1]["id"] == MODEL
    assert server.request("GET", "/v1/models/missing")[0] == 404
    assert server.embed(model="missing")[0] == 404
    for method, path in [("POST", "/v1/completions"), ("POST", "/v1/chat/completions"), ("POST", "/v1/cache/prefill"),
                         ("GET", "/v1/cache/index"), ("GET", "/v1/cache/stats")]:
        assert server.request(method, path, {} if method == "POST" else None)[0] == 404
    assert server.request("GET", "/v1/embeddings")[0] == 405
    metrics = server.request("GET", "/metrics")[1]
    assert "gewell:embedding_scratch_bytes" in metrics
    assert "generation_tokens" not in metrics and "kv_cache" not in metrics


@pytest.mark.parametrize("bad", [None, "", [], [""], ["ok", ""], [1], ["ok", 1], [[1]], 1, {}, True])
def test_invalid_input_is_atomic(server, bad):
    before = server.metric("inputs_total")
    assert server.embed(bad)[0] == 400
    assert server.metric("inputs_total") == before


@pytest.mark.parametrize("fields", [
    {"dimensions": d} for d in [0, 127, 129, 769, -128, 128.0, True, None, "128", 2**64-1]
] + [{"encoding_format": e} for e in [None, 1, "binary", "FLOAT"]] +
    [{"stream": False}, {"temperature": 0}, {"user": "a"}, {"messages": []}, {"model": None}])
def test_invalid_fields(server, fields):
    assert server.embed(**fields)[0] == 400


def test_malformed_and_missing_fields(server):
    for raw in ["{", "[]", "null", '{"model":"google/embeddinggemma-2"}', '{"input":"hello"}']:
        assert server.request("POST", "/v1/embeddings", raw=raw)[0] == 400
    for marker in ["<|image|>", "<|audio|>", "<|video|>"]:
        assert server.embed(["hello", marker])[0] == 400
    assert server.embed("<mask>" * 8191)[0] == 400


@pytest.mark.parametrize("dimension", [128, 256, 512, 768])
def test_http_vectors_exactly_match_native_corpus(server, dimension):
    path = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_NATIVE_EXPORT",
                ROOT / "artifacts/embeddinggemma2/text-native-audio-regression-v1.json"))
    if not path.exists():
        pytest.skip("provide the saved native parity export")
    cases = json.loads(path.read_text())["cases"]
    status, result = server.embed([c["text"] for c in cases], dimensions=dimension)
    assert status == 200, result
    assert result["object"] == "list" and result["model"] == MODEL
    assert len(result["data"]) == len(cases)
    tokens = sum(len(c["token_ids"]) for c in cases)
    assert result["usage"] == {"prompt_tokens": tokens, "total_tokens": tokens}
    for i, (row, case) in enumerate(zip(result["data"], cases, strict=True)):
        assert row["object"] == "embedding" and row["index"] == i
        assert row["embedding"] == case["embeddings"][str(dimension)]
        assert abs(sum(v*v for v in row["embedding"])**.5 - 1) <= .005


@pytest.mark.parametrize("dimension", [128, 256, 512, 768])
def test_float_base64_and_single_batch_equivalence(server, dimension):
    texts = [" task: search result | query: hello\n", "<bos>literal<eos>", "<pad>", "é 日本語"]
    status, floats = server.embed(texts, dimensions=dimension)
    assert status == 200
    status, encoded = server.embed(texts, dimensions=dimension, encoding_format="base64")
    assert status == 200
    assert floats["usage"] == encoded["usage"]
    for text, row, coded in zip(texts, floats["data"], encoded["data"], strict=True):
        decoded = list(struct.unpack("<" + "f"*dimension, base64.b64decode(coded["embedding"], validate=True)))
        assert decoded == row["embedding"] == server.embed(text, dimensions=dimension)[1]["data"][0]["embedding"]


def test_openai_client(server):
    from openai import OpenAI
    with OpenAI(api_key="local", base_url=f"http://127.0.0.1:{server.port}/v1", max_retries=0) as client:
        result = client.embeddings.create(model=MODEL, input=["hello", "日本語"], dimensions=256)
        direct = server.embed(["hello", "日本語"], dimensions=256)[1]
        assert [r.embedding for r in result.data] == [r["embedding"] for r in direct["data"]]
        assert result.usage.total_tokens == direct["usage"]["total_tokens"]
        assert len(client.embeddings.create(model=MODEL, input="hello", encoding_format="float").data[0].embedding) == 768


def test_configured_limits_and_model_alias(factory):
    bounded = factory(max_inputs=2, max_batch_tokens=513, max_output_bytes=10000)
    assert bounded.embed(["a", "b", "c"], dimensions=128)[0] == 400
    assert bounded.embed("<mask>"*512, dimensions=128)[0] == 400
    assert bounded.embed("<mask>"*511, dimensions=128)[0] == 200
    assert bounded.embed(dimensions=768, encoding_format="float")[0] == 400
    assert bounded.embed(dimensions=768, encoding_format="base64")[0] == 200
    aliased = factory(model="my-encoder")
    assert aliased.embed()[0] == 404
    assert aliased.embed(model="my-encoder")[1]["model"] == "my-encoder"


def test_overload_cancel_and_reuse(factory):
    server = factory(max_pending=1)
    connection = server.start(["<mask>"*8190]*32)
    try:
        server.wait_metric("requests_running", lambda n: n == 1)
        assert server.embed()[0] == 503
    finally:
        connection.close()
    server.wait_metric("requests_cancelled_total", lambda n: n == 1)
    assert server.metric("inputs_total") < 32
    assert server.embed()[0] == 200
    server.wait_metric("requests_completed_total", lambda n: n == 1)
    assert server.metric("requests_rejected_total") == 1
    assert server.metric("requests_running") == 0


def test_queued_cancellation_is_removed_without_execution(factory):
    server = factory(max_pending=2)
    active = server.start(["<mask>"*8190]*32)
    try:
        server.wait_metric("requests_running", lambda n: n == 1)
        queued = server.start("queued input")
        try:
            server.wait_metric("requests_waiting", lambda n: n == 1)
        finally:
            queued.close()
        server.wait_metric("requests_cancelled_total", lambda n: n == 1)
        assert server.metric("requests_waiting") == 0
        assert server.metric("requests_running") == 1
    finally:
        active.close()
    server.wait_metric("requests_cancelled_total", lambda n: n == 2)
    assert server.embed()[0] == 200


def test_body_and_connection_time_limits(factory):
    server = factory(max_body_bytes=256, max_body_total_bytes=512, socket_timeout_seconds=1)
    assert server.embed("hello"*100)[0] == 413
    with socket.create_connection(("127.0.0.1", server.port), timeout=5) as connection:
        connection.sendall(b"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n{")
        assert connection.recv(1) == b""
    assert server.embed()[0] == 200
