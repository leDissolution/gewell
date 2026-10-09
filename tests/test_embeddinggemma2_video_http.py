"""Real-weight video endpoint parity, resource ownership and cancellation."""
import base64
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

import pytest

from tests.test_embeddinggemma2_http import Server, ROOT, MODEL

pytestmark = pytest.mark.skipif(os.environ.get("GEWELL_EMBEDDINGGEMMA2_HTTP") != "1", reason="requires SM120 GPU and saved video controls")
BUNDLE = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_IMAGE_BUNDLE", ROOT/"artifacts/embeddinggemma2/native-bf16-image"))
ARTIFACTS = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_ARTIFACTS", ROOT/"artifacts/embeddinggemma2"))


@pytest.fixture(scope="module")
def cases():
    result = {}
    for group in ("video", "video-edge"):
        variable = "GEWELL_EMBEDDINGGEMMA2_"+group.upper().replace("-","_")+"_EXPORT"
        native = Path(os.environ.get(variable,ARTIFACTS/f"{group}-native-audio-regression-v1/vectors.json"))
        fixtures = ARTIFACTS/f"{group}-fixtures-v1"
        if not native.exists() or not fixtures.exists():
            pytest.skip("save base/edge video fixtures and native parity exports")
        for case in json.loads(native.read_text())["cases"]:
            result[case["id"]] = {**case, "fixtures": fixtures}
    return result


def media(path):
    mime = {".mp4":"video/mp4", ".webm":"video/webm", ".png":"image/png"}[path.suffix]
    kind = "image_url" if path.suffix == ".png" else "video_url"
    return {"type":kind, kind:{"url":f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()}}


def content(case):
    return {"content":[{"type":"text", "text":p["text"]} if "text" in p else
        media(case["fixtures"]/p["video" if "video" in p else "image"]) for p in case["content"]]}


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("video-embedding-http")
    servers = []
    def create(vision=True, **options):
        server = Server(directory, bundle=BUNDLE, **({"vision":True} if vision else {}), **options)
        servers.append(server)
        return server
    yield create
    for server in reversed(servers):
        server.close()


@pytest.fixture(scope="module")
def server(factory):
    return factory()


@pytest.mark.parametrize("dimension", [128,256,512,768])
def test_every_video_case_matches_native_exactly(server, cases, dimension):
    ordered = list(cases.values())
    # A pair of maximum-length fixtures fits the default prepared-memory budget.
    for begin in range(0,len(ordered),2):
        group = ordered[begin:begin+2]
        assert len({c["max_soft_tokens"] for c in group}) == 1
        status, result = server.embed([content(c) for c in group], dimensions=dimension,
            mm_processor_kwargs={"max_soft_tokens":group[0]["max_soft_tokens"]})
        assert status == 200, result
        assert [r["index"] for r in result["data"]] == list(range(len(group)))
        assert [r["embedding"] for r in result["data"]] == [c["embeddings"][str(dimension)] for c in group]
        assert result["usage"] == dict.fromkeys(("prompt_tokens","total_tokens"),sum(len(c["token_ids"]) for c in group))


def test_mixed_batch_base64_and_separate_media_accounting(server, cases):
    group = [cases[k] for k in ("query-red", "interleaved", "red")]
    counters = ("images_total","image_soft_tokens_total","videos_total","video_frames_total","video_soft_tokens_total")
    before = {k:server.metric(k) for k in counters}
    status, result = server.embed([group[0]["text"],content(group[1]),content(group[2])], dimensions=256,
        encoding_format="base64", mm_processor_kwargs={"max_soft_tokens":70})
    assert status == 200, result
    assert [list(struct.unpack("<256f",base64.b64decode(r["embedding"]))) for r in result["data"]] == [c["embeddings"]["256"] for c in group]
    spans = [(c["token_ids"][begin],end-begin) for c in group for begin,end in c["spans"]]
    expected = {
        "images_total":sum(token==258880 for token,n in spans),
        "image_soft_tokens_total":sum(n for token,n in spans if token==258880),
        "videos_total":sum(len(c["videos"]) for c in group),
        "video_frames_total":sum(token==258884 for token,n in spans),
        "video_soft_tokens_total":sum(n for token,n in spans if token==258884),
    }
    assert {k:server.metric(k)-before[k] for k in counters} == expected


def test_image_budget_does_not_change_video(server, cases):
    value = content(cases["lowfps"])
    results = [server.embed(value,mm_processor_kwargs={"max_soft_tokens":budget}) for budget in (70,280,1120)]
    assert results[0][0] == 200
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("part,suffix", [
    ({"type":"video_url"},"video_url"),
    ({"type":"video_url","video_url":[]},"video_url"),
    ({"type":"video_url","video_url":{}},"video_url.url"),
    ({"type":"video_url","video_url":{"url":1}},"video_url.url"),
    ({"type":"video_url","video_url":{"url":"x","fps":1}},"video_url.fps"),
    ({"type":"video_url","video_url":{"url":"x"},"max_frames":32},"max_frames"),
])
def test_indexed_schema_errors_are_atomic(server, part, suffix):
    before = server.metric("inputs_total")
    status, result = server.embed(["valid",{"content":[part]}])
    assert status == 400, result
    assert result["error"]["param"] == "input[1].content[0]."+suffix
    assert server.metric("inputs_total") == before


@pytest.mark.parametrize("url", [
    "https://example.com/video.mp4", "/tmp/video.mp4", "data:video/avi;base64,AAAA",
    "data:video/mp4;base64,", "data:video/mp4;base64,AB==", "data:video/mp4;base64,AAAA",
    "data:video/webm;base64,AAAA",
])
def test_invalid_video_transport(server, url):
    status, result = server.embed({"content":[{"type":"video_url","video_url":{"url":url}}]})
    assert status == 400 and result["error"]["code"] == "invalid_video", result
    assert result["error"]["param"] == "input.content[0].video_url.url"


@pytest.mark.parametrize("name", ["unsupported-codec.mp4", "too-wide.mp4", "too-many-pixels.mp4", "truncated-mp4.mp4", "truncated-webm.webm", "changing-size.mp4"])
def test_rejected_decoder_inputs_are_atomic(server, cases, name):
    before = server.metric("inputs_total")
    status, result = server.embed(["valid",{"content":[media(cases["vfr"]["fixtures"]/name)]}])
    assert status == 400 and result["error"]["code"] == "invalid_video", result
    assert result["error"]["param"] == "input[1].content[0].video_url.url"
    assert server.metric("inputs_total") == before


def test_disabled_vision_and_wrong_container(factory, cases):
    disabled = factory(vision=False)
    assert disabled.embed(content(cases["short"]))[1]["error"]["code"] == "unsupported_parameter"
    value = content(cases["red"])
    value["content"][0]["video_url"]["url"] = value["content"][0]["video_url"]["url"].replace("video/mp4","video/webm",1)
    enabled = factory()
    assert enabled.embed(value)[1]["error"]["code"] == "invalid_video"


def test_data_url_size_bound(factory):
    server = factory(max_body_bytes=9*1024*1024,max_body_total_bytes=16*1024*1024)
    value = {"content":[{"type":"video_url","video_url":{"url":"data:video/mp4;base64,"+"A"*(8*1024*1024)}}]}
    status, result = server.embed(value)
    assert status == 400 and result["error"]["code"] == "invalid_video", result


def test_expanded_context_and_response_limits(factory, cases):
    case = cases["short"]
    n = len(case["token_ids"])
    bounded = factory(max_batch_tokens=n,max_output_bytes=10000,max_inputs=2)
    value = content(case)
    assert bounded.embed(value,dimensions=128)[0] == 200
    over = {"content":[{"type":"text","text":"<mask>"}]+value["content"]}
    status, result = bounded.embed(over,dimensions=128)
    assert status == 400 and result["error"]["code"] == "context_length_exceeded"
    assert bounded.embed(value)[1]["error"]["code"] == "output_limit_exceeded"
    assert bounded.embed([value]*3,dimensions=128)[0] == 400
    full = factory()
    for extra in (0,1):
        value = {"content":[{"type":"text","text":"<mask>"*(8192-n+extra)}]+content(case)["content"]}
        status, result = full.embed(value,dimensions=128)
        assert status == (400 if extra else 200), result
        if extra: assert result["error"]["code"] == "context_length_exceeded"
        else: assert result["usage"]["total_tokens"] == 8192


def test_partial_frame_preparation_releases_capacity(factory, cases):
    server = factory(max_body_total_bytes=6*1024*1024)
    # One frame fits; the three-frame clip fails when reserving its second frame.
    status, result = server.embed(content(cases["red"]))
    assert status == 503 and result["error"]["code"] == "capacity_exceeded", result
    assert server.metric("inputs_total") == 0
    short = content(cases["short"])
    assert server.embed(short)[0] == 200
    broken = {"content":short["content"]+[{"type":"text","text":None}]}
    assert server.embed(broken)[0] == 400
    assert server.embed(short)[0] == 200


def test_video_execution_cancel_overload_and_recovery(factory, cases):
    server = factory(max_pending=1)
    active = server.start([content(cases["uniform"])]*2)
    try:
        server.wait_metric("requests_running",lambda n:n==1)
        assert server.embed("busy")[0] == 503
    finally:
        active.close()
    server.wait_metric("requests_cancelled_total",lambda n:n==1)
    assert server.metric("inputs_total") < 2
    assert server.embed(content(cases["short"]))[0] == 200
    assert server.request("GET","/health")[0] == 200


def test_queued_video_cancellation_releases_capacity(factory, cases):
    server = factory(max_pending=2,max_body_total_bytes=6*1024*1024)
    active = server.start(["<mask>"*8190]*32)
    try:
        server.wait_metric("requests_running",lambda n:n==1)
        for count in (1,2):
            queued = server.start(content(cases["short"]))
            try:
                server.wait_metric("requests_waiting",lambda n:n==1)
            finally:
                queued.close()
            server.wait_metric("requests_cancelled_total",lambda n:n==count)
    finally:
        active.close()
    server.wait_metric("requests_cancelled_total",lambda n:n==3)
    assert server.embed(content(cases["short"]))[0] == 200


@pytest.fixture(scope="module")
def long_video(tmp_path_factory, cases):
    path = tmp_path_factory.mktemp("video-decode-cancellation")/"long.mp4"
    # Packet copying preserves the pinned supported stream and produces enough
    # frames to observe the count pass before any GPU admission.
    subprocess.run(["ffmpeg","-v","error","-n","-stream_loop","255","-i",
        str(cases["red"]["fixtures"]/"red.mp4"),"-c:v","copy","-an",str(path)],check=True)
    return {"content":[media(path)]}


def test_decode_disconnect_and_recovery(factory, cases, long_video):
    server = factory(max_body_total_bytes=6*1024*1024)
    active = server.start(long_video)
    time.sleep(.02)
    assert server.metric("requests_running") == 0
    active.close()
    assert server.embed(content(cases["short"]))[0] == 200
    assert server.metric("inputs_total") == 1


def test_completed_slow_reader_releases_frame_memory(factory, cases):
    server = factory(max_inputs=256)
    value = content(cases["uniform"])
    body = json.dumps({"model":MODEL,"input":[value]*2+["hello"]*254}).encode()
    client = socket.socket()
    client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
    client.settimeout(30)
    client.connect(("127.0.0.1",server.port))
    try:
        client.sendall(f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body)
        server.wait_metric("inputs_total",lambda n:n==256)
        server.wait_metric("requests_running",lambda n:n==0)
        assert server.metric("requests_completed_total") == 0, "response did not exercise backpressure"
        assert server.embed(value)[0] == 200
    finally:
        client.close()


@pytest.mark.parametrize("during_decode", [False,True])
def test_shutdown_with_video_work(tmp_path, cases, long_video, during_decode):
    server = Server(tmp_path,bundle=BUNDLE,vision=True)
    active = server.start(long_video if during_decode else [content(cases["uniform"])]*2)
    try:
        if during_decode:
            time.sleep(.02)
            assert server.metric("requests_running") == 0
        else:
            server.wait_metric("requests_running",lambda n:n==1)
    finally:
        server.close()
        active.close()
