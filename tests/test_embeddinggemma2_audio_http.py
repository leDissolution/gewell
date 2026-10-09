"""Real-weight audio endpoint parity, bounded preparation and request lifecycle."""
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

pytestmark = pytest.mark.skipif(os.environ.get("GEWELL_EMBEDDINGGEMMA2_HTTP") != "1", reason="requires SM120 GPU and saved audio controls")
BUNDLE = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_AUDIO_BUNDLE",ROOT/"artifacts/embeddinggemma2/native-bf16-multimodal"))
FIXTURES = ROOT/"artifacts/embeddinggemma2/audio-fixtures-v1"
NATIVE = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_AUDIO_EXPORT",ROOT/"artifacts/embeddinggemma2/audio-native-replay-v10/vectors.json"))


@pytest.fixture(scope="module")
def cases():
    if not NATIVE.exists() or not FIXTURES.exists():
        pytest.skip("save audio fixtures and native parity export")
    return {c["id"]:c for c in json.loads(NATIVE.read_text())["cases"]}


def content(case):
    return {"content":[{"type":"text","text":p["text"]} if "text" in p else
        media(FIXTURES/p[next(kind for kind in ("audio","image","video") if kind in p)]) for p in case["content"]]}


def media(path):
    mime = {".wav":"audio/wav",".mp3":"audio/mpeg",".flac":"audio/flac",".png":"image/png",".mp4":"video/mp4"}[path.suffix]
    kind = mime.split("/")[0]+"_url"
    return {"type":kind,kind:{"url":f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()}}


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("audio-embedding-http")
    servers = []
    def create(audio=True, vision=True, **options):
        server = Server(directory,bundle=BUNDLE,**({"audio":True} if audio else {}),
            **({"vision":True} if vision else {}),**options)
        servers.append(server)
        return server
    yield create
    for server in reversed(servers):
        server.close()


@pytest.fixture(scope="module")
def server(factory):
    return factory()


@pytest.mark.parametrize("dimension",[128,256,512,768])
def test_all_audio_cases_match_native_exactly(server, cases, dimension):
    ordered = list(cases.values())
    for begin in range(0,len(ordered),2):
        group = ordered[begin:begin+2]
        status,result = server.embed([content(c) for c in group],dimensions=dimension,
            mm_processor_kwargs={"max_soft_tokens":70})
        assert status == 200, result
        assert [r["index"] for r in result["data"]] == list(range(len(group)))
        assert [r["embedding"] for r in result["data"]] == [c["embeddings"][str(dimension)] for c in group]
        assert result["usage"] == dict.fromkeys(("prompt_tokens","total_tokens"),sum(len(c["token_ids"]) for c in group))


def test_base64_interleaving_and_separate_media_metrics(server, cases):
    group = [cases[name] for name in ("query-dog","interleaved","two-audios")]
    counters = ("audios_total","audio_soft_tokens_total","images_total","image_soft_tokens_total",
        "videos_total","video_frames_total","video_soft_tokens_total")
    before = {k:server.metric(k) for k in counters}
    status,result = server.embed([group[0]["text"],content(group[1]),content(group[2])],dimensions=256,
        encoding_format="base64",mm_processor_kwargs={"max_soft_tokens":70})
    assert status == 200, result
    assert [list(struct.unpack("<256f",base64.b64decode(r["embedding"]))) for r in result["data"]] == [c["embeddings"]["256"] for c in group]
    spans = [span for c in group for span in c["media_spans"]]
    expected = {"audios_total":sum(s["kind"]=="audio" for s in spans),
        "audio_soft_tokens_total":sum(s["end"]-s["begin"] for s in spans if s["kind"]=="audio"),
        "images_total":sum(s["kind"]=="image" for s in spans),
        "image_soft_tokens_total":sum(s["end"]-s["begin"] for s in spans if s["kind"]=="image"),
        "videos_total":sum(len(c["videos"]) for c in group),
        "video_frames_total":sum(s["kind"]=="video" for s in spans),
        "video_soft_tokens_total":sum(s["end"]-s["begin"] for s in spans if s["kind"]=="video")}
    assert {k:server.metric(k)-before[k] for k in counters} == expected
    assert server.metric("audio_weight_bytes") == 613490688
    assert server.metric("audio_scratch_bytes") == 849659120


def test_audio_independent_of_vision_flag_and_image_budget(factory, cases):
    audio_only = factory(vision=False)
    value = content(cases["dog-wav"])
    results = [audio_only.embed(value,mm_processor_kwargs={"max_soft_tokens":n}) for n in (70,280,1120)]
    assert results[0][0] == 200 and results[0] == results[1] == results[2]
    assert results[0][1]["data"][0]["embedding"] == cases["dog-wav"]["embeddings"]["768"]
    assert audio_only.metric("vision_weight_bytes") == 0
    assert audio_only.embed(content(cases["interleaved"]))[1]["error"]["code"] == "unsupported_parameter"
    disabled = factory(audio=False)
    assert disabled.metric("audio_weight_bytes") == disabled.metric("audio_scratch_bytes") == 0
    assert disabled.embed(value)[1]["error"]["code"] == "unsupported_parameter"
    assert disabled.embed("still works")[0] == 200


def test_audio_component_is_validated_at_startup():
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1",0));port=reservation.getsockname()[1]
    process = subprocess.run([str(ROOT/"build/gewell"),"serve-embeddings",
        str(ROOT/"artifacts/embeddinggemma2/native-bf16"),"--audio","--port",str(port)],capture_output=True,text=True,timeout=30)
    assert process.returncode != 0 and "audio" in (process.stdout+process.stderr).lower()


@pytest.mark.parametrize("part,suffix",[
    ({"type":"audio_url"},"audio_url"),
    ({"type":"audio_url","audio_url":[]},"audio_url"),
    ({"type":"audio_url","audio_url":{}},"audio_url.url"),
    ({"type":"audio_url","audio_url":{"url":1}},"audio_url.url"),
    ({"type":"audio_url","audio_url":{"url":"x","sample_rate":16000}},"audio_url.sample_rate"),
    ({"type":"audio_url","audio_url":{"url":"x"},"truncate":True},"truncate"),
])
def test_indexed_schema_errors_are_atomic(server, part, suffix):
    before = server.metric("inputs_total")
    status,result = server.embed(["valid",{"content":[part]}])
    assert status == 400 and result["error"]["param"] == "input[1].content[0]."+suffix, result
    assert server.metric("inputs_total") == before


@pytest.mark.parametrize("url",[
    "https://example.com/a.wav","/tmp/a.wav","data:audio/ogg;base64,AAAA",
    "data:audio/wav;base64,","data:audio/wav;base64,AB==","data:audio/wav;base64,AAAA",
    "data:audio/mpeg;base64,AAAA","data:audio/flac;base64,AAAA",
])
def test_invalid_audio_transport(server, url):
    status,result = server.embed({"content":[{"type":"audio_url","audio_url":{"url":url}}]})
    assert status == 400 and result["error"]["code"] == "invalid_audio", result
    assert result["error"]["param"] == "input.content[0].audio_url.url"


@pytest.mark.parametrize("name",["too-short-0.wav","too-short-1.wav","too-short-160.wav","nonfinite.wav","unsupported.wav","truncated.wav"])
def test_rejected_audio_is_atomic_and_releases_prepared_inputs(server, cases, name):
    before = server.metric("inputs_total")
    status,result = server.embed([content(cases["dog-wav"]),{"content":[media(FIXTURES/name)]}])
    assert status == 400 and result["error"]["code"] == "invalid_audio", result
    assert result["error"]["param"] == "input[1].content[0].audio_url.url"
    assert server.metric("inputs_total") == before


@pytest.mark.parametrize("rate,channels",[(24000,1),(16000,2)])
def test_midstream_format_change_rejected(server, cases, tmp_path, rate, channels):
    for name,hz,count in [("first",16000,1),("second",rate,channels)]:
        subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i",f"sine=sample_rate={hz}:duration=0.5",
            "-ac",str(count),"-c:a","libmp3lame","-write_xing","0","-id3v2_version","0",str(tmp_path/(name+".mp3"))],check=True)
    path=tmp_path/"changed.mp3";path.write_bytes((tmp_path/"first.mp3").read_bytes()+(tmp_path/"second.mp3").read_bytes())
    status,result=server.embed({"content":[media(path)]})
    assert status == 400 and result["error"]["code"] == "invalid_audio",result
    assert "format changes within the stream" in result["error"]["message"]
    assert server.embed(content(cases["samples-161-wav"]))[0] == 200


def test_data_url_size_bound(factory):
    bounded=factory(max_body_bytes=9*1024*1024,max_body_total_bytes=16*1024*1024)
    status,result=bounded.embed({"content":[{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,"+"A"*(8*1024*1024)}}]})
    assert status == 400 and result["error"]["code"] == "invalid_audio",result


def test_exact_context_limits_and_output_admission(factory, server, cases):
    case=cases["samples-161-wav"]
    bounded=factory(max_batch_tokens=5,max_inputs=2,max_output_bytes=10000)
    assert bounded.embed(content(case),dimensions=128)[0] == 200
    extra={"content":content(case)["content"]+[{"type":"text","text":"<mask>"}]}
    assert bounded.embed(extra,dimensions=128)[1]["error"]["code"] == "context_length_exceeded"
    assert bounded.embed(content(cases["samples-801-wav"]),dimensions=128)[1]["error"]["code"] == "context_length_exceeded"
    assert bounded.embed(content(case))[1]["error"]["code"] == "output_limit_exceeded"
    assert bounded.embed([content(case)]*3,dimensions=128)[0] == 400
    status,result=server.embed({"content":[media(FIXTURES/"context-over.flac")]})
    assert status == 400 and result["error"]["code"] == "context_length_exceeded",result
    for extra in (0,1):
        value={"content":[{"type":"text","text":"<mask>"*(8187+extra)}]+content(case)["content"]}
        status,result=server.embed(value,dimensions=128)
        assert status == (400 if extra else 200),result
        if extra:assert result["error"]["code"] == "context_length_exceeded"
        else:assert result["usage"]["total_tokens"] == 8192


def test_feature_budget_failure_and_partial_request_release(factory, cases):
    bounded=factory(max_body_total_bytes=750000)
    value=content(cases["dog-wav"])
    assert bounded.embed(value)[0] == 200
    status,result=bounded.embed([value,value])
    assert status == 503 and result["error"]["code"] == "capacity_exceeded",result
    assert bounded.embed(value)[0] == 200
    status,result=bounded.embed({"content":value["content"]+[{"type":"text","text":None}]})
    assert status == 400,result
    assert bounded.embed(value)[0] == 200


def test_audio_and_visual_features_share_one_budget(factory, cases):
    bounded=factory(max_body_total_bytes=4200000)
    audio=content(cases["chirp-30-flac"])
    image={"content":[media(FIXTURES/"red.png")]}
    for value in (audio,image):assert bounded.embed(value,mm_processor_kwargs={"max_soft_tokens":70})[0] == 200
    status,result=bounded.embed({"content":audio["content"]+image["content"]},mm_processor_kwargs={"max_soft_tokens":70})
    assert status == 503 and result["error"]["code"] == "capacity_exceeded",result
    assert bounded.embed(image,mm_processor_kwargs={"max_soft_tokens":70})[0] == 200


def test_active_audio_cancellation_overload_and_recovery(factory, cases):
    bounded=factory(max_pending=1)
    active=bounded.start([content(cases["context-limit-flac"])]*4)
    try:
        bounded.wait_metric("requests_running",lambda n:n==1)
        assert bounded.embed("busy")[0] == 503
    finally:active.close()
    bounded.wait_metric("requests_cancelled_total",lambda n:n==1)
    assert bounded.metric("inputs_total") < 4
    assert bounded.embed(content(cases["dog-wav"]))[0] == 200


def test_queued_audio_cancellation_releases_capacity(factory, cases):
    bounded=factory(max_pending=2,max_inputs=256,max_body_bytes=16*1024*1024,max_body_total_bytes=19*1024*1024)
    # Keep GPU work active while both full-context clips finish CPU preparation.
    active=bounded.start(["<mask>"*8190]*256)
    try:
        bounded.wait_metric("requests_running",lambda n:n==1)
        for count in (1,2):
            queued=bounded.start(content(cases["context-limit-flac"]))
            try:bounded.wait_metric("requests_waiting",lambda n:n==1)
            finally:queued.close()
            bounded.wait_metric("requests_cancelled_total",lambda n:n==count)
    finally:active.close()
    bounded.wait_metric("requests_cancelled_total",lambda n:n==3)
    assert bounded.embed(content(cases["context-limit-flac"]))[0] == 200


def test_preparation_disconnect_and_recovery(factory, cases):
    bounded=factory(max_body_total_bytes=19*1024*1024)
    active=bounded.start(content(cases["context-limit-flac"]))
    time.sleep(.02)
    assert bounded.metric("requests_running") == 0
    active.close()
    assert bounded.embed(content(cases["context-limit-flac"]))[0] == 200
    assert bounded.metric("inputs_total") == 1


def test_completed_slow_reader_releases_audio_features(factory, cases):
    bounded=factory(max_inputs=256,max_body_total_bytes=19*1024*1024)
    value=content(cases["context-limit-flac"])
    body=json.dumps({"model":MODEL,"input":[value]+["hello"]*255}).encode()
    client=socket.socket();client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024);client.settimeout(30)
    client.connect(("127.0.0.1",bounded.port))
    try:
        client.sendall(f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body)
        bounded.wait_metric("inputs_total",lambda n:n==256)
        bounded.wait_metric("requests_running",lambda n:n==0)
        assert bounded.metric("requests_completed_total") == 0,"response did not exercise backpressure"
        assert bounded.embed(value)[0] == 200
    finally:client.close()


@pytest.mark.parametrize("preparation",[True,False])
def test_shutdown_with_audio_work(tmp_path, cases, preparation):
    server=Server(tmp_path,bundle=BUNDLE,audio=True)
    active=server.start([content(cases["context-limit-flac"])]*(1 if preparation else 4))
    try:
        if preparation:
            time.sleep(.02)
            assert server.metric("requests_running") == 0
        else:server.wait_metric("requests_running",lambda n:n==1)
    finally:
        server.close();active.close()
