"""Real-weight structured image endpoint and bounded-lifecycle acceptance."""
import base64
import json
import os
from pathlib import Path
import socket
import struct
import time

import pytest

from tests.test_embeddinggemma2_http import Server, ROOT, MODEL

pytestmark = pytest.mark.skipif(os.environ.get("GEWELL_EMBEDDINGGEMMA2_HTTP") != "1", reason="requires SM120 GPU and saved image bundle")
BUNDLE = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_IMAGE_BUNDLE",ROOT/"artifacts/embeddinggemma2/native-bf16-image"))
FIXTURES = ROOT/"artifacts/embeddinggemma2/image-fixtures-v1"
NATIVE = Path(os.environ.get("GEWELL_EMBEDDINGGEMMA2_IMAGE_EXPORT",ROOT/"artifacts/embeddinggemma2/image-native-audio-regression-v1/vectors.json"))


@pytest.fixture(scope="module")
def cases():
    if not NATIVE.exists() or not FIXTURES.exists():
        pytest.skip("save image fixtures and native parity export")
    return {c["id"]:c for c in json.loads(NATIVE.read_text())["cases"]}


@pytest.fixture(scope="module")
def factory(tmp_path_factory):
    directory=tmp_path_factory.mktemp("image-embedding-http")
    servers=[]
    def create(vision=True,**options):
        server=Server(directory,bundle=BUNDLE,**({"vision":True} if vision else {}),**options)
        servers.append(server)
        return server
    yield create
    for server in reversed(servers):
        server.close()


@pytest.fixture(scope="module")
def server(factory):
    return factory()


def content(case):
    parts=[]
    for part in case["content"]:
        if "text" in part:
            parts.append({"type":"text","text":part["text"]})
        else:
            path=FIXTURES/part["image"]
            mime="image/jpeg" if path.suffix==".jpg" else "image/png"
            url=f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()
            parts.append({"type":"image_url","image_url":{"url":url}})
    return {"content":parts}


def image(cases,name="red"):
    return content(cases[name])


@pytest.mark.parametrize("dimension",[128,256,512,768])
def test_every_native_case_matches_http_exactly(server,cases,dimension):
    for budget in (70,140,280,560,1120):
        group=[c for c in cases.values() if c["max_soft_tokens"]==budget]
        status,result=server.embed([content(c) for c in group],dimensions=dimension,mm_processor_kwargs={"max_soft_tokens":budget})
        assert status==200,result
        assert result["usage"]["total_tokens"]==sum(len(c["token_ids"]) for c in group)
        assert [r["index"] for r in result["data"]]==list(range(len(group)))
        assert [r["embedding"] for r in result["data"]]==[c["embeddings"][str(dimension)] for c in group]


def test_mixed_batch_base64_and_accounting(server,cases):
    values=[cases["query-red"]["text"],image(cases,"red"),image(cases,"blue")]
    images,soft=server.metric("images_total"),server.metric("image_soft_tokens_total")
    status,result=server.embed(values,dimensions=256,encoding_format="base64")
    assert status==200,result
    vectors=[list(struct.unpack("<256f",base64.b64decode(row["embedding"]))) for row in result["data"]]
    assert vectors==[cases[name]["embeddings"]["256"] for name in ("query-red","red","blue")]
    assert server.metric("images_total")==images+2
    assert server.metric("image_soft_tokens_total")==soft+512
    assert server.metric("vision_weight_bytes")==335806464
    assert server.metric("vision_scratch_bytes")==408943424


def test_adjacent_text_and_empty_parts_keep_verbatim_semantics(server):
    parts=["", "hel", "lo 日本", "語 <bos>", "<eos>"]
    plain=server.embed("".join(parts))[1]
    structured=server.embed({"content":[{"type":"text","text":p} for p in parts]})[1]
    assert structured==plain
    assert server.embed({"content":[{"type":"text","text":""}]})[0]==400
    for kind in ("image","audio","video"):
        assert server.embed({"content":[{"type":"text","text":"<|"},{"type":"text","text":kind+"|>"}]})[0]==400


@pytest.mark.parametrize("bad,suffix",[
    ({"content":[],"extra":1},"extra"),
    ({"content":[]},"content"),
    ({"content":[{"type":"text","text":"x","extra":1}]},"content[0].extra"),
    ({"content":[{"type":"text","text":None}]},"content[0].text"),
    ({"content":[{"type":"audio_url"}]},"content[0].audio_url"),
    ({"content":[{"type":"image_url","image_url":{"url":"x","detail":"auto"}}]},"content[0].image_url.detail"),
    ({"content":[{"type":"image_url","image_url":{}}]},"content[0].image_url.url"),
    ({"content":[{"type":"image_url","image_url":{"url":1}}]},"content[0].image_url.url"),
])
def test_indexed_errors_are_atomic(server,bad,suffix):
    before=server.metric("inputs_total")
    status,result=server.embed(["valid",bad])
    assert status==400,result
    assert result["error"]["param"]=="input[1]."+suffix
    assert server.metric("inputs_total")==before


@pytest.mark.parametrize("url",["data:image/png;base64,","https://example.com/image.png","/tmp/image.png","data:image/gif;base64,AAAA","data:image/jpeg;base64,AAAA"])
def test_invalid_image_transport(server,url):
    status,result=server.embed({"content":[{"type":"image_url","image_url":{"url":url}}]})
    assert status==400 and result["error"]["code"]=="invalid_image",result
    assert result["error"]["param"]=="input.content[0].image_url.url"


@pytest.mark.parametrize("options",[None,[],{"unknown":1},{"max_soft_tokens":True},{"max_soft_tokens":70.0},{"max_soft_tokens":71},{"max_soft_tokens":2**64-1}])
def test_processor_kwargs_checked_on_plain_text(server,options):
    assert server.embed("hello",mm_processor_kwargs=options)[0]==400


def test_disabled_vision_and_model_alias(factory,cases):
    disabled=factory(vision=False)
    status,result=disabled.embed(image(cases))
    assert status==400 and result["error"]["code"]=="unsupported_parameter"
    assert disabled.embed("hello",mm_processor_kwargs={"max_soft_tokens":70})[0]==200
    assert disabled.metric("vision_weight_bytes")==0
    aliased=factory(model="images")
    assert aliased.embed(image(cases))[0]==404
    assert aliased.embed(image(cases),model="images")[0]==200


def test_expanded_context_and_output_limits(factory,cases):
    bounded=factory(max_batch_tokens=68,max_output_bytes=10000,max_inputs=2)
    value=image(cases,"red-70")
    fields={"mm_processor_kwargs":{"max_soft_tokens":70},"dimensions":128}
    assert bounded.embed(value,**fields)[0]==200
    over={"content":[{"type":"text","text":"<mask>"}]+value["content"]}
    status,result=bounded.embed(over,**fields)
    assert status==400 and result["error"]["code"]=="context_length_exceeded"
    assert bounded.embed(value,mm_processor_kwargs={"max_soft_tokens":70})[0]==400
    assert bounded.embed([value]*3,**fields)[0]==400
    full=factory()
    for extra,expected in ((0,200),(1,400)):
        request={"content":[{"type":"text","text":"<mask>"*(8124+extra)}]+value["content"]}
        status,result=full.embed(request,**fields)
        assert status==expected,result
        if extra: assert result["error"]["code"]=="context_length_exceeded"
        else: assert result["usage"]["total_tokens"]==8192


def test_prepared_memory_exhaustion_and_release(factory,cases):
    bounded=factory(max_body_total_bytes=9*1024*1024)
    value=image(cases)
    status,result=bounded.embed([value,value])
    assert status==503 and result["error"]["code"]=="capacity_exceeded",result
    # The failed request must release its first prepared image's lease.
    assert bounded.embed(value)[0]==200
    broken={"content":[value["content"][0],{"type":"text","text":None}]}
    assert bounded.embed(broken)[0]==400
    assert bounded.embed(value)[0]==200


def test_image_cancel_overload_and_reuse(factory,cases):
    server=factory(max_pending=1)
    value=image(cases,"red-1120")
    # Default budget is280;16 images retain about124MB and keep execution active
    # long enough for the owner to observe the disconnect at a vision boundary.
    active=server.start([value]*16)
    try:
        server.wait_metric("requests_running",lambda n:n==1)
        assert server.embed("busy")[0]==503
    finally:
        active.close()
    server.wait_metric("requests_cancelled_total",lambda n:n==1)
    assert server.metric("inputs_total")<16
    assert server.embed(value)[0]==200
    assert server.request("GET","/health")[0]==200


def test_preparation_disconnect_releases_image_memory(factory,cases):
    server=factory(max_body_total_bytes=40*1024*1024)
    fields={"mm_processor_kwargs":{"max_soft_tokens":1120}}
    # One prepared image uses about31MB. Disconnect while the bounded worker
    # prepares a large-budget image; a new one must fit after its lease retires.
    active=server.start(image(cases),**fields)
    time.sleep(.02)
    active.close()
    deadline=time.monotonic()+10
    while True:
        status,result=server.embed(image(cases),**fields)
        if status==200:
            break
        assert status==503 and result["error"]["code"]=="capacity_exceeded",result
        assert time.monotonic()<deadline,"preparation lease did not retire"
        time.sleep(.01)


def test_completed_slow_reader_does_not_retain_image_memory(factory,cases):
    server=factory(max_inputs=256)
    fields={"mm_processor_kwargs":{"max_soft_tokens":1120}}
    # Eight images reserve about248MB. Text inputs make the response larger
    # than the socket send buffer without consuming additional image memory.
    body=json.dumps({"model":MODEL,"input":[image(cases)]*8+["hello"]*248,**fields}).encode()
    client=socket.socket()
    client.setsockopt(socket.SOL_SOCKET,socket.SO_RCVBUF,1024)
    client.settimeout(30)
    client.connect(("127.0.0.1",server.port))
    try:
        client.sendall(f"POST /v1/embeddings HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(body)}\r\n\r\n".encode()+body)
        server.wait_metric("inputs_total",lambda n:n==256)
        server.wait_metric("requests_running",lambda n:n==0)
        assert server.metric("requests_completed_total")==0,"response did not exercise backpressure"
        status,result=server.embed(image(cases),**fields)
        assert status==200,result
    finally:
        client.close()


def test_queued_image_cancellation_releases_capacity(factory,cases):
    server=factory(max_pending=2,max_body_total_bytes=10*1024*1024)
    active=server.start(["<mask>"*8190]*32)
    try:
        server.wait_metric("requests_running",lambda n:n==1)
        queued=server.start(image(cases))
        try:
            server.wait_metric("requests_waiting",lambda n:n==1)
        finally:
            queued.close()
        server.wait_metric("requests_cancelled_total",lambda n:n==1)
        # A second image can be prepared only if the queued image's lease was released.
        replacement=server.start(image(cases))
        try:
            server.wait_metric("requests_waiting",lambda n:n==1)
        finally:
            replacement.close()
        server.wait_metric("requests_cancelled_total",lambda n:n==2)
    finally:
        active.close()
    server.wait_metric("requests_cancelled_total",lambda n:n==3)
    assert server.embed(image(cases))[0]==200


def test_shutdown_during_image_execution(tmp_path,cases):
    server=Server(tmp_path,bundle=BUNDLE,vision=True)
    active=server.start([image(cases)]*16)
    try:
        server.wait_metric("requests_running",lambda n:n==1)
        server.close()
    finally:
        active.close()
