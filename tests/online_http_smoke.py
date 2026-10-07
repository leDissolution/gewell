#!/usr/bin/env python3
"""Opt-in native HTTP/OpenAI SDK smoke against a real local model bundle.

The client uses Python; its native child has no Python or HF cache on PATH.
This script does not package or modify the supplied model directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

if __package__:
    from . import native_log
else:
    import native_log


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def http_json(port, path, body=None, *, method=None, timeout=120):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(method or ("GET" if body is None else "POST"), path,
                           None if body is None else json.dumps(body),
                           {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), json.loads(response.read())
    finally:
        connection.close()


def control(port, path, body=None):
    status, _, payload = http_json(port, path, body)
    require(status == 200, f"{path}: HTTP {status}: {payload}")
    return payload


def usage_counts(usage):
    require(isinstance(usage, dict), "missing usage")
    prompt, completion = usage["prompt_tokens"], usage["completion_tokens"]
    require(type(prompt) is int and prompt > 0 and type(completion) is int and completion >= 0,
            f"invalid usage counts: {usage}")
    require(usage["total_tokens"] == prompt + completion, "usage total differs")
    require(0 <= usage["prompt_tokens_details"]["cached_tokens"] <= prompt,
            "cached usage exceeds prompt")
    return usage


def generation(client, model, *, chat=True, stream=False, include_usage=False,
               prompt="List the integers from 1 to 100, separated by commas.",
               count=32, cache=None, barrier=None):
    if barrier is not None:
        barrier.wait(30)
    body = {"model": model, "max_tokens": count, "temperature": 0, "stream": stream}
    if cache is not None:
        body["extra_body"] = {"cache": cache}
    if stream:
        body["stream_options"] = {"include_usage": include_usage}
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
        create = client.chat.completions.create
    else:
        body["prompt"] = prompt
        create = client.completions.create
    started = time.monotonic()
    response = create(**body)
    if not stream:
        payload = response.model_dump(exclude_unset=True)
        require(bool(response._request_id), "missing x-request-id header")
        choice = payload["choices"][0]
        usage = usage_counts(payload["usage"])
        require(1 <= usage["completion_tokens"] <= count, "generation exceeded token budget")
        return {"content": choice["message"]["content"] if chat else choice["text"],
                "reasoning": choice["message"].get("reasoning_content", "") if chat else "",
                "finish_reason": choice["finish_reason"], "usage": usage,
                "elapsed_seconds": time.monotonic() - started, "response": payload}

    chunks, content, reasoning = [], [], []
    identity, usage, finish_reason, first_text = None, None, None, None
    try:
        for chunk in response:
            payload = chunk.model_dump(exclude_unset=True)
            chunks.append(payload)
            current = (payload["id"], payload["created"], payload["model"])
            identity = current if identity is None else identity
            require(current == identity and current[2] == model, "stream identity changed")
            if not payload["choices"]:
                require(include_usage and usage is None, "unexpected aggregate usage chunk")
                usage = usage_counts(payload.get("usage"))
                continue
            if include_usage:
                require("usage" in payload and payload["usage"] is None, "ordinary chunk needs usage:null")
            else:
                require("usage" not in payload, "usage was emitted without include_usage")
            choice = payload["choices"][0]
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
            fields = choice["delta"] if chat else {"content": choice.get("text", "")}
            text = fields.get("content") or ""
            thought = fields.get("reasoning_content") or ""
            if (text or thought) and first_text is None:
                first_text = time.monotonic()
            content.append(text)
            reasoning.append(thought)
    finally:
        response.close()
    ended = time.monotonic()
    require(chunks and finish_reason in {"stop", "length"}, "stream lacks successful finish")
    require((usage is not None) == include_usage, "conditional stream usage differs")
    if usage is not None:
        require(1 <= usage["completion_tokens"] <= count, "stream exceeded token budget")
    if chat:
        require(chunks[0]["choices"][0]["delta"]["role"] == "assistant", "missing initial role delta")
    require(first_text is not None and first_text < ended, "stream produced no incremental text")
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "finish_reason": finish_reason, "usage": usage, "chunks": chunks,
            "first_text_seconds": first_text - started, "elapsed_seconds": ended - started}


def wait_idle(port, timeout):
    deadline = time.monotonic() + timeout
    while True:
        stats = control(port, "/v1/cache/stats")
        if stats["execution_count"] == 0:
            return stats
        require(time.monotonic() < deadline, "native cache executions did not drain")
        time.sleep(0.02)


def records(log_path):
    return [record for record in native_log.events(log_path)
            if record["kind"].startswith("server_http_")]


def wait_records(log_path, predicate, timeout, description):
    deadline = time.monotonic() + timeout
    while True:
        value = predicate(records(log_path))
        if value is not None:
            return value
        require(time.monotonic() < deadline, f"timed out waiting for {description}; see {log_path}")
        time.sleep(0.01)


def trace(log_path, begin):
    return [item["value"] for item in records(log_path)[begin:]
            if item["kind"] == "server_http_event"]


def shared_producer_disconnect(client, port, model, timeout, log_path):
    begin = len(records(log_path))
    prefix = [12403] + [902] * 4095
    producer = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    body = json.dumps({"model": model, "prompt": prefix + [903], "max_tokens": 64,
                       "temperature": 0, "stream": True, "cache": {"mode": "reuse_only"}}).encode()
    try:
        producer.sendall((f"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                          f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body)
        first = wait_records(log_path, lambda values: next((item["value"] for item in values[begin:]
                             if item["kind"] == "server_http_event" and item["value"]["event"] == "prefill"), None),
                             timeout, "producer prefill")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(generation, client, model, chat=False, prompt=prefix + [904 + index],
                                   count=8, cache={"mode": "reuse_only"}) for index in range(2)]
            def shared_progress(values):
                events = [item["value"] for item in values[begin:] if item["kind"] == "server_http_event"]
                joined = {event["request"] for event in events
                          if event["event"] == "join" and event["work"] == first["work"]}
                completed = [event for event in events if event["event"] == "prefill_complete"
                             and event["work"] == first["work"]]
                return completed[-1] if len(joined) == 2 and completed and completed[-1]["end"] < len(prefix) else None
            progress = wait_records(log_path, shared_progress, timeout, "survivors sharing unfinished prefill")
            producer.close()
            survivors = [future.result(timeout=timeout) for future in futures]
    finally:
        producer.close()
    idle = wait_idle(port, timeout)
    cancelled = wait_records(log_path, lambda values: next((item["value"] for item in values[begin:]
                             if item["kind"] == "server_http_cancelled"), None), timeout, "producer cancellation")
    require(cancelled["completion_tokens"] == 0, "producer disconnected after shared prefill finished")
    events = trace(log_path, begin)
    at = next(index for index, event in enumerate(events)
              if event["event"] == "cancel" and event["request"] == cancelled["id"])
    require(any(event["event"] == "prefill_complete" and event["work"] == progress["work"]
                and event["execution"] == progress["execution"] for event in events[at + 1:]),
            "disconnect did not preserve the shared native execution")
    require(not any(item["kind"] == "server_http_result" and item["value"]["id"] == cancelled["id"]
                    for item in records(log_path)[begin:]), "cancelled producer published success")
    return {"survivors": survivors, "cancelled": cancelled, "trace": events, "idle": idle}


def slow_reader(client, port, model, timeout, log_path):
    begin = len(records(log_path))
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    response = None
    try:
        connection.connect()
        connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "model": model, "messages": [{"role": "user", "content":
                "Write every integer from 1 to 10000, separated by commas. Do not abbreviate or explain."}],
            "max_tokens": 1024, "temperature": 0, "stream": True,
            "cache": {"mode": "reuse_only"},
        }), {"Content-Type": "application/json"})
        response = connection.getresponse()
        require(response.status == 200, "nonreading client failed before streaming")
        request_id = response.getheader("x-request-id").rsplit("-", 1)[1]
        # Leave the entire SSE body unread while another request completes.
        healthy = generation(client, model, chat=False, prompt="Hello", count=8,
                             cache={"mode": "reuse_only"})
        require(not any(item["kind"] in {"server_http_result", "server_http_cancelled"}
                        and item["value"]["id"] == request_id for item in records(log_path)[begin:]),
                "nonreading producer finished before the independent request")
    finally:
        if response is not None:
            response.close()
        connection.close()
    cancelled = wait_records(log_path, lambda values: next((item["value"] for item in values[begin:]
                             if item["kind"] == "server_http_cancelled" and item["value"]["id"] == request_id), None),
                             timeout, "nonreading client cancellation")
    require(cancelled["completion_tokens"] > 0, "nonreading client never generated a token")
    return {"healthy": healthy, "cancelled": cancelled, "idle": wait_idle(port, timeout)}


def cold_restore(client, port, model, timeout, log_path):
    reuse_only = {"mode": "reuse_only"}
    prefix = [12401] + [902] * 31
    untouched = generation(client, model, chat=False, prompt=prefix, count=4, cache=reuse_only)
    empty = wait_idle(port, timeout)
    require(empty["gpu"]["used_bytes"] == empty["cpu"]["used_bytes"] == 0,
            "reuse-only control retained a checkpoint")
    warm = control(port, "/v1/cache/prefill", {"model": model, "prompt": prefix,
                   "cache": {"prompt_id": "cold-owner"}})
    gpu = generation(client, model, chat=False, prompt=prefix, count=4, cache=reuse_only)
    generation(client, model, chat=False, prompt=[12402] + [903] * 1791, count=1, cache=reuse_only)
    spilled = wait_idle(port, timeout)
    require(spilled["gpu"]["used_bytes"] == 0 and spilled["cpu"]["used_bytes"] > 0,
            "pressure did not move the retained checkpoint entirely to CPU")
    before = [item["value"] for item in records(log_path) if item["kind"] == "server_http_stats"][-1]
    cpu = generation(client, model, chat=False, prompt=prefix, count=4, cache=reuse_only)
    wait_idle(port, timeout)
    after = wait_records(log_path, lambda values: next((item["value"] for item in reversed(values)
                         if item["kind"] == "server_http_stats" and
                         item["value"]["cold_restore_count"] > before["cold_restore_count"]), None),
                         timeout, "CPU restore telemetry")
    require(after["cold_restore_count"] == before["cold_restore_count"] + 1 and
            after["cold_restore_bytes"] > before["cold_restore_bytes"], "cold hit did not execute one restore")
    for value in (gpu, cpu):
        require(value["usage"]["prompt_tokens_details"]["cached_tokens"] == len(prefix),
                "GPU or CPU hit lost exact prefix credit")
        require((value["content"], value["finish_reason"], value["usage"]["completion_tokens"]) ==
                (untouched["content"], untouched["finish_reason"], untouched["usage"]["completion_tokens"]),
                "cache restore changed greedy generation")
    control(port, "/v1/cache/finish", {"prompt_id": "cold-owner"})
    final = wait_idle(port, timeout)
    require(final["gpu"]["used_bytes"] == final["cpu"]["used_bytes"] == final["checkpoint_count"] == 0,
            "finish leaked cold retained state")
    return {"untouched": untouched, "prefill": warm, "gpu": gpu, "spilled": spilled,
            "cpu": cpu, "before_restore": before, "after_restore": after, "final_stats": final}



def run_checks(client, port, model, timeout, log_path):
    report = {"health": control(port, "/health")}
    capacity = native_log.fields(log_path)["server_max_context_tokens"]
    models = client.models.list()
    require([item.id for item in models.data] == [model], "model discovery differs")
    require(client.models.retrieve(model).id == model, "model retrieval differs")
    report["models"] = models.model_dump()
    for chat in (True, False):
        buffered = generation(client, model, chat=chat)
        require(bool(buffered["content"]), "generation produced no visible answer")
        streams = [generation(client, model, chat=chat, stream=True, include_usage=include)
                   for include in (False, True)]
        for streamed in streams:
            require((streamed["content"], streamed["reasoning"], streamed["finish_reason"]) ==
                    (buffered["content"], buffered["reasoning"], buffered["finish_reason"]),
                    "greedy buffered/streamed output differs")
        report["chat" if chat else "completion"] = {"buffered": buffered, "streams": streams}

    errors = []
    for path, body, expected in (
        ("/v1/completions", {"prompt": "hello"}, 400),
        ("/v1/completions", {"model": "missing-model", "prompt": "hello"}, 404),
        ("/v1/completions", {"model": model, "prompt": []}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "temperature": True}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "temperature": 2.1}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "n": 2}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "logprobs": False}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "frequency_penalty": 1}, 400),
        ("/v1/completions", {"model": model, "prompt": "hello", "stop": [""]}, 400),
        ("/v1/completions", {"model": model, "prompt": [2, 1234], "max_tokens": capacity}, 400),
        ("/v1/chat/completions", {"model": model, "messages": [{"role": "tool", "content": "result"}]}, 400),
    ):
        status, headers, payload = http_json(port, path, body)
        require(status == expected, f"unexpected HTTP {status}: {payload}")
        require("error" in payload and payload["error"].get("code"), "missing structured error code")
        require(any(key.lower() == "x-request-id" for key in headers), "error lacks x-request-id")
        errors.append({"status": status, "body": payload})
    report["errors"] = errors

    # The SDK consumes [DONE], so inspect one stream's framing directly too.
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request("POST", "/v1/completions", json.dumps({
            "model": model, "prompt": "Hello", "temperature": 0, "max_tokens": 4, "stream": True,
        }), {"Content-Type": "application/json"})
        response = connection.getresponse()
        require(response.status == 200 and response.getheader("Content-Type").startswith("text/event-stream"),
                "raw stream lacks SSE headers")
        wire = response.read().decode("utf-8")
        require(wire.endswith("data: [DONE]\n\n"), "successful stream lacks terminal DONE event")
        report["raw_sse"] = wire
    finally:
        connection.close()

    prefix = [2, 12402] + [902] * 30
    owner = {"prompt_id": "native-http-smoke", "priority": "high"}
    warmed = control(port, "/v1/cache/prefill", {"model": model, "prompt": prefix, "cache": owner})
    require(warmed["retained"] and warmed["checkpoint_tokens"] == len(prefix), "prefill did not retain prompt")
    require(usage_counts(warmed["usage"])["completion_tokens"] == 0, "prefill generated tokens")
    cached = generation(client, model, chat=False, prompt=prefix, count=8, cache=owner)
    reused = generation(client, model, chat=False, prompt=prefix, count=8, cache={"mode": "reuse_only"})
    for value in (cached, reused):
        require(value["usage"]["prompt_tokens_details"]["cached_tokens"] == len(prefix), "prefill reuse missed")
    require(cached["content"] == reused["content"], "cache controls changed greedy text")
    finished = control(port, "/v1/cache/finish", {"prompt_id": owner["prompt_id"]})
    require(finished == {"prompt_id": owner["prompt_id"], "released": True}, "finish response differs")
    report["cache"] = {"prefill": warmed, "cached": cached, "reuse_only": reused, "finish": finished}

    barrier = threading.Barrier(2)
    samples = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(generation, client, model, stream=stream, include_usage=stream,
                               count=64, barrier=barrier) for stream in (False, True)]
        deadline = time.monotonic() + timeout
        while not all(future.done() for future in futures):
            require(time.monotonic() < deadline, "concurrent requests did not complete")
            samples.append(control(port, "/v1/cache/stats")["execution_count"])
            time.sleep(0.02)
        concurrent = [future.result() for future in futures]
    require(max(samples, default=0) >= 2, "concurrent requests never overlapped native execution")
    report["concurrent"] = {"responses": concurrent, "peak_executions": max(samples)}

    report["shared_producer_disconnect"] = shared_producer_disconnect(client, port, model, timeout, log_path)
    report["nonreading_client"] = slow_reader(client, port, model, timeout, log_path)

    # A partial HTTP body must release admission storage after disconnect.
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as partial:
        partial.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1024\r\n\r\n{")
    report["after_partial_body"] = generation(client, model, chat=False, count=4)
    report["final_stats"] = wait_idle(port, timeout)
    return report


@contextmanager
def native_server(binary, bundle, directory, model, batch, gpu_mib, mtp_depth, timeout, assistant=None):
    import httpx
    import openai

    directory.mkdir()
    empty = directory / "empty"
    empty.mkdir()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(binary), "--log-format", "json", "serve-http",
               "--model-dir", str(bundle), "--port", str(port), "--max-batch", str(batch),
               "--kv-cache-gpu-mib", str(gpu_mib), "--model", model, "--mtp-depth", str(mtp_depth),
               "--kv-cache-cpu-mib", "1024", "--kv-checkpoint-interval-tokens", "0"]
    if assistant is not None:
        command.extend(["--assistant", str(assistant.resolve())])
    environment = {**os.environ, "PATH": str(empty), "HF_HOME": str(empty),
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    (directory / "launch.json").write_text(json.dumps({"command": command}, indent=2) + "\n")
    log_path = directory / "server.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=directory)
        try:
            deadline = time.monotonic() + max(300, timeout)
            while True:
                require(process.poll() is None, f"native server exited during startup; see {log_path}")
                try:
                    if http_json(port, "/health", timeout=1)[0] == 200:
                        break
                except (OSError, http.client.HTTPException):
                    pass
                require(time.monotonic() < deadline, f"native server did not become ready; see {log_path}")
                time.sleep(0.1)
            with openai.OpenAI(api_key="unused", base_url=f"http://127.0.0.1:{port}/v1",
                               timeout=timeout, max_retries=0,
                               http_client=httpx.Client(trust_env=False, timeout=timeout)) as client:
                yield client, port, log_path
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError(f"native shutdown did not complete; see {log_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(os.environ.get("GEWELL_TEST_NATIVE_BINARY", "build/gewell")))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("GEWELL_TEST_NATIVE_ARTIFACT"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="gewell-http-smoke")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--kv-mib", type=int, default=8192)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--main-only", action="store_true")
    selection.add_argument("--cold-only", action="store_true")
    args = parser.parse_args(argv)
    if args.mtp_depth > 0 and args.assistant is None:
        parser.error("MTP checks require --assistant PATH")
    if args.model_dir is None:
        parser.error("--model-dir or GEWELL_TEST_NATIVE_ARTIFACT is required (an artifact directory)")
    require(args.max_batch >= 2 and args.kv_mib > 0, "smoke requires batch >=2 and positive KV MiB")
    require(math.isfinite(args.timeout) and args.timeout > 0, "timeout must be finite and positive")
    import openai

    binary, bundle, work = args.binary.resolve(), args.model_dir.resolve(), args.work_dir.resolve()
    require(binary.is_file() and (bundle / "manifest.json").is_file(), "native binary and model bundle must exist")
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest["schema_version"] == 6, "smoke requires a packaged schema-6 bundle")
    work.mkdir()
    metadata = {"openai_version": openai.__version__, "model_directory": str(bundle),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "model": manifest["model"], "serving": manifest["serving"], "artifact": manifest["artifact"]}
    (work / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started, report = time.monotonic(), {}
    cases = [("main", run_checks, args.max_batch, args.kv_mib, args.mtp_depth),
             ("cold", cold_restore, 2, 900, 0)]
    for name, check, batch, gpu_mib, mtp_depth in cases:
        if (args.main_only and name != "main") or (args.cold_only and name != "cold"):
            continue
        with native_server(binary, bundle, work / name, args.model, batch, gpu_mib,
                           mtp_depth, args.timeout, args.assistant) as (client, port, log_path):
            report[name] = check(client, port, args.model, args.timeout, log_path)
    report_path = work / "report.json"
    report_path.write_text(json.dumps({"status": "pass", "elapsed_seconds": time.monotonic() - started,
                                       "provenance": metadata, "cases": report}, indent=2) + "\n")
    print(json.dumps({"status": "pass", "report": str(report_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
