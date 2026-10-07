#!/usr/bin/env python3
"""Opt-in native HTTP stop, seed, and function-calling smoke with the real SDK.

The application executes only the local echo function defined below. Native
launch, offline child environment, and readiness checks share the stage-2 smoke.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from online_http_smoke import control, http_json, native_server, records, require, usage_counts, wait_idle, wait_records


ECHO = {"type": "function", "function": {
    "name": "echo", "description": "Return the supplied value unchanged.",
    "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                   "required": ["value"]},
}}


def collect(client, model, *, messages=None, prompt=None, stream=False, **options):
    chat = messages is not None
    body = {"model": model, "temperature": 0, "max_tokens": 256, **options,
            "stream": stream}
    body["messages" if chat else "prompt"] = messages if chat else prompt
    if stream:
        body["stream_options"] = {"include_usage": True}
    create = client.chat.completions.create if chat else client.completions.create
    response = create(**body)
    if not stream:
        payload = response.model_dump(exclude_unset=True)
        choice = payload["choices"][0]
        message = choice["message"] if chat else {"content": choice["text"]}
        return {"content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or "",
                "calls": message.get("tool_calls") or [], "message": message,
                "request_id": response._request_id,
                "finish_reason": choice["finish_reason"], "usage": usage_counts(payload["usage"])}

    content, reasoning, calls = [], [], {}
    identity, usage, finish = None, None, None
    try:
        for chunk in response:
            item = chunk.model_dump(exclude_unset=True)
            current = (item["id"], item["created"], item["model"])
            identity = current if identity is None else identity
            require(identity == current and current[2] == model, "SSE identity changed")
            if not item["choices"]:
                require(usage is None and finish is not None, "usage preceded completion or repeated")
                usage = usage_counts(item["usage"])
                continue
            require(item.get("usage", "missing") is None, "ordinary SSE chunk lacks usage:null")
            choice = item["choices"][0]
            require(finish is None, "choice arrived after terminal finish")
            delta = choice["delta"] if chat else {"content": choice.get("text", "")}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            for fragment in delta.get("tool_calls") or []:
                index = fragment["index"]
                require(type(index) is int and index >= 0, "invalid call index")
                call = calls.setdefault(index, {"id": "", "type": "function",
                                               "function": {"name": "", "arguments": ""}})
                if fragment.get("id"):
                    require(not call["id"] or call["id"] == fragment["id"], "call ID changed")
                    call["id"] = fragment["id"]
                require(fragment.get("type", "function") == "function", "non-function call delta")
                function = fragment.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason")
    finally:
        response.close()
    require(finish in {"stop", "length", "tool_calls"} and usage is not None,
            "SSE ended without a successful finish and usage")
    require(list(calls) == list(range(len(calls))), "call indices were not contiguous")
    return {"content": "".join(content), "reasoning": "".join(reasoning),
            "calls": list(calls.values()), "finish_reason": finish, "usage": usage}


def equivalent(left, right):
    for key in ("content", "reasoning", "finish_reason"):
        require(left[key] == right[key], f"buffered/SSE {key} differ")
    require([call["function"] for call in left["calls"]] ==
            [call["function"] for call in right["calls"]], "buffered/SSE arguments differ")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(left["usage"][key] == right["usage"][key], f"buffered/SSE {key} differ")


def tool_checks(client, model, report):
    values = ["café ☃ \"quoted\"", "second independent value"]
    one = [{"role": "user", "content":
        "Use echo exactly once with value " + json.dumps(values[0], ensure_ascii=False) +
        ". Do not answer until you receive the tool result."}]
    buffered = collect(client, model, messages=one, tools=[ECHO])
    report["one_call"] = buffered
    require(buffered["finish_reason"] == "tool_calls" and len(buffered["calls"]) == 1,
            "model quality: expected exactly one echo call")
    call = buffered["calls"][0]
    require(call["id"] and call["type"] == "function", "missing call identity")
    arguments = json.loads(call["function"]["arguments"])
    require(call["function"]["name"] == "echo" and arguments == {"value": values[0]},
            "model quality: echo name or arguments differ")
    streamed = collect(client, model, messages=one, tools=[ECHO], stream=True)
    report["one_call_stream"] = streamed
    equivalent(buffered, streamed)
    require(call["id"] != streamed["calls"][0]["id"], "call IDs reused across requests")

    # Execute a fixed harmless local implementation; never dispatch model names.
    result = arguments["value"]
    history = one + [{"role": "assistant", "content": buffered["content"] or None,
                      "tool_calls": buffered["calls"]},
                     {"role": "tool", "tool_call_id": call["id"], "content": result}]
    final = collect(client, model, messages=history, tools=[ECHO], tool_choice="none")
    report["roundtrip"] = final
    require(not final["calls"] and final["finish_reason"] == "stop" and final["content"],
            "model quality: tool result did not produce a final answer")
    require(values[0] in final["content"], "model quality: final answer lost the echoed value")

    multiple = [{"role": "user", "content":
        "Before answering, call echo twice in parallel, once for each of these two independent values: " +
        json.dumps(values, ensure_ascii=False) + ". Make both calls in this turn."}]
    parallel = collect(client, model, messages=multiple, tools=[ECHO], parallel_tool_calls=True)
    report["multiple_calls"] = parallel
    require(parallel["finish_reason"] == "tool_calls" and len(parallel["calls"]) == 2,
            "model quality: expected two independent echo calls")
    require(len({entry["id"] for entry in parallel["calls"]}) == 2, "duplicate call IDs")
    require(all(entry["function"]["name"] == "echo" for entry in parallel["calls"]),
            "model quality: unexpected function")
    require(sorted(json.loads(entry["function"]["arguments"])["value"] for entry in parallel["calls"]) ==
            sorted(values), "model quality: parallel call arguments differ")
    parallel_stream = collect(client, model, messages=multiple, tools=[ECHO], stream=True)
    report["multiple_calls_stream"] = parallel_stream
    equivalent(parallel, parallel_stream)

    plain = [{"role": "user", "content": "Do not call any function. Reply with exactly READY."}]
    for choice in ("auto", "none"):
        text = collect(client, model, messages=plain, tools=[ECHO], tool_choice=choice)
        report["no_call_" + choice] = text
        require(not text["calls"] and text["finish_reason"] == "stop" and text["content"],
                "model quality: expected a plain final answer")
    short = collect(client, model, messages=one, tools=[ECHO], max_tokens=1, stream=True)
    report["one_token_limit"] = short
    require(short["finish_reason"] == "length", "one token fabricated a complete tool handoff")


def constrained_tool_checks(client, model, report):
    from jsonschema import Draft202012Validator

    options = {"tools": [{"type": "function", "function": {"name": "ready", "strict": True}}],
               "tool_choice": "required", "parallel_tool_calls": False,
               "messages": [{"role": "user", "content": "Call ready with no arguments."}]}
    empty = collect(client, model, **options)
    report["no_arguments"] = empty
    require(empty["finish_reason"] == "tool_calls" and len(empty["calls"]) == 1 and
            json.loads(empty["calls"][0]["function"]["arguments"]) == {}, "no-argument strict call failed")
    equivalent(empty, collect(client, model, stream=True, **options))

    kilo = json.loads((Path(__file__).parent / "fixtures/kilo_tool.json").read_text())
    messages = [{"role": "user", "content":
        "Call todowrite with one pending, high priority task whose content is Check tools."}]
    buffered = collect(client, model, messages=messages, tools=[kilo])
    report["kilo"] = buffered
    require(buffered["finish_reason"] == "tool_calls" and len(buffered["calls"]) == 1,
            "model quality: expected a Kilo-style todowrite call")
    require(buffered["calls"][0]["function"]["name"] == "todowrite", "wrong Kilo-style function")
    Draft202012Validator(kilo["function"]["parameters"]).validate(
        json.loads(buffered["calls"][0]["function"]["arguments"]))
    equivalent(buffered, collect(client, model, messages=messages, tools=[kilo], stream=True))

    strict = {**ECHO, "function": {**ECHO["function"], "strict": True, "parameters": {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
        "properties": {"value": {"type": "string", "enum": ["café ☃"]}},
        "required": ["value"], "additionalProperties": False}}}
    other = {**strict, "function": {**strict["function"], "name": "other"}}
    messages = [{"role": "user", "content":
        "Call echo exactly once, with value café ☃. Do not answer before receiving its result."}]
    choices = {
        "auto": "auto", "required": "required",
        "named": {"type": "function", "function": {"name": "echo"}},
        "allowed": {"type": "allowed_tools", "allowed_tools": {"mode": "required", "tools": [
            {"type": "function", "function": {"name": "echo"}}]}},
    }
    for name, choice in choices.items():
        # allowed_tools is newer than some installed SDK type signatures.
        options = {"tools": [strict, other], "messages": messages,
                   "extra_body": {"tool_choice": choice}, "parallel_tool_calls": False}
        buffered = collect(client, model, **options)
        report[name] = buffered
        require(buffered["finish_reason"] == "tool_calls" and len(buffered["calls"]) == 1,
                f"{name}: did not finish with exactly one call")
        call = buffered["calls"][0]
        require(call["function"]["name"] == "echo", f"{name}: selected an unexpected tool")
        require(json.loads(call["function"]["arguments"]) == {"value": "café ☃"},
                f"{name}: strict arguments differ")
        equivalent(buffered, collect(client, model, stream=True, **options))

    # Enforce referenced nested schemas too, with ordinary SDK call/result history.
    kilo["function"]["strict"] = True
    nested = collect(client, model, tools=[kilo], tool_choice="required",
        parallel_tool_calls=False, messages=[{"role": "user", "content":
            "Call todowrite with a single pending, high priority task: Check tools."}])
    report["strict_reference"] = nested
    require(nested["finish_reason"] == "tool_calls" and len(nested["calls"]) == 1,
            "strict reference call did not complete")
    Draft202012Validator(kilo["function"]["parameters"]).validate(
        json.loads(nested["calls"][0]["function"]["arguments"]))

    answer = {"type": "object", "properties": {"answer": {"type": "string", "enum": ["blue"]}},
              "required": ["answer"], "additionalProperties": False}
    options = {"tools": [strict], "response_format": {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True, "schema": answer}}}
    tool = collect(client, model, messages=messages, tool_choice="required", **options)
    report["tool_with_answer_schema"] = tool
    require(tool["finish_reason"] == "tool_calls", "answer schema prevented a tool call")
    final = collect(client, model, tool_choice="none", messages=messages + [
        {"role": "assistant", "content": None, "tool_calls": tool["calls"]},
        {"role": "tool", "tool_call_id": tool["calls"][0]["id"], "content": "blue"},
        {"role": "user", "content": "Reply with the JSON answer blue."}], **options)
    report["answer_after_tool"] = final
    require(final["finish_reason"] == "stop" and not final["calls"], "JSON roundtrip did not complete")
    Draft202012Validator(answer).validate(json.loads(final["content"]))


def stop_seed_checks(client, port, model, timeout, report):
    prompt = "The numbers are: 1, 2, 3,"
    owner = {"prompt_id": "native-stop-seed-smoke", "priority": "high"}
    warmed = control(port, "/v1/cache/prefill", {"model": model, "prompt": prompt, "cache": owner})
    # Establish a known selected-token boundary without retokenizing model text.
    for count in range(1, 5):
        first = collect(client, model, prompt=prompt, max_tokens=count)
        if first["content"]:
            break
    require(first["content"], "model produced no visible raw text in four tokens")
    stop = first["content"][0]
    stopped = []
    for stream in (False, True):
        result = collect(client, model, prompt=prompt, max_tokens=16, stop=stop, stream=stream,
                         extra_body={"cache": {**owner, "finished": True}})
        require(result["content"] == "" and result["finish_reason"] == "stop", "stop leaked visible text")
        require(result["usage"]["completion_tokens"] == count, "stop selected tokens beyond first match")
        require(result["usage"]["prompt_tokens_details"]["cached_tokens"] == warmed["checkpoint_tokens"],
                "stopped request lost cached-prefix credit")
        stopped.append(result)
    equivalent(*stopped)
    reused = collect(client, model, prompt=prompt, max_tokens=count, extra_body={"cache": {"mode": "reuse_only"}})
    equivalent(first, reused)
    report["stop"] = {"match": stop, "selected_boundary": count, "prefill": warmed,
                      "responses": stopped, "subsequent_reuse": reused, "idle": wait_idle(port, timeout)}

    # Sequential requests keep batch shape fixed; another request consumes many
    # random draws between repeats. Cross-batch numerical invariance is excluded.
    samples = []
    for seed in (-(1 << 63), (1 << 63) - 1):
        options = {"prompt": "Invent a name for a small blue robot:", "max_tokens": 24,
                   "temperature": 0.9, "seed": seed}
        before = collect(client, model, **options)
        noise = collect(client, model, prompt="Write a whimsical story about a kite.",
                        max_tokens=64, temperature=1.3, seed=87)
        after = collect(client, model, **options)
        equivalent(before, after)
        samples.append({"seed": seed, "before": before, "noise": noise, "after": after})
    report["seed_independence"] = samples


def validation_checks(port, model):
    base = {"model": model, "messages": [{"role": "user", "content": "Hello"}], "tools": [ECHO]}
    call = {"id": "call_fixture", "type": "function",
            "function": {"name": "echo", "arguments": '{"value":"ok"}'}}
    assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
    result = {"role": "tool", "tool_call_id": call["id"], "content": "ok"}
    invalid = [
        {"tool_choice": "unknown"},
        {"tool_choice": {"type": "function", "function": {"name": "unknown"}}},
        {"parallel_tool_calls": "false"},
        {"tools": [{**ECHO, "function": {**ECHO["function"], "strict": True}}]},
        {"tools": [{**ECHO, "function": {**ECHO["function"], "strict": True}}], "tool_choice": "none"},
        {"stop": ""}, {"stop": []}, {"stop": ["a"] * 5}, {"stop": ["a", ""]},
        {"seed": True}, {"seed": 1.5}, {"seed": 1 << 63}, {"seed": -(1 << 63) - 1},
        {"tools": [ECHO, ECHO]},
        {"messages": [result]},
        {"messages": base["messages"] + [assistant]},
        {"messages": base["messages"] + [assistant, {**result, "tool_call_id": "missing"}]},
        {"messages": base["messages"] + [assistant, result, result]},
        {"messages": base["messages"] + [{**assistant, "tool_calls": [call, call]}, result]},
    ]
    for arguments in ("not json", "[]"):
        bad_call = {**call, "function": {**call["function"], "arguments": arguments}}
        invalid.append({"messages": base["messages"] + [{**assistant, "tool_calls": [bad_call]}, result]})
    errors = []
    for changes in invalid:
        status, _, payload = http_json(port, "/v1/chat/completions", {**base, **changes})
        require(status == 400 and payload.get("error", {}).get("param"), f"expected typed HTTP 400: {payload}")
        errors.append({"request": changes, "error": payload})
    return errors


def mixed_stop_mtp_check(client, port, model, timeout, log_path, mtp_depth):
    messages = [{"role": "user", "content":
        "Write every integer from 1 to 1000, separated by commas. Do not abbreviate or explain."}]
    # Use the same ordinary path as the stopped request to establish exactly
    # where selected token48 completes a unique visible match.
    probe_options = {"messages": messages, "stop": "⟦gewell_absent_stop_probe⟧"}
    previous = collect(client, model, max_tokens=47, **probe_options)
    baseline = collect(client, model, max_tokens=48, **probe_options)
    require(previous["finish_reason"] == baseline["finish_reason"] == "length" and
            baseline["content"].startswith(previous["content"]) and
            len(baseline["content"]) > len(previous["content"]),
            "model quality: mixed-batch probe lacks visible output at selected token48")
    text = baseline["content"]
    match = next((text[-size:] for size in range(8, min(48, len(text) - 1) + 1)
                  if text.find(text[-size:]) == len(text) - size and text[-size:] not in previous["content"]), None)
    require(match is not None, "model quality: mixed-batch probe has no unique late stop suffix")
    expected = text[:-len(match)]
    owner = {"prompt_id": "native-mixed-stop-mtp", "priority": "high"}
    warmed = control(port, "/v1/cache/prefill", {"model": model, "messages": messages, "cache": owner})
    wait_idle(port, timeout)
    before = records(log_path)
    histogram = next((entry["value"]["decode_batch_histogram"] for entry in reversed(before)
                      if entry["kind"] == "server_http_stats"), {})
    barrier = threading.Barrier(3)

    def run(**options):
        barrier.wait(timeout)
        return collect(client, model, **options)

    samples = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        stopped_future = pool.submit(run, messages=messages, max_tokens=96, stop=match,
                                     extra_body={"cache": {**owner, "finished": True}})
        neighbor_future = pool.submit(run, messages=messages,
                                     max_tokens=512, extra_body={"cache": {"mode": "reuse_only"}})
        barrier.wait(timeout)
        deadline = time.monotonic() + timeout
        while not stopped_future.done():
            require(time.monotonic() < deadline, "mixed-batch stop did not complete")
            samples.append(control(port, "/v1/cache/stats")["execution_count"])
            time.sleep(0.01)
        stopped = stopped_future.result()
        neighbor = neighbor_future.result(timeout=timeout)
    require(max(samples, default=0) >= 2, "mixed requests never overlapped native execution")
    require(stopped["finish_reason"] == "stop" and stopped["content"] == expected,
            "mixed-batch stop leaked text or changed its greedy prefix")
    require(stopped["usage"]["completion_tokens"] == 48, "mixed-batch stop crossed its selected-token boundary")
    require(stopped["usage"]["prompt_tokens_details"]["cached_tokens"] == warmed["checkpoint_tokens"],
            "mixed stopped request lost cached-prefix credit")
    request_ids = [value["request_id"].rsplit("-", 1)[1] for value in (stopped, neighbor)]

    def completed(values):
        selected = {entry["value"]["id"]: entry["value"] for entry in values[len(before):]
                    if entry["kind"] == "server_http_result" and entry["value"]["id"] in request_ids}
        return selected if len(selected) == 2 else None

    terminal = wait_records(log_path, completed, timeout, "mixed-batch terminal results")
    stop_record, neighbor_record = [terminal[identity] for identity in request_ids]
    require(stop_record["mtp_depth"] == stop_record["mtp_proposed"] == 0,
            "stop request used speculative proposals")
    require(neighbor_record["mtp_depth"] == mtp_depth and neighbor_record["mtp_proposed"] > 0,
            "un-stopped neighbor did not retain MTP")
    require(stop_record["completion_tokens"] == 48 and stop_record["finish_reason"] == "stop" and
            stop_record["processed_tokens"] == stop_record["prompt_tokens"] + 47,
            "mixed stopped request committed the wrong token prefix")
    idle = wait_idle(port, timeout)
    after = records(log_path)
    mixed_batches = max((entry["value"]["decode_batch_histogram"].get("2", 0)
                         for entry in after[len(before):] if entry["kind"] == "server_http_stats"), default=0)
    # Actual shared decode progress proves overlap even if the un-stopped
    # neighbor chooses EOS early or its HTTP response arrives first.
    require(mixed_batches - histogram.get("2", 0) >= 2, "mixed requests did not share multiple decode batches")
    require(not any(entry["kind"] == "server_http_cancelled" and entry["value"]["id"] in request_ids
                    for entry in after[len(before):]), "mixed completion became cancellation")
    reused = collect(client, model, max_tokens=48, **probe_options,
                     extra_body={"cache": {"mode": "reuse_only"}})
    equivalent(baseline, reused)
    require(reused["usage"]["prompt_tokens_details"]["cached_tokens"] == warmed["checkpoint_tokens"],
            "mixed completion broke subsequent prefix reuse")
    return {"match": match, "expected_selected_tokens": 48, "baseline": baseline,
            "prefill": warmed, "stopped": stopped, "neighbor": neighbor,
            "native_results": terminal, "peak_executions": max(samples),
            "shared_decode_batches": mixed_batches - histogram.get("2", 0),
            "subsequent_reuse": reused, "idle": idle}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path(os.environ.get("GEWELL_TEST_NATIVE_BINARY", "build/gewell")))
    parser.add_argument("--model-dir", type=Path, default=os.environ.get("GEWELL_TEST_NATIVE_ARTIFACT"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model", default="gewell-tools-smoke")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--kv-mib", type=int, default=8192)
    parser.add_argument("--assistant", type=Path)
    parser.add_argument("--mtp-depth", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if args.mtp_depth > 0 and args.assistant is None:
        parser.error("MTP checks require --assistant PATH")
    if args.model_dir is None:
        parser.error("--model-dir or GEWELL_TEST_NATIVE_ARTIFACT is required")
    require(args.max_batch > 0 and args.kv_mib > 0 and math.isfinite(args.timeout) and args.timeout > 0,
            "batch, KV budget, and timeout must be positive")
    require(not args.mtp_depth or args.max_batch >= 2, "MTP smoke requires batch capacity >=2")
    import openai

    binary, bundle, work = args.binary.resolve(), args.model_dir.resolve(), args.work_dir.resolve()
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(binary.is_file() and manifest["schema_version"] == 6, "native binary and schema-6 bundle required")
    work.mkdir()
    provenance = {"openai_version": openai.__version__, "model_directory": str(bundle),
                  "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "model": manifest["model"], "serving": manifest["serving"], "artifact": manifest["artifact"]}
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = {"status": "running", "provenance": provenance, "model_quality": {}}
    started = time.monotonic()
    try:
        with native_server(binary, bundle, work / "native", args.model, args.max_batch,
                           args.kv_mib, args.mtp_depth, args.timeout, args.assistant) as (client, port, log_path):
            report["validation"] = validation_checks(port, args.model)
            stop_seed_checks(client, port, args.model, args.timeout, report)
            if args.mtp_depth:
                report["mixed_stop_mtp"] = mixed_stop_mtp_check(
                    client, port, args.model, args.timeout, log_path, args.mtp_depth)
            tool_checks(client, args.model, report["model_quality"])
            report["constrained_tools"] = {}
            constrained_tool_checks(client, args.model, report["constrained_tools"])
            if args.mtp_depth:
                identity = report["constrained_tools"]["required"]["request_id"].rsplit("-", 1)[1]
                terminal = wait_records(log_path, lambda values: next((entry["value"] for entry in values
                    if entry["kind"] == "server_http_result" and entry["value"]["id"] == identity), None),
                    args.timeout, "constrained tool MTP result")
                require(terminal["mtp_depth"] == args.mtp_depth and terminal["mtp_proposed"] > 0,
                        "strict tool request did not retain speculative decoding")
                report["constrained_tool_mtp"] = terminal
            report["final_stats"] = wait_idle(port, args.timeout)
        report["status"] = "pass"
    except Exception as error:
        report.update(status="fail", error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (work / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "pass", "report": str(work / "report.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
