# Launch and first request

Start with a [built executable](build.md) and a complete [model bundle](models.md).
For a BF16 bundle on a 96 GiB RTX PRO 6000:

```bash
MODEL_DIR=/absolute/path/to/gemma-4-31b-bundle
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 1 \
  --kv-cache-gpu-mib 8192 --model gemma-4-31b
```

The listener defaults to `127.0.0.1:6311`. The named values set the maximum
active batch size and GPU KV-cache budget in MiB. This example reserves 8 GiB
for cache with one active request. Weights and executor scratch require
additional VRAM. Startup prints memory accounting, context capacity, and
readiness; reduce batch/cache capacity if the chosen bundle does not fit.

In another terminal, wait for readiness and inspect the model:

```bash
curl --fail http://127.0.0.1:6311/health
curl --fail http://127.0.0.1:6311/v1/models
```

Then request a streamed response:

```bash
curl --fail-with-body --no-buffer http://127.0.0.1:6311/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"Explain what a GPU does in two sentences."}],"max_tokens":128,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

The response is Server-Sent Events, ending with `data: [DONE]`. Use
`"stream":false` for one JSON response. Client requests must use the model
name supplied by `--model`.

Clients that support an OpenAI-compatible Chat Completions endpoint can use
`http://127.0.0.1:6311/v1` as their base URL. The native listener binds to
loopback by default and has no authentication configuration. `--host` accepts
an alternate IPv4 bind address; non-loopback exposure requires appropriate
network controls. See the [HTTP subset](http-api.md) for supported controls and
transport limits.

## Runtime choices

For concurrent requests and optional prefix spill to CPU memory:

```bash
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 4 \
  --kv-cache-gpu-mib 8192 --model gemma-4-31b --kv-cache-cpu-mib 1024
```

Batch capacity is a limit on concurrent active executions. Feasible context
length per request changes with concurrency and cache use; it is not fixed
by batch capacity alone.

Add `--assistant /path/to/model.safetensors --mtp-depth 3` to enable three assistant proposals per speculative cycle.
Add `--vision /path/to/vision.safetensors` to enable image input. Omitted
components are not loaded; a positive depth without an assistant warns and
falls back to zero.
MTP defaults to zero; its speed depends on acceptance rate and workload, and
its staging buffers count against the configured memory budget.

Images default to a budget of 280 soft tokens each. Add
`--image-max-soft-tokens 1120` after `serve-http` for a larger server default,
or select a budget per request with
`"mm_processor_kwargs":{"max_soft_tokens":1120}`. Supported budgets are
70, 140, 280, 560, and 1120; requests can override the default in either
direction. Larger budgets use more time and memory. See
[image controls and capacity](http-api.md#images) for examples and buffer limits.

Use `--kv-local-format fp8 --kv-global-format fp8` to opt into compressed KV
storage. `--attention-global-compute fp8` separately opts into FP8 global
attention matmuls for text prefill, decode, and both MTP paths. The independent
`--attention-local-compute` option covers local attention in those same phases.
Image-containing prefill chunks retain BF16 attention. Both choices can change
outputs; validate quality for your workload. Weight precision comes from the
artifact itself.

## Logs and shutdown

Console logs are readable text by default. Add `--log-format json` for JSON
Lines. `run-jobs` reserves stdout for its protocol and sends diagnostics to
stderr; ordinary commands and HTTP serving log to stdout with errors on stderr.

`GET /metrics` exposes Prometheus metrics. `GET /v1/cache/index` reports retained
prefixes and active executions. Stop the server with Ctrl-C or SIGTERM; it
cancels pending work and releases GPU resources before exiting.
