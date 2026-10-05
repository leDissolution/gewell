# Command-line interface

`--assistant PATH` selects the original assistant safetensors file. It is loaded
only for positive `--mtp-depth`; a positive depth without the path warns and
falls back to zero. `--vision PATH` selects the extracted vision + projector
safetensors file. Without it, HTTP image input returns 400, offline image jobs
are rejected, and `caption` fails. Both paths are optional for text generation
and may precede or follow the command. Unused components consume no weight VRAM.


Run `gewell --help` for the command list. Examples use `build/gewell` from a
source checkout; an installed executable accepts the same arguments.

`MODEL_DIR` is a complete serving bundle. `ARTIFACT` is its native weight file.
Integer memory budgets ending in `_MIB` or `-mib` use 1,048,576-byte MiB.
Options take separate values: `--mtp-depth 3`, not `--mtp-depth=3`.

## Commands

| Command | Arguments and purpose |
|---|---|
| `serve-http` | `--model-dir DIR --max-batch N --kv-cache-gpu-mib N [OPTIONS]` — persistent HTTP server |
| `generate` | `ARTIFACT PROMPT.u32 NEW_TOKENS OUTPUT.u32 [LOGITS.bf16]` — one token-based generation |
| `caption` | `ARTIFACT PROMPT.u32 PIXELS.f32 POSITIONS.i32 MAX_NEW_TOKENS OUTPUT.u32 [LOGITS.bf16]` — one prepared image |
| `generate-batch` | `ARTIFACT REQUESTS.tsv MAX_BATCH KV_MIB OUTPUT_DIR [EVENTS.tsv]` — offline request queue |
| `run-jobs` | `ARTIFACT MAX_BATCH KV_MIB [OPTIONS]` — persistent JSONL jobs on stdin/stdout |
| `replay-rollout` | `ARTIFACT REQUESTS.tsv CHUNK_ROWS HEAD_ROWS OUTPUT_DIR` — teacher-forced replay and logit comparison |
| `inspect` | `PATH` — weight header and tensor-table metadata |
| `verify` | `PATH` — full payload hash and padding verification |
| `probe` | `PATH` — full verification and GPU tensor-access checks |
| `load` | `PATH` — measure GPU weight loading |
| `text-codec` | `MODEL_DIR` — JSONL prompt rendering, tokenization, and decoding |

`inspect` and `text-codec` perform no GPU work, but the executable still needs
its linked native libraries. The [offline guide](offline.md) describes token
files and request formats. Prefer HTTP for ordinary text and image input.

## Option placement

`--log-format human|json` is global and can appear before or after a command.
`--help` and `-h` apply to the executable, not individual subcommands.

Place `serve-http` options after the command and `run-jobs` options after its
positional arguments. For `generate`, `caption`, and `generate-batch`, place
applicable generation/execution options before the command:

```bash
build/gewell serve-http --model-dir MODEL_DIR --max-batch 4 --kv-cache-gpu-mib 8192 \
  --assistant ASSISTANT.safetensors --mtp-depth 3 --model gemma-4-31b
build/gewell --temperature 0.8 --top-p 0.95 --seed 73 \
  generate ARTIFACT prompt.u32 128 output.u32
build/gewell --assistant ASSISTANT.safetensors --mtp-depth 3 --kv-global-format fp8 \
  generate-batch ARTIFACT requests.tsv 4 8192 output-directory
```

Sampling parameters for HTTP and batch/jobs are supplied per request.

## Execution and cache

The first seven options below apply to serving, jobs, generation, captioning,
and offline batches. The physical prefill batch cap, logical prefill budget, minimum MTP depth, decode
width, and MTP statistics apply only to serving, jobs, and offline batches.
Cache budget/checkpoint options apply to serving and jobs.

| Option | Default | Meaning |
|---|---|---|
| `--mtp-depth N` | `0` | Maximum assistant proposal depth, `0..1279`; zero disables MTP |
| `--prefill-chunk-tokens N` | `1024` | Per-prompt text chunk cap, `1..4096`; images retain separate complete spans |
| `--nvfp4-activation-policy POLICY` | `always` | `always` or `prefill`; the latter keeps NVFP4 decode/MTP activations BF16 |
| `--kv-local-format FORMAT` | `bf16` | Local KV storage: `bf16` or `fp8` |
| `--kv-global-format FORMAT` | `bf16` | Compact global KV storage: `bf16` or `fp8` |
| `--attention-local-compute FORMAT` | `bf16` | Local text attention (prefill, decode, MTP): `bf16` or `fp8` |
| `--attention-global-compute FORMAT` | `bf16` | Global text attention (prefill, decode, MTP): `bf16` or `fp8` |
| `--prefill-batch-tokens N` | `2048` | Combined text/image projection/MLP rows per forward, `1..4096`, at least the per-prompt chunk cap |
| `--prefill-budget-tokens N` | `0` | Soft prefill token budget between decode opportunities; 0 runs decode after each packed forward/head |
| `--mtp-min-depth N` | `0` | Minimum adaptive proposal depth, `0..mtp_depth`; request correctness limits can shorten it |
| `--decode-width N` | `0` | Target pending + proposal rows, `0..4294967295`; 0 keeps fixed maximum depth |
| `--kv-cache-gpu-mib N` | required for `serve-http`; positional for `run-jobs` | Positive GPU KV budget |
| `--kv-cache-cpu-mib N` | `0` | Host prefix-cache budget; zero disables cold storage |
| `--kv-cache-index-mib N` | `512` | Positive host cache-index budget |
| `--kv-checkpoint-interval-tokens N` | `8192` | Periodic checkpoint spacing; zero disables periodic captures |

Serving, jobs, and offline batches pack whole ready text chunks and image spans into each
forward without waiting for arrivals. Positions and attention histories stay
independent. With the defaults, `--prefill-chunk-tokens 1024 --prefill-batch-tokens 2048`,
two prompts can each contribute 1024 rows to one dense forward while their attention
stays in separate 1024-row chunks. Larger batch caps allocate more dense
activation workspace; attention workspace follows the per-prompt cap and image
minimum. Image spans stay complete, with private attention histories. Tower
passes run serially before the packed decoder forward. An image larger than the
batch cap runs alone without splitting.
The best total cap depends on prompt lengths: the measured caption workload
benefits from larger batches, while four full 1024-row chunks per GEMM were
slower than one on the tested G0 setup.
The 2048 default fits two full chunks and was neutral on that workload.

Add `--prefill-budget-tokens 8192` to allow roughly four full forwards before
a decode opportunity. This can build decode concurrency faster, trading longer TTFT for higher average throughput. The default
logical budget remains 0: one forward/head, then a decode opportunity. Both
larger physical batches and larger logical budgets can lengthen decode gaps.

With positive decode width `W`, each decode pass chooses
`clamp(max(0, floor(W / B) - 1), mtp_min_depth, mtp_depth)` using its actual
ready decode batch `B`. Minimum 0 permits ordinary decoding. The minimum can
exceed the width target; output limits, checkpoints, and ordinary-only requests
can shorten individual depths. Staging is still reserved for the maximum, so
`capacity * (mtp_depth + 1) <= 1280` and the GPU KV budget must hold.

Positive logical budgets group existing microbatches; they do not enlarge GPU
microbatches or reserve additional KV. Packed forwards and image spans remain
atomic and may overshoot the budget; first-token heads count as one token. Polling
and first-token emission continue between forwards. Blocked prefill falls back to
decode immediately. Larger budgets can improve throughput at the cost of
longer inter-token gaps; TTFT depends on the workload.

## HTTP options

All HTTP options follow `serve-http`.

| Option | Default | Meaning |
|---|---|---|
| `--model-dir DIR` | required | Complete serving bundle |
| `--max-batch N` | required | Maximum simultaneous execution batch |
| `--host HOST` | `127.0.0.1` | IPv4 bind address |
| `--port N` | `6311` | HTTP port |
| `--model NAME` | `gewell-gemma-4-31b-bf16` | Public model ID required in generation requests |
| `--image-max-soft-tokens N` | `280` | Default per-image budget: `70`, `140`, `280`, `560`, or `1120`; requests may override it |
| `--max-connections N` | `64` | Accepted connection limit |
| `--max-body-bytes N` | `8388608` | Per-request body limit |
| `--max-body-total-bytes N` | `268435456` | Combined encoded-body and prepared-image tensor limit |
| `--max-output-bytes N` | `8388608` | Buffered output limit per connection |
| `--socket-timeout-seconds N` | `60` | Connection timeout |

The transport limits above require positive integers. The listener has no
API-key flag; expose a non-loopback bind only behind appropriate network
controls.

Image input requires `--vision PATH`. The server image budget is a default,
not a ceiling: [image requests](http-api.md#images) can select any supported
budget with `mm_processor_kwargs.max_soft_tokens`. This is independent of
the text prefill chunk size and logical prefill budget.

## Single-request sampling

Place these options before `generate` or `caption`:

| Option | Default | Values |
|---|---|---|
| `--temperature T` | `0` | Finite nonnegative number |
| `--top-p P` | `1` | `0..1` |
| `--top-k K` | `0` | `0..262144`; zero keeps the whole vocabulary |
| `--seed S` | generated | Unsigned 64-bit integer |

Temperature zero, top-p zero, or top-k one selects greedy decoding with the
lowest-token-ID tie break. Repeatability is scoped to the same execution
configuration; batch shape and numerical precision can change results.

## Text codec

The command reads JSON objects, one per line, and emits one JSON result per
line. Supported operations are `encode`, `decode`, `completion`, `chat`, and
`chat_decode`:

```bash
build/gewell text-codec MODEL_DIR <<'JSONL'
{"operation":"encode","text":"Hello"}
{"operation":"chat","messages":[{"role":"user","content":"Hello"}]}
{"operation":"decode","tokens":[2,105]}
JSONL
```

`completion` accepts either `text` or `tokens`; `chat` accepts messages and
optional tools/template settings. `chat_decode` separates content and reasoning.
This diagnostic uses the built-in Gemma 4 text template; send raw images through HTTP.
Invalid operations emit an error object, processing continues, and the final
exit status is nonzero if any line failed.

## Preparation and diagnostics

The [converter](models.md) is `python3 tools/convert.py`. It accepts
safetensors or native weights and writes `weights.gwt`; use `--help` for its
options and `--verify WEIGHT_FILE` to check a complete payload.
`tools/extract_vision.py --snapshot SOURCE_DIR --output VISION.safetensors`
extracts the optional vision component.

`--qdq-mask PATH` is an advanced numerical diagnostic that applies weight
quantize/dequantize recipes. It does not convert or select a native artifact.
It precedes `generate`, `generate-batch`, or `replay-rollout`; for `serve-http`
and `run-jobs` it can follow the command arguments. Put it after other leading
execution options. The mask uses `LAYER PROJECTION RECIPE` rules;
normal deployment uses the precision stored in its bundle.

`generate-batch`'s optional `EVENTS.tsv` injects deterministic arrival,
cancellation, and failure events for scheduler testing. `replay-rollout`
consumes four TSV fields: request ID, prompt token path, continuation token
path, and saved generating-model logits. Neither is needed to launch serving.

The separately built `gewell_diagnostics` contains fixed capture/profile
commands: `bos`, `pair`, `cached-pair`, `short-decode`, `local-boundary`,
`local-boundary-prefill`, `graph-decode`, `profile-decode`, and `vision`.
Its fixtures are separate from the model-serving bundle and are intended for development/debug purposes.
