# HTTP API

The server implements a subset of OpenAI-compatible Chat Completions and
Completions. [Launch it](launch.md) with a local bundle and use its configured
model name in requests.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness; successful when the model is ready |
| `GET /v1/models` | List the configured model |
| `GET /v1/models/{id}` | Retrieve that model |
| `POST /v1/chat/completions` | Text/image chat, buffered or SSE |
| `POST /v1/completions` | Raw text or token-ID prompt, buffered or SSE |
| `POST /v1/cache/prefill` | Prepare a prefix without generating output |
| `POST /v1/cache/finish` | Release named prefix retention |
| `GET /v1/cache/stats` | Cache capacity and use |
| `GET /v1/cache/index` | Retained prefixes and active executions |
| `GET /metrics` | Prometheus metrics |
| `POST /v1/embeddings` | Text/image/video embeddings, only in `serve-embeddings` mode |

## Embeddings

```bash
curl http://127.0.0.1:6311/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"google/embeddinggemma-2","input":["task: search result | query: What causes auroras?","title: none | text: Charged particles from the sun cause auroras."],"dimensions":768}'
```

`model` is required. `input` accepts a nonempty string, a content object, or a
nonempty array of either. Text and task prefixes are passed verbatim; BOS/EOS are added
by the tokenizer. `dimensions` accepts 128, 256, 512 or 768 (default).
`encoding_format` accepts `float` (default) or `base64` of little-endian FP32.
Other fields and raw media placeholders in text are rejected. The response contains
`object:"list"`, `model`, and an ordered `data` array with `object:"embedding"`,
zero-based `index`, and `embedding` in each item. `usage.prompt_tokens` and
`usage.total_tokens` count actual tokens including BOS/EOS, media boundaries and
inserted features, without vision padding.

Image and video requests require an image-capable bundle started with `--vision`;
audio requires an audio-capable bundle started with `--audio`.
Each structured input is exactly `{"content":[...]}` with ordered parts:

| Part | Shape |
|---|---|
| Text | `{"type":"text","text":"..."}` |
| Image | `{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}` |
| Video | `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,..."}}` |
| Audio | `{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}` |

The abbreviated URLs illustrate the shape; supply actual base64 media bytes.
Images support PNG/JPEG. Videos support H.264 in MP4 and VP8/VP9 in WebM
(`data:video/webm;base64,...`). Audio supports WAV (PCM or IEEE float), MP3
(`audio/mpeg`) and FLAC (`audio/flac`); it is downmixed to 16 kHz mono without
loudness normalization and is not truncated (about 25 tokens per second). Local
paths, remote URLs, `detail`, and unknown fields are not accepted. Text parts concatenate without separators;
empty parts are allowed when the complete input contains nonempty text or media.
Multiple media parts and text form one embedding in their supplied order.

Optional top-level `mm_processor_kwargs` accepts only `max_soft_tokens`, one of
70, 140, 280, 560, 1120 (default 280), applied to every image in the request. The
processor's actual feature count can be smaller. These kwargs do not alter video.
Video samples at a fixed 1 FPS, uniformly caps the selections to 32 frames and
uses a maximum of 140 soft tokens per frame. Sampling uses frame count and the
stream's declared average frame rate, including for variable-rate clips. Missing
FPS uses all frames before the uniform cap; sub-1-FPS sources may repeat frames.
Timestamps are not inserted. Container rotation is not applied, and the soundtrack
is ignored. Clips with changing frame dimensions are rejected.

The assembled sequence, including all media, must fit the token limit. Each media
data URL is bounded to 8 MiB, decoded sides to 8192 pixels, and decoded pixels per
image/frame to 16,777,216. Prepared
patch/position buffers share the aggregate request-memory limit. Invalid image,
video or audio data returns 400 `invalid_image`, `invalid_video` or
`invalid_audio`; expanded overflow returns 400
`context_length_exceeded`, and prepared-memory exhaustion returns 503
`capacity_exceeded`. Errors identify indexed input/part paths.

Each input is limited to 8192 tokens by default; there is no truncation.
Malformed input, unsupported dimensions and request/output limits return 400;
unknown model IDs return 404; a full execution queue returns 503. The entire
request is validated before execution. Batch results are returned together;
disconnecting cancels video preparation, queued work or active execution at a
layer boundary.
See [CLI limits](cli.md) for request, scratch, queue and transport settings.

## Generation

Chat accepts `messages` with system, developer, user, assistant, and tool roles.
Assistant history can carry `reasoning_content` and `tool_calls`; tool messages
must follow their assistant call and carry the matching `tool_call_id`. Raw
Completions accepts a string or one token-ID array in `prompt`. Each request
produces one choice. Supported sampling controls are `temperature`, `top_p`,
`top_k`, and unsigned 64-bit `seed`. Use `max_tokens`, or
`max_completion_tokens` for chat, to bound output. EOS stops generation.

Set `stream:true` for SSE. `stream_options.include_usage:true` requests a final
usage event; otherwise streaming omits usage. Buffered responses always include
usage, including `prompt_tokens_details.cached_tokens`. The total context
horizon is at most 262,144 processed tokens and can be lower for a particular
memory configuration.

`stop` accepts a string or an array of strings; matched stop text is removed
from the output. `chat_template_kwargs` accepts the booleans `enable_thinking`
and `preserve_thinking`; unknown keys such as `clear_thinking` are ignored.
Rendering follows the built-in Gemma 4 template described in the
[bundle guide](models.md). Reasoning is returned separately as
`reasoning_content`. For compatibility, chat also accepts
`reasoning_effort`: `none` disables thinking; `low`, `medium`, and `high` all
enable it. A supplied value overrides `chat_template_kwargs.enable_thinking`.
Omission or null keeps the template setting, which defaults to false. This
changes the thinking switch, not its budget. Chat prefill uses the same mapping.
Chat also accepts `logprobs:true`
and `top_logprobs` from 0 through 20; supplying `top_logprobs` requires logprobs
to be enabled. Raw Completions' integer `logprobs` control is unsupported.

## Tools and JSON answers

Chat supports function definitions in `tools`, generated `tool_calls`, and
matching tool-result history. `tool_choice` supports `auto`, `none`, `required`,
named functions, and `allowed_tools` subsets. Named selection forces one call;
`parallel_tool_calls:false` limits the response to at most one call.
`strict:true` enforces the supported JSON Schema subset for arguments. Each
strict object requires `additionalProperties:false` and every property in
`required`; use nullable types for optional values. Unsupported strict assertions
are rejected before generation. Non-strict schemas retain metadata, references,
unions, and assertions as prompt guidance; `$schema` string metadata is accepted
and removed during normalization. JSON argument keys can contain Unicode and
punctuation.
The server never executes tools; the client runs them and submits the
results.

`response_format` supports `text`, `json_object`, and `json_schema`. Example:

```json
{
  "model": "gemma-4-31b",
  "messages": [{"role": "user", "content": "Return a short greeting."}],
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "greeting",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
        "additionalProperties": false
      }
    }
  }
}
```

Constraints apply to the answer channel and work with MTP. Stop strings cannot
be combined with a constrained response format. When tool definitions are also
supplied, `response_format` constrains the answer branch and each tool's schema
and strictness govern its arguments. Invalid or unsupported strict schemas receive a
request error. A response stopped by the token limit can still be incomplete.

## Images

User message content can contain text parts and `image_url` parts. Images must
be inline base64 PNG or JPEG data URLs; remote URLs and local file URLs are not
fetched. Enable image input with `--vision PATH`. Multiple images and images
in multiple user turns are supported.

The image budget is a maximum number of soft tokens **per image**. Supported
values are **70, 140, 280, 560, and 1120**. The server defaults to **280**;
set `--image-max-soft-tokens N` after `serve-http` to change that default.
A request can override it in either direction with the top-level
`mm_processor_kwargs.max_soft_tokens` field:

```json
{
  "model": "gemma-4-31b",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Read the small text in this picture."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_IMAGE_BYTES"}}
  ]}],
  "mm_processor_kwargs": {"max_soft_tokens": 1120},
  "max_tokens": 128
}
```

The override applies to every image in the request, including earlier user
turns. The same option works on `/v1/cache/prefill` with `messages`. Omitted,
null, or empty `mm_processor_kwargs` uses the server default. A supplied
`max_soft_tokens` must be an integer from the list above; null, other values,
and unknown processor options return 400. Raw Completions and cache prefill
with `prompt` reject non-null processor options.

For clients using the OpenAI Python SDK, put this extension in
`extra_body={"mm_processor_kwargs": {"max_soft_tokens": 1120}}` on the chat
completion call. `image_url.detail` can be omitted, null, or `"auto"`; each
uses the selected token budget. `"low"` and `"high"` are unsupported.

Larger budgets retain more spatial detail and increase processing time, prompt
length, and memory use. Aspect-ratio-preserving resizing can produce fewer
tokens than the selected maximum. For example, a square image uses 256 image
tokens at budget 280 and 1089 at budget 1120. The 1280-row execution capacity
is not an image-budget option.

The server decodes and prepares images natively. Body, pixel, tensor, and GPU
capacity limits apply. Prepared tensors cost about 7.4 MiB per image at 280
and 29.6 MiB at 1120, charged alongside encoded bodies against
`--max-body-total-bytes` (default 256 MiB). Exhausting this limit returns 503;
raise it when serving many concurrent large images. The server retains one GPU
vision workspace sized for the largest image processed: up to about 112 MiB
at 280 or 449 MiB at 1120, including uploads and the tower's output. Packed
prefill uses a separate image-feature buffer: up to 21 MiB with the default
`--prefill-batch-tokens 2048`, or 42 MiB at the maximum 4096. These buffers
grow as needed and remain until shutdown; they need room alongside model
weights, text executor scratch and the configured KV cache.

Prepared image identity participates in prefix caching. Repeating the same
image and budget can reuse its cached prefix; changing the budget changes
that identity. Existing retained prefixes remain available for requests using
their original budget.

## Cache

Prefix reuse is automatic. The optional `cache` object supports `mode` (`auto`
or `reuse_only`) and named retention through `prompt_id`, `priority`, and
`finished`. `reuse_only` reuses existing prefixes without retaining new ones
and cannot carry owner controls. Named retention consumes the configured
cache budget; release it through `/v1/cache/finish` when finished.

`/v1/cache/prefill` accepts `prompt_id` inside `cache`, for example
`{"prompt":"Hello","cache":{"prompt_id":"document"}}`.
`/v1/cache/finish` instead takes `{"prompt_id":"document"}` at the top level.
The [cache guide](cache.md) explains matching, ownership, `reuse_only`, shared
prefill, checkpoint storage, GPU/CPU eviction, and usage accounting with diagrams
and complete examples.

## Transport and validation

The listener uses HTTP/1.1 with `Content-Length` request bodies and
`Connection: close` responses. Chunked uploads and persistent HTTP connections
are outside this subset. Disconnecting, including a request-side TCP
half-close, cancels the associated work. Slow output consumers are bounded by
the configured buffers and timeouts.

Recognized controls are accepted only at their neutral value: `n:1`, zero
`frequency_penalty`/`presence_penalty`, empty `logit_bias`, and, in chat,
`store:false` and `modalities:["text"]`. Always
rejected are `functions`, `function_call`, and the
prompt-cleanup fields `audio`, `moderation`, `prediction`,
`prompt_cache_retention`, `service_tier`, `verbosity`, and
`web_search_options`. Each endpoint also rejects the other endpoint's fields:
chat rejects `prompt`, `echo`, `best_of`, and `suffix`; Completions rejects
`messages`, `max_completion_tokens`, `chat_template_kwargs`, `top_logprobs`,
`logprobs`, `response_format`, `tools`, `tool_choice`, `store`, `modalities`,
`parallel_tool_calls`, `reasoning_effort`, and `mm_processor_kwargs`, and accepts `echo` and `best_of` only at their
defaults. Unknown top-level fields are ignored. Nested messages, tools,
schemas, processor options, and cache objects are validated. There is no `/v1/responses`
endpoint. Error responses include an `x-request-id` for matching server logs;
an error after streaming has started is reported within the stream.
