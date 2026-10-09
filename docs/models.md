# Model bundles and conversion

EmbeddingGemma 2 uses a separate BF16 text bundle on SM120. A prepared bundle
with all three components (text, vision, audio) is published at
[LeDissolution/EmbeddingGemma-2-Gewell](https://huggingface.co/LeDissolution/EmbeddingGemma-2-Gewell):

```bash
hf download LeDissolution/EmbeddingGemma-2-Gewell --local-dir EMBEDDING_DIR
build/gewell serve-embeddings EMBEDDING_DIR --vision --audio
```

Omit `--vision` or `--audio` to leave that component unloaded. To extract a
bundle yourself, for example from a fine-tune:

```bash
python3 tools/extract_embeddinggemma2.py --snapshot SOURCE_DIR --output EMBEDDING_DIR
build/gewell serve-embeddings EMBEDDING_DIR
```

The source must have the supported EmbeddingGemma 2 architecture and BF16 text
weights. Extraction validates all 413 tensors, preserves their values and
creates `text.safetensors`, `config.json` and `tokenizer.json` in a new directory.
It accepts compatible fine-tunes without source/revision allowlists. Native
startup validates configuration, tokenizer and tensor inventory. Text inputs
support up to 8192 tokens including BOS/EOS and 128/256/512/768 dimensions.
No task prefixes are inserted automatically. Explicit GPU allocations are
274,210,816 weight bytes and 167,773,696 scratch bytes at the 8192-token
capacity, excluding CUDA/cuBLAS overhead; the token-embedding table stays in
host memory. Embedding serving runs independently of generation serving.

For image or video inputs, extract an image-capable bundle:

```bash
python3 tools/extract_embeddinggemma2.py --snapshot SOURCE_DIR --vision --output IMAGE_EMBEDDING_DIR
build/gewell serve-embeddings IMAGE_EMBEDDING_DIR --vision
```

This adds 211 BF16 tower/bridge tensors in `vision.safetensors` and copies
`processor_config.json`. The combined explicit allocations are 610,017,280
weight bytes and 178,782,528 scratch bytes at the 8192-token capacity, excluding
CUDA/cuBLAS overhead. Vision weights stay resident; vision and text execution
share scratch. Without the boolean `--vision`, only text weights are loaded.

For audio inputs, add `--audio` to extraction and serving:

```bash
python3 tools/extract_embeddinggemma2.py --snapshot SOURCE_DIR --vision --audio --output MULTIMODAL_DIR
build/gewell serve-embeddings MULTIMODAL_DIR --vision --audio
```

This adds 752 BF16 audio tower/bridge tensors in `audio.safetensors`. Omit
`--vision` from both commands for a text/audio bundle. With all three
modalities the explicit allocations are 1,223,507,968 weight bytes and
1,028,441,648 scratch bytes, excluding CUDA/cuBLAS/cuDNN overhead. Requests
that mix images or video with audio need both options.

The remaining bundle and conversion instructions describe Gemma 4 generation.

A serving bundle contains text weights and tokenizer data:

```text
model-directory/
  manifest.json
  weights.gwt
  tokenizer.json
```

All main weight precisions use `.gwt`. The header and per-tensor metadata
identify storage; the filename does not select precision. Startup validates
weight structure and bundle integrity. `gewell verify WEIGHT_FILE` additionally
scans the full weight payload and padding. Checksums detect corrupted files;
repositories, commits, and source hashes are not model allowlists.

`tokenizer.json` supplies the vocabulary and BPE merges. Its token IDs and
pipeline must match the supported Gemma 4 contract. Tokenization behavior,
chat rendering, model architecture, and generation defaults are implemented
in the runtime. Source configuration and chat-template files are not bundled
or interpreted as runtime settings. Python reference and calibration tools
read those files from their original source checkpoints.

## Download a prepared bundle

Download a published bundle into a local directory:

```bash
MODEL_REPO=organization/model-bundle
MODEL_DIR=/absolute/path/to/model-bundle
hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
```

Use `hf auth login` if the repo requires authentication. An optional
`--revision` selects a particular release for reproducibility. Once the
bundle is local, serving needs no Hugging Face connection.

## Convert safetensors

Conversion uses Python 3.10+ and NumPy. `tools/convert.py` accepts a Gemma 4 31B or 26B A4B safetensors directory or
single file through `--snapshot`. Fine-tunes and repacked checkpoints are
accepted when required tensor names, shapes, and storage layouts match.
Shard names and counts are unrestricted. A safetensors index is optional;
when present, it must agree with the shard contents. Unrelated tensors are
allowed, and text conversion does not require vision or assistant weights.

Supported source encodings are BF16, FP16, FP32, E4M3 FP8 (with an optional
scalar FP32 `weight_scale`), and packed NVFP4: U8 pairs of E2M1 weights,
group-16 E4M3 `weight_scale`, and scalar FP32 `weight_scale_2`.
Other quantization layouts require decoding to one of these encodings first.
Activation `input_scale` values come from explicit overrides, the source, or
the bundled reference calibration, as described below.

```bash
SOURCE_DIR=/absolute/path/to/gemma4-31b-safetensors
MODEL_DIR=/absolute/path/to/gewell-model

python3 tools/convert.py --snapshot "$SOURCE_DIR" --plan
python3 tools/convert.py --snapshot "$SOURCE_DIR" --output "$MODEL_DIR"
build/gewell verify "$MODEL_DIR/weights.gwt"
```

The tokenizer defaults to `tokenizer.json` in the source directory. Use
`--serving-snapshot DIR` when it is stored separately. Conversion keeps source files in place and
refuses to replace existing output weights or a manifest. `--plan` validates
metadata and scales without copying weights. BF16 text weights occupy about
57.2 GiB for 31B and 47.0 GiB for 26B A4B; packed projections reduce that size. Conversion streams tensors,
with temporary decoding storage bounded by one tensor.

## Select precision

For Gemma 4 26B A4B, pass `--architecture gemma4_26b_a4b` to conversion,
planning, and native repacking. Source import supports Google's stacked BF16
experts and separately named expert projections, including NVIDIA's packed
NVFP4 checkpoint. Omitted mask entries retain their source precision.

```bash
python3 tools/convert.py --architecture gemma4_26b_a4b \
  --snapshot /path/to/gemma4-26b-a4b --plan
python3 tools/convert.py --architecture gemma4_26b_a4b \
  --snapshot /path/to/gemma4-26b-a4b --output /path/to/26b-bundle
```

26B layers are `0..29`, and experts are `0..127`. Plain `gate_proj`,
`up_proj`, and `down_proj` select the shared MLP; use
`experts.*.gate_proj` or `experts.7.down_proj` to select routed projections.
Rules support `*` layers and later rules win. Attention, shared, and routed
projections can each use BF16, FP8, or NVFP4. The router, norms, embeddings,
and tied head remain BF16. Native NVFP4 pads physical output rows while
preserving logical dimensions, including the 704-wide expert MLP.

Both `--nvfp4-activation-policy always` and `prefill` apply to 26B execution.
The latter uses BF16 activations with packed NVFP4 weights during decode.
Conversion coverage and quality limits are described under
[activation calibration](#activation-calibration).

The remaining examples in this section use the 31B inventory.

A mask contains `LAYER PROJECTION STORAGE` rules. `LAYER` is `*` or `0..59`;
storage is `bf16`, `fp8_w8a8`, or `nvfp4_w4a4`. Projection names are `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Global
layers share K/V, so there is no separate global V projection.

For example:

```text
* gate_proj nvfp4_w4a4
* up_proj fp8_w8a8
* down_proj bf16
```

Later rules override earlier ones. Omitted projections retain their source
storage; FP16 and FP32 inputs become BF16. Embeddings and norms use BF16.
Newly quantized or requantized projections use the bundled reference
activation scales by default. No GPU or separate calibration file is required.

```bash
python3 tools/convert.py --snapshot "$SOURCE_DIR" \
  --mask /path/to/recipe.mask \
  --output /path/to/converted-model
```

Native `.gwt` input is also supported:

```bash
python3 tools/convert.py --artifact "$MODEL_DIR/weights.gwt" \
  --mask /path/to/recipe.mask \
  --output /path/to/repacked-model
python3 tools/convert.py --verify /path/to/repacked-model/weights.gwt
```

A lower-precision source can be decoded to a higher-precision target. The
converter warns and proceeds: storing NVFP4 values in BF16 or FP8 cannot
recover information already lost during quantization. Unchanged native
entries preserve their packed bytes and scales unless an explicit activation
scale override is supplied; that changes only the input scale.

### G0 reference mask

`tools/masks/g0.mask` supplies the G0 precision layout: **266 FP8 W8A8 and
144 NVFP4 W4A4** text projections. Q/K/V use FP8 throughout. O and MLP
projections use FP8 in layers **0–5, 10–11, 16–17, 22–23, 28–29, 34–35,
40–41, 46–47, 52–53, and 58–59**, and NVFP4 elsewhere. These are zero-based
indices: the first six-layer block, then every global layer and its preceding
local layer. Embeddings and norms remain BF16.

The mask specifies every text projection, so its precision layout is
independent of source storage. Activation scales follow the
[selection rules below](#activation-calibration), including the bundled
measurements from the all-NVFP4 reference run.

```bash
python3 tools/convert.py --snapshot "$SOURCE_DIR" \
  --mask tools/masks/g0.mask --plan
python3 tools/convert.py --snapshot "$SOURCE_DIR" \
  --mask tools/masks/g0.mask --output /path/to/g0-model
```

## Activation calibration

For each quantized projection the converter selects its activation scale in
this order:

1. An explicit `--input-scales FILE` entry.
2. The source's `input_scale`, if retaining that projection's quantization format.
3. The bundled reference calibration for missing scales and changed formats.

The 31B profile, `tools/default_calibration.json`, contains measured activation
absolute maxima for all 410 text projections. The reference run observed 74
histories (64 text, eight image, two video), up to 32,641 input tokens, using
BF16 forward activations and quantize/dequantize (QDQ) of packed all-NVFP4
weights. The converter derives a positive FP32 dequantization multiplier as
`input_amax / 448` for FP8 and `input_amax / 2688` for NVFP4. These are saved
reference measurements, not new measurements of the selected source or mask.
Fine-tunes and different workloads may benefit from custom scales; the
reference provenance never restricts which structurally compatible sources
can be converted.

26B conversion (`--architecture gemma4_26b_a4b`) uses
`tools/gemma4_26b_default_calibration.json`. It supplies 11,683 measured ranges
from 512 BF16 text prompts, capped at 512 tokens each, with observation counts
and pinned source/corpus provenance. Its 42 uncovered expert projections have
no default scale: a plan requesting them reports their names and requires
source scales, explicit overrides, or an explicit BF16 mask entry. These
measurements do not establish image, long-context, or quantized-model quality.

When defaults are used, the converter prints one notice. `--plan` reports
how many scales come from overrides, the source, and defaults. The output
manifest records the resolved scales, their origins, and the default
profile's provenance and checksum. BF16 projections need no activation scale.

Optional overrides are a JSON object keyed by logical tensor name:

```json
{"layers.0.mlp.gate_proj.weight": 0.125}
```

The number illustrates syntax. Supply the dequantization multiplier for the
**target** format, not an absolute maximum or reciprocal. A partial file
overrides only its named projections; the usual source/default selection
applies to the rest. Unknown names and invalid scales for quantized targets
are rejected. Entries for projections left in BF16 are unused.

```bash
python3 tools/convert.py --snapshot "$SOURCE_DIR" \
  --mask /path/to/recipe.mask --input-scales /path/to/scales.json \
  --output /path/to/custom-calibrated-model
```

## Optional assistant and vision

Load a structurally compatible BF16 assistant directly:

```bash
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 1 --kv-cache-gpu-mib 8192 \
  --assistant /path/to/assistant/model.safetensors --mtp-depth 3
```

Extract the vision tower and projector from any source containing the required
tensors, using the same safetensors reader and supported encodings as text
conversion:

```bash
python3 tools/extract_vision.py --snapshot "$SOURCE_DIR" \
  --output /path/to/vision.safetensors
build/gewell serve-http --model-dir "$MODEL_DIR" --max-batch 1 --kv-cache-gpu-mib 8192 \
  --vision /path/to/vision.safetensors
```

Extraction preserves BF16 values and decodes other supported encodings to
BF16, warning when widening quantized weights. Vision extraction does not
require text weights. Runtime component files currently use BF16.

The two component paths can be combined. Omitting `--vision` rejects image
requests. Omitting `--assistant` with a positive depth warns and forces depth
zero. Depth zero never loads assistant weights.

Run each tool with `--help` for its complete argument list. Calibration reports
and recommended recipes belong with the published bundle.
