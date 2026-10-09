<img width="1672" height="350" alt="gewell" src="https://github.com/user-attachments/assets/53d54098-4310-4714-a343-a9cf6d84600f" />

Gewell is a single-GPU inference engine for Gemma 4 31B, 26B A4B and EmbeddingGemma 2 on NVIDIA Blackwell
GPUs with compute capability 12.0 (`sm_120a`) (*potentially* more to come).

Features include streamed responses, PNG/JPEG image input with configurable
per-image token budgets, tool conversations, JSON-constrained answers,
continuous batching, prefix caching with optional CPU storage, and
MTP speculative decoding using the Gemma assistant. Local and global
KV storage can use BF16 or FP8 (no, I will not be implementing FP4. Go lobotomize your models somewhere else).

EmbeddingGemma 2 provides BF16 text, image, video, and audio embeddings through a
dedicated `serve-embeddings` server.

## Get started

1. [Build the engine](docs/build.md).
2. [Obtain or convert a model bundle](docs/models.md).
3. [Launch the server and send a request](docs/launch.md).


## Reference

- [Command-line interface](docs/cli.md)
- [HTTP API](docs/http-api.md)
- [Prefix caching, prefill, and retention](docs/cache.md)
- [Offline jobs and token files](docs/offline.md)
- [HTTP benchmarking](docs/benchmark.md)
