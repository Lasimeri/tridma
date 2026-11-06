# The serving configuration

Qwen3.8-27B Q8_0 with its MTP head (Q4_0) served by one llama-server on
127.0.0.1:8095 (OpenAI API): every weight of the model in host RAM, the Xeon Phi
7120A keeping 37.5 % of every matrix's rows and multiplying them for single
tokens, the GPU (an RTX 3090 Ti) holding the context (KV cache), the MTP head
and running prefill. Started with [`scripts/serve.sh`](../scripts/serve.sh) in a
llama-phi work directory (`WORK`, see the [README](../README.md)).

## Command

```
M=~/models/Qwen3.8-27B
SERVE_ALIAS=qwen3.8-27b-q8_0 SERVE_CTX=204800 \
SERVE_ARGS="-md $M/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp -ngld 99 -lv 4" \
  bash scripts/serve.sh WORK $M/Qwen3.8-27B-Q8_0.gguf start
```

which runs

```
WORK/llama.cpp/build-native-cuda/bin/llama-server
  -m ~/models/Qwen3.8-27B/Qwen3.8-27B-Q8_0.gguf --alias qwen3.8-27b-q8_0
  -ngl 99 -ot .*=CPU -t 14 -c 204800 -np 1 -b 2048 -ub 2048 --fit off --jinja
  --host 127.0.0.1 --port 8095
  -md ~/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp -ngld 99 -lv 4
env GGML_BACKEND_PATH=WORK/Intel-Phi-AVX-512-C/host/asm/out/libggml_phi.so
    PHI_GGML_CARDS=0 PHI_GGML_THREADS=120 PHI_GGML_CARD_BYTES=10500000000
    PHI_GGML_OFFLOAD=1 PHI_GGML_HOST_THREADS=14 PHI_GGML_TG_ONLY=31
```

The models: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
`Qwen3.8-27B-Q8_0.gguf` (29,047,086,048 bytes) and
`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 bytes). `--spec-type draft-mtp` is
required: with `-md` alone the draft loads and nothing speculates. `-lv 4` (debug
log) prints the buffer sizes below; drop it for a quieter log.

| Setting | Why |
|---|---|
| `-ngl 99 -ot .*=CPU` | every layer is the GPU's, so the KV cache and attention live in VRAM, while every weight tensor stays in host RAM |
| `PHI_GGML_OFFLOAD=1`, `CARD_BYTES` 10.5 GB | the card keeps its rows alone (the host drops their pages after the upload) |
| `PHI_GGML_TG_ONLY=31` | multiplies of 32 tokens and more reach the GPU (ggml offers an op to CUDA only when its weight sits on the CPU backend; see [cuda.md](cuda.md)); up to 31 (generation, MTP verification) stay with the card |
| `-c 204800` | 12,800 MiB of KV cache; 64 KiB a token |
| `-b 2048 -ub 2048` | one pass of the weights over PCIe per 2048 prompt tokens |

The card worker runs as llama-phi's `scripts/worker.sh` starts it:
`phi-vpu-worker -v -e 0 -O -S 1 -C 59 -T 2 -R 128 -s 1000 121` with 5600 huge pages.

## Where it sits (llama.cpp's load lines, `-lv 4`)

| Buffer | Where | MiB |
|---|---|---|
| model weights, 66 layers | host RAM | 27,690.97 |
| of them on the card | Xeon Phi | 37.5 % of every row, 10.47 GB |
| KV cache, 204,800 tokens | GPU | 12,800.00 |
| recurrent state | GPU | 598.50 |
| compute | GPU | 1,618.08 |
| MTP head | GPU / host | 774.71 / 521.00 |
| MTP KV cache / compute | GPU | 800.00 / 1,072.06 |
| llama-server on the GPU (nvidia-smi) | GPU | 17,966 |

## Measured

One request, a 699-token prompt, 8 tokens out, low reasoning effort: prefill
28.49 s = 24.54 tok/s (PCIe receive 822 MB/s mean, 3654 peak: the weights
streaming to the GPU), generation 8 tokens in 5.35 s, the MTP draft accepted 5 of
6. The limit is host RAM: on a 32 GB host shared with a desktop, 12.6 of the
file's 27.1 GiB stayed cached, so each prefill batch re-reads most of the weights
from NVMe. Not measured yet: a larger `-ub`, repeated runs.

## Stopping and the card's memory

`serve.sh stop` sends SIGTERM, waits 10 s (llama-server finishes its current batch
first), then SIGKILL; frees every weight on the card (`vpu_dma -F`) and returns
the worker's free huge pages to the card (`nr_hugepages` down to the pages in
use, about 172). `start` grows the pool back to 5600 first. Without that, the
card shows the pool (10.9 GiB) as used even with no weights in it.

## omp

[`omp/models.yml`](../omp/models.yml) is the provider file of an omp profile
(`~/.omp/profiles/llama-phi/agent/models.yml`): provider `llama-phi` at
`http://127.0.0.1:8095/v1`, model `qwen3.8-27b-q8_0` with a 172,032-token
prompt window and 32,768 output tokens (204,800 together), the template's
reasoning efforts `low`, `medium`, `xhigh` (`high` maps to `xhigh`; `low` asks
for brief thinking). The profile's `config.yml` is one's own omp config with
every model role set to `llama-phi/qwen3.8-27b-q8_0:low`,
`defaultThinkingLevel: low`, compaction thresholds at 155000 tokens and
`providers.maxInFlightRequests.llama-phi: 1` (one server slot).
[`scripts/omp-phi`](../scripts/omp-phi) starts omp on that profile only when
port 8095 is held by a llama-server of the same user.
