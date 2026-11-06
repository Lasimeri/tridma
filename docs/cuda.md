# The GPU beside the card

What this adds to design A ([design.md](design.md)): an NVIDIA GPU (an
RTX 3090 Ti on PCIe Gen4 x8 here) for prompt prefill and the KV cache,
while the weights stay in host RAM and on the Xeon Phi. llama.cpp is still
unmodified; everything is a script, a launch setting or a program beside it.

| File | What it is |
|---|---|
| `scripts/serve.sh` | one llama-server: weights in host RAM + on the card, the GPU for prefill and the context; `stop` frees the card |
| `cuda/vpu_dma.cu` | CUDA kernels that drive the card worker by DMA through its host window ([vpu_dma.md](../cuda/vpu_dma.md)) |
| `scripts/pd.sh`, `tools/pd.c` | two llama-servers, prefill on the GPU and generation on the card, overlapped across prompts |
| `tools/evict.c` | drops a model file from the page cache without root ([evict.md](../tools/evict.md)) |

## serve.sh: the layout

```
serve.sh WORK MODEL.gguf start|stop|status
SERVE_CTX (262144)  SERVE_PORT (8095)  SERVE_ALIAS  SERVE_ARGS (extra llama-server arguments)
CARD_BYTES (10500000000)  HOST_THREADS (14)  TG_ONLY (31)
```

| Setting | Effect |
|---|---|
| `-ngl 99 -ot ".*=CPU"` | every layer is the GPU's, so its KV cache and attention live in VRAM; every weight tensor stays in host RAM |
| `libggml_phi.so`, `PHI_GGML_OFFLOAD=1` | the card keeps a share of every weight matrix's rows (up to `CARD_BYTES`) and multiplies them; the host's threads do the rest |
| `PHI_GGML_TG_ONLY=31` | a multiply of more than 31 tokens is declined by the card backend, so it reaches CUDA (below) |
| `-b 2048 -ub 2048` | each prefill batch carries the weights across PCIe once for 2048 tokens |

**Why `TG_ONLY` is needed.** ggml's scheduler offers a multiply to a GPU
(op offload, batches of 32 tokens and up) only when the weight's backend is
the last one, the CPU (`ggml-backend.cpp`, the `op_offload` branch of
`ggml_backend_sched_backend_id_from_cur`). The card backend accepts host
buffers, so without the setting it takes every host weight and no prefill
ever reaches the GPU: measured with the 27B below, GPU SM 7-29 % and PCIe
receive 2-230 MB/s during a prompt. With `TG_ONLY=31` the card backend
declines the batch (`phi_ggml_take` in `host/asm/ggml-phi/backend.S`, which
checks it before the offload setting), the multiply falls to the CPU backend,
and the scheduler hands it to CUDA. Multiplies of up to 31 tokens (single
tokens, a speculative draft's verification batch) stay with the card.

`stop` sends SIGTERM, waits 10 s (llama-server finishes its current batch
first, which takes long with weights in host RAM), then SIGKILL; then it
frees every weight on the card (`vpu_dma -F`) and returns the worker's free
huge pages to the card (`nr_hugepages` down to the pages in use). `start`
grows the pool back to `PHI_VPU_HUGEPAGES` (5600) first.

## Measured, Qwen3.8-27B Q8_0 + its MTP head Q4_0

`SERVE_CTX=204800 SERVE_ARGS="-md mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp -ngld 99"`,
buffer sizes from llama.cpp's own load lines (`-lv 4`):

| Buffer | Where | MiB |
|---|---|---|
| model weights (66 layers) | host RAM | 27690.97 |
| KV cache, 204800 tokens | GPU | 12800.00 (8192.00 at 131072: 64 KiB a token) |
| recurrent state | GPU | 598.50 |
| compute | GPU | 1618.08 |
| MTP head weights | GPU / host | 774.71 / 521.00 |
| MTP KV / compute | GPU | 800.00 / 1072.06 |
| llama-server on the GPU (nvidia-smi) | GPU | 17966 |

The card keeps 37.5 % of every matrix's rows (10.47 GB), the host the rest.

One request, a 699-token prompt, 8 tokens out (`nvidia-smi dmon -s ut -d 1`
beside it): prefill 28.49 s = 24.54 tok/s with PCIe receive 822 MB/s mean
(3654 peak, about one pass of the weights for the one batch), generation
8 tokens in 5.35 s, the MTP draft accepted 5 of 6. The limit is host RAM:
`fincore` showed 12.6 of the file's 27.1 GiB cached, with 17.7 GB of swap
in use, so each batch re-reads most of the weights from NVMe. Not yet
measured: a larger `-ub` (fewer weight passes per token), repeated runs.

Qwen3.8-35B-A3B Q8_0 in the same layout (`SERVE_CTX=262144`, before the
`TG_ONLY` fix, so prefill on the host and card): one 1362-token request,
prefill 56.25 tok/s, generation 3.92 tok/s. With most of the model on the
GPU instead (`-ngl 99 -ncmoe 21`, the card keeping half of the host-side
experts, 9.12 GB), an agent session's 12710-token prompt prefilled at
195.37 tok/s and generated at 34.19 to 42.25 tok/s.

## vpu_dma: the GPU drives the card

[vpu_dma.md](../cuda/vpu_dma.md) has the protocol. Measured against the
running worker (4096 x 2048 float32, one column, 2000 round trips each):

| | p50 | mean | card's share |
|---|---|---|---|
| GPU-driven (one kernel: put, doorbell, wait, get) | 302.1 us | 323.1 us | 301.6 us |
| host-orchestrated (copy engine, CPU doorbell) | 301.3 us | 322.2 us | |

GPU writes into the window: 13.14 GB/s by the copy engine, 13.01 GB/s by
kernel stores. Results match an fp64 reference to 9.661e-08 of the largest
value; the worker's DMA counters show exactly the activation and result
bytes of every request. The GPU path costs the same as the host's; what it
removes is the CPU from the loop. It is not wired into llama.cpp's graph.

## pd: prefill and generation overlapped

`pd.sh` starts a prefill server (the GPU, its own `--fit` placement) and a
decode server (design A, the GPU hidden), both with one `--slot-save-path`;
`pd` hands each prompt over as a slot file: the prefill server gets the
prompt minus its last token (a hybrid model cannot drop tokens from its
recurrent state, so the decode server must extend the cached prompt), the
decode server restores it and evaluates one token. Measured on the 35B: the
decode server found 1361 of 1362 tokens cached and evaluated 1; the first
token's top-5 log-probabilities differ from design A alone by at most 0.17
(the GPU and the CPU round differently; design A against the GPU alone
differ by up to 0.127). Four prompts, 128 tokens each: overlapped 227.81 s,
one after the other 457.95 s, but the host slowed between the two runs (its
page cache), so the gain is not established.
