# The tridma router: host RAM, a Xeon Phi and a GPU serving one model

A software DMA router for one machine with three processing units: a Ryzen 7
5800X (32 GB), an Intel Xeon Phi 7120A (16 GB GDDR5, PCIe Gen2 x8) and an RTX
3090 Ti (24 GB, PCIe Gen4 x8). It serves Qwen3.8-27B Q8_0 (29.05 GB) with its
MTP draft head through an unmodified llama.cpp with these rules:

- the model lives in memory, never on the disk during inference: the host keeps
  only the rows the card does not hold;
- the card holds 13.1 GB of the weights (a 13.5 GB pool);
- the GPU does all prompt prefill, holds the KV cache (204,800 tokens) and runs
  the MTP head;
- every transfer has one owner per link, and no thread waits on I/O.

Nothing here can be a llama.cpp setting: its loader reads the whole file into
memory, and its GPU prefill copies whole weight matrices from the host's mapping
of the file, so the card's rows would come back from the disk. The router fixes
both from outside, with three pieces and two changes to the card's software.

## Where every byte lives

| Bytes | Where | Kept there by |
|---|---|---|
| host rows of every matrix, norms, `token_embd` (15.84 GB) | host RAM, page cache, locked | [`tools/pin.c`](../tools/pin.md) |
| card rows of every split matrix (13.1 GB) | the card's huge pages (worker pool 6437) | `libggml_phi.so` upload, then the router |
| card rows while a prompt is prefilled | host RAM, at most 512 MiB at a time | [`tools/tridma_router.c`](../tools/tridma_router.md) |
| MTP draft model's host tensors (1.10 GB) | host RAM, locked | `pin` with share 0 |
| KV cache, recurrent state, MTP head, compute buffers | the GPU | llama.cpp (`-ngl 99 -ot .*=CPU -ngld 99`) |

## How each link is used

| Link | Direction | What moves | Mechanism |
|---|---|---|---|
| host RAM -> GPU | prefill | host rows of each matrix, activations | CUDA op offload (llama.cpp), pageable copies from locked pages |
| card -> host RAM -> GPU | prefill | card rows of each matrix | page fault -> `K_DOWNLOAD` (card DMA into the window) -> `UFFDIO_COPY` -> the CUDA copy continues |
| GPU <-> host RAM | generation | one token's activations and results | llama.cpp's scheduler copies |
| host RAM <-> card | generation | activations in, results out | `libggml_phi.so` requests, card DMA channel 7 |
| card's own memory | generation | 46.9 % of every matrix multiply | the worker's 120-thread pool |

Batches of 32 tokens and more (prefill) go to the GPU: `PHI_GGML_TG_ONLY=31` makes
the card backend decline them, and ggml then offers them to CUDA (it does so only
for weights on the CPU backend). Batches of 31 and fewer (generation, MTP
verification) stay with the card and the host's 14 threads.

## The pieces

1. **`K_DOWNLOAD` (kernel 32) in the card worker** (Intel-Phi-AVX-512-C
   `card/vpu`): bytes of a resident tensor pushed back to the window by the
   card's DMA (`push_data`). 32 because 9 to 16 are the slot protocol's kinds;
   the worker's dispatch matches it exactly and keeps 9 to 16 unknown on the main
   channel. Worker md5 `080e8a1c55bd2eec47d5cac6f34f786f`.
2. **The doorbell lock** (`VPU_OFF_LOCK`, window offset 15360): `doorbell` takes
   it, `wait_reply` frees it on every return, so any number of host-side clients
   can share the card's single request slot; the backend clears it when it maps
   the window (a client killed mid-request would leave it taken).
3. **`back_or_drop`** in `libggml_phi.so`: after uploading a dense matrix's card
   rows, it hands them to the router (`tridma_back`, found with `dlsym`) instead
   of `MADV_PAGEOUT`.
4. **`pin`**: locks the host rows, drops the card rows from the page cache.
5. **`libtridma.so`**: no populate for GGUF mappings, and the card-backed pages.
6. **Card boot** (Intel-Phi-7120A stack): the card no longer swaps into host
   memory (`PHI_SWAP_HOSTMEM=1` on the kernel command line turns it back on), and
   its host-memory window is 2 GiB instead of 4: 2 GiB of host RAM back.

## Running it

```
W=~/llama-phi-cuda; R=$W/runs/serve; M=~/models/Qwen3.8-27B
cd $R
nohup ./pin $M/Qwen3.8-27B-Q8_0.gguf 0.468 > pin.log 2>&1 &            # host rows locked first
nohup ./pin $M/MTP/mtp-Qwen3.8-27B-Q4_0.gguf 0 > pin-mtp.log 2>&1 &    # the draft's host tensors
LD_PRELOAD=$R/libtridma.so TG_ONLY=31 PHI_VPU_HUGEPAGES=6437 CARD_BYTES=13100000000 \
SERVE_ALIAS=qwen3.8-27b-q8_0 SERVE_CTX=204800 \
SERVE_ARGS="-md $M/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp -ngld 99" \
  bash $W/repo/scripts/serve.sh $W $M/Qwen3.8-27B-Q8_0.gguf start
```

The order matters on a 32 GB host: the pins first, then the server, with the
host's available memory watched (a guard loop killed the server below 1 GiB; it
never fired). The card worker runs from llama-phi's `scripts/worker.sh` with
`PHI_VPU_HUGEPAGES=6437`.

## Measured (11-06, RTX 3090 Ti, Ryzen 7 5800X, Xeon Phi 7120A)

Load: the server read 13,859 MiB from the disk once (the card's rows on their way
to the card); the host rows came from the locked pages.

| Request | Prompt | Prefill | Generation | MTP accepted | Card DMA out | Disk reads |
|---|---|---|---|---|---|---|
| host + card prefill (before the router) | 1392 | 16.42 tok/s | 4.47 tok/s | 18/36 | | 122 MiB |
| GPU prefill through the router, 1st | 1392 | **52.20 tok/s** (26.67 s) | 5.55 tok/s | 10/12 | 11,911 MiB | 988 MiB |
| GPU prefill through the router, 2nd | 1412 | **59.01 tok/s** (23.93 s) | 5.03 tok/s | 10/14 | 11,717 MiB | 131 MiB |

During the first: 588 faults, 1,536 chunks of 8 MiB fetched (11.39 GB at 0.71 GB/s),
1,464 recycled, 507 MiB resident at most; GPU SM 31 % mean, 47 % at most; host
iowait 1 per 1000 of CPU time; host `MemAvailable` 4.1 to 5.2 GB.

Supporting measurements:

- `K_DOWNLOAD`: 10 downloads of a 32 MiB matrix byte-identical to the upload;
  card to window 0.81 GB/s, window to GPU 13.40 GB/s (`cuda/vpu_dma -D`).
- userfaultfd under CUDA: 16,384 user-mode faults served during one `cudaMemcpy`,
  0 of 16,777,216 words wrong.
- no populate: 1024.0 MiB resident without the shim, 0.0 MiB with it.

## Limits and what is next

- **The card's DMA out is the bound.** 0.71 to 0.81 GB/s: `push_data` copies each
  960 KiB piece into the landing area with the cores, then posts one engine copy
  and waits, serially. DMA straight from the resident huge pages (physically
  contiguous 2 MiB pages whose frames the worker reads) with several descriptors
  in flight should approach the 3.2 to 3.6 GB/s the card's engine reaches; by
  arithmetic that is about 4x the prefill rate.
- **131 MiB of disk reads per prompt remain.** `pin` assumes every matrix is
  split; the backend split 369 of the 545 (the rest it never multiplies on the
  card), so their remaining rows sit unlocked. `pin` needs the backend's actual
  split set.
- Generation (5.0 to 5.6 tok/s) is bound by the host's 14 threads reading their
  15 GB share each token; the card's share and the DMA are not the limit there.
