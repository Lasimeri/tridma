# tridma

DMA among three processing units at once (a host CPU, an NVIDIA GPU and an
Intel Xeon Phi 7120A, Knights Corner), every link in both directions, with no
blocking wait anywhere; and the llama.cpp serving configuration it was built
beside: Qwen3.8-27B Q8_0 with its weights in host RAM and on the Xeon Phi, the
GPU holding the context and running prefill.

It builds on [llama-phi](https://github.com/Lasimeri/llama-phi) (the Xeon Phi as
a matrix co-processor for unmodified llama.cpp): its `scripts/setup.sh` makes the
work directory (`WORK`) every script here uses, with llama.cpp, the patched
[Intel-Phi-AVX-512-C](https://github.com/Lasimeri/Intel-Phi-AVX-512-C) and its
card worker. llama.cpp is not modified.

| File | What it is |
|---|---|
| [`cuda/tridma.cu`](cuda/tridma.cu) | one persistent CUDA kernel: host <-> GPU rings, GPU <-> card worker requests, both copy engines, all at once ([tridma.md](cuda/tridma.md)) |
| [`cuda/vpu_dma.cu`](cuda/vpu_dma.cu) | the GPU driving the card worker through its host window ([vpu_dma.md](cuda/vpu_dma.md)); `-F` frees the card |
| [`scripts/serve.sh`](scripts/serve.sh) | one llama-server: weights in host RAM + on the card, the GPU for prefill and the KV cache |
| [`docs/config.md`](docs/config.md) | the serving configuration: command, environment, where every buffer sits, measurements, omp |
| [`docs/cuda.md`](docs/cuda.md) | the GPU beside the card: the layouts tried, the scheduler setting prefill needs, measurements |
| [`docs/design.md`](docs/design.md) | llama-phi's design (the card's side), for reference |
| [`scripts/pd.sh`](scripts/pd.sh), [`tools/pd.c`](tools/pd.c) | two servers: prefill on the GPU, generation on the card, overlapped across prompts |
| [`tools/evict.c`](tools/evict.c) | drops a model file from the page cache without root ([evict.md](tools/evict.md)) |
| [`omp/models.yml`](omp/models.yml), [`scripts/omp-phi`](scripts/omp-phi) | an omp profile's provider for the server, and a launcher that starts omp only when the server holds the port |
| [`docs/router.md`](docs/router.md) | the DMA router: the card's memory backs the card's weight rows (page faults, `K_DOWNLOAD`), host rows pinned, every prompt on the GPU, no disk reads during inference |
| [`tools/tridma_router.c`](tools/tridma_router.c), [`tools/pin.c`](tools/pin.c), [`tools/nopopulate.c`](tools/nopopulate.c) | the router's LD_PRELOAD library, the residency daemon, the no-populate shim |
| [`cuda/uffd_check.cu`](cuda/uffd_check.cu) | the premise: CUDA pageable copies served by an unprivileged userfaultfd |
| [`scripts/build.sh`](scripts/build.sh) | builds the programs into `WORK/runs` (tridma, vpu_dma, pd, evict, pin, libtridma.so) |

## How the three meet

All three share pinned host memory: the card reaches the worker's window
(`/dev/shm/phi-hostmem`) with its own DMA engine through the IOMMU, the GPU maps
the same pages and reaches them with its copy engines or its SMs' PCIe reads and
writes, the CPU reads and writes them directly. Every wait is a spin on a
sequence word (`ld.acquire.sys` on the GPU, `pause` on the host): no thread
sleeps in the kernel and no CPU goes into I/O wait.

```
┌────────────────────────────┐   ┌────────────────────────────┐   ┌────────────────────────────┐
│ RTX 3090 Ti (Gen4 x8)      │   │ Ryzen 7 5800X + DDR4       │   │ Xeon Phi 7120A (Gen2 x8)   │
│ k_tri persistent kernel    │   │ host_thread x1 each way    │   │ phi-vpu-worker 120 thr     │
│ h2g/g2h blocks, card block │   │ spin on seq, no syscalls   │   │ cdma ch 7 pull / push      │
│ 2 copy engines (streams)   │   │ iowait 0 jiffies (all)     │   │ K_MATMUL, 2 B/D halves     │
└────────────────────────────┘   └────────────────────────────┘   └────────────────────────────┘
        ▲ │                         ▲ │                             ▲ │                         
        │ ▼                         │ ▼                             │ ▼                         
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ pinned host memory: rings (head/tail on own 128-B lines, slot = seq + sum + payload)         │
│ and the card window /dev/shm/phi-hostmem (req 64, reply 256, desc 13312, B 640M, D 704M)     │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
 alone: rings 3.68 in / 11.02 out GB/s, copy engines 13.37 / 13.18 GB/s, card 3146 req/s        
 all at once: GPU link 7.92 in + 11.44 out = 19.36 GB/s, card 1531 req/s, 0 errors, iowait 0    
```

## Measured

RTX 3090 Ti on PCIe Gen4 x8, Ryzen 7 5800X with 32 GB, Xeon Phi 7120A on PCIe
Gen2 x8; `tridma`, 3 s per phase, a 4096 x 2048 float32 matrix resident on the card.

| Phase | Into the GPU | Out of the GPU | Card | Errors | Host iowait |
|---|---|---|---|---|---|
| rings alone (4 each way) | 3.68 GB/s | 11.02 GB/s | | 0 / 0 | 1 jiffy |
| copy engines, each alone | 13.37 GB/s | 13.18 GB/s | | | 0 |
| copy engines, both at once | 13.16 GB/s | 10.70 GB/s | | | 0 |
| card channel alone | | | 3146.3 req/s, 317.8 us each | 0 | 1 jiffy |
| everything at once | 7.92 GB/s | 11.44 GB/s | 1531.3 req/s, 653.1 us each | 0 on every channel | 0 |

Both copy engines together move 1.81 times one direction alone: the GPU's link is
full duplex. Every card result matched the first multiply's checksum, which
matches an fp64 reference to 9.661e-08 of the largest value. Limits as measured:
the worker serialises pull, compute and push within a request, so the card's own
link is not yet full duplex; under the full load the card's request rate halves
while its own traffic is 12.5 / 25 MB/s (contention for host memory or the PCIe
root, not isolated). Not wired into llama.cpp's graph.

The serving configuration (a 699-token prompt): prefill 24.54 tok/s with the
weights streaming to the GPU at 822 MB/s, generation 8 tokens in 5.35 s, the MTP
draft accepted 5 of 6; bound by host RAM. Details in [docs/config.md](docs/config.md).

## Use

```
# once: llama-phi's setup (llama.cpp, the patched Intel-Phi-AVX-512-C, the worker)
git clone https://github.com/Lasimeri/llama-phi && llama-phi/scripts/setup.sh WORK
llama-phi/scripts/worker.sh WORK            # start the card worker

scripts/build.sh WORK                       # tridma, vpu_dma, pd, evict into WORK/runs
WORK/runs/tri/tridma -M host                # rings, GPU <-> host only (no card)
WORK/runs/tri/tridma -M ce                  # copy engines alone and together
WORK/runs/tri/tridma -M all                 # everything, the card included (card to itself)
scripts/serve.sh WORK MODEL.gguf start      # the serving layout (docs/config.md)
```

`tridma -M card` and `-M all` need the card to themselves: they refuse while any
other program maps its window, so stop a server first (`scripts/serve.sh WORK x
stop`).

## License

MIT: [LICENSE](LICENSE).
