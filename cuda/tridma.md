# tridma.cu

One persistent CUDA kernel that moves data among the three processing units at
once (the host CPU, the GPU and the Xeon Phi's VPU worker), every link in both
directions, with no blocking wait anywhere: every wait is a spin on a sequence
word in memory (`ld.acquire.sys` on the GPU, `pause` on the host), so no CPU
goes into I/O wait and no thread sleeps in the kernel.

## Where the data goes

All three meet in pinned host memory. The card reaches the worker's window
(`/dev/shm/phi-hostmem`) with its DMA channel 7 through the IOMMU; the GPU maps
the same pages (`cudaHostRegister`, mapped) and reaches them with its copy
engines or with its SMs' own PCIe reads and writes; the host CPU reads and
writes them directly. A direct GPU-to-card path would need peer mappings of
the card's memory and is not attempted, so each crossing is one DMA into host
RAM and one out.

| Channel | Producer -> consumer | Mechanism |
|---|---|---|
| host -> GPU rings | host threads -> GPU blocks | lock-free rings in pinned mapped memory |
| GPU -> host rings | GPU blocks -> host threads | the same, the other way |
| card channel | GPU <-> card | the worker's matrix service, double-buffered |
| copy engines | host <-> GPU | both engines at once (`asyncEngineCount` 2 on a 3090 Ti) |

### Rings

Each ring is a 256-byte header (the producer's `head` and the consumer's
`tail`, each on its own 128-byte line, so a GPU block and a host thread never
write the same line) and `N` slots of `S` bytes; a slot is a 64-byte header
(sequence number, checksum) and the payload. Producer: wait until
`head - tail < N`, fill the slot, write the header, then release-store
`head + 1` (on the GPU: every thread fences with `__threadfence_system`,
the block barriers, thread 0 stores with `st.release.sys`). Consumer: wait
until `head > tail` (acquire), read the payload (on the GPU `__ldcv`, never
a stale line, four independent 16-byte loads in flight per thread), check the
sequence number and the sum of the payload's words against the header,
release-store `tail + 1`. One GPU block per ring; host threads serve their
rings round-robin and skip a full or empty one instead of waiting on it.

### Card channel

One block drives the worker (`vpu_proto.h`, `vpu_matmul.h`: descriptor at
13312, request at 64, the sequence number written last, the reply's echoed)
through two halves of the activation area (640 MiB and +32 MiB) and of the
result area (704 MiB and +32 MiB): when request `i` is answered, request
`i + 1` is rung at once on the other halves, and the block reads `D_i` and
writes the next activations while the card pulls, computes and pushes
`i + 1`. Every `D_i` is checksummed against the first multiply's, which is
checked against an fp64 reference on the GPU. The program uploads its own
matrix (id "TRIO") and frees only that id; it refuses to start while anyone
but the stack's `phictl` or `phitop` maps the window (one program per card).

## Modes

`tridma -M host|ce|card|all [-s secs] [-r rings each way] [-t host threads each way] [-S slot bytes] [-N slots] [-m rows -k width -n columns]`

| Mode | Runs | Needs the card |
|---|---|---|
| `host` | the rings both ways | no |
| `ce` | each copy engine alone, then both at once | no |
| `card` | the card channel | yes |
| `all` | rings + copy engines + card channel at once | yes |

Each phase prints the rates, the checksum errors, and the host CPUs' busy and
iowait share from `/proc/stat` over the phase.

## Measured (RTX 3090 Ti on PCIe Gen4 x8, Ryzen 7 5800X; 3 s phases)

| Phase | host -> GPU | GPU -> host | errors | host iowait |
|---|---|---|---|---|
| rings, 4 each way, 1 host thread each way | 3.68 GB/s | 11.02 GB/s | 0 / 0 | 1 jiffy |
| rings, 4 host threads each way | 3.67 GB/s | 11.01 GB/s | 0 / 0 | 3 jiffies |
| copy engine alone | 13.36 GB/s | 13.15 GB/s | | 4 / 1 jiffies |
| copy engines both at once | 13.16 GB/s | 10.70 GB/s | | 0 jiffies |

An idle 3 s sample of the same host showed 1 jiffy of iowait (it was
swapping), so the phases' 0 to 4 jiffies are the host's own noise. Both copy
engines together move 23.86 GB/s, 1.81 times one direction alone: the link
is full duplex. The rings' host -> GPU side was 1.30 GB/s with one 16-byte
load in flight per thread and 3.68 GB/s with four. The card channel and
`all` on the card:

| Phase | result |
|---|---|
| card channel alone | 9439 requests in 3 s = 3146.3/s, 317.8 us each (serial loop: 323.1 us), 0 status errors, 0 result mismatches; fp64 check 9.661e-08 |
| everything at once | rings 3.15 / 7.92 GB/s, copy engines 4.77 / 3.52 GB/s (host -> GPU 7.92, GPU -> host 11.44, 19.36 GB/s across the link), card 4594 requests = 1531.3/s, 653.1 us each; 0 errors on every channel; host iowait 0 jiffies |

Pipelining hides only the GPU side (about 21 us): the worker serialises pull,
compute and push within a request (301.6 us measured). Under the full load the
card's request rate halves while its own traffic is 12.5 / 25 MB/s, so it loses to
contention for host memory and the PCIe root (or to delayed doorbell polls), not
to its link; which of the two is not isolated.

Build: `nvcc -O3 -arch=sm_86 -std=c++17 -Xcompiler -Wall,-Wextra
-I<Intel-Phi-AVX-512-C>/card/vpu -o tridma tridma.cu -lpthread`.
