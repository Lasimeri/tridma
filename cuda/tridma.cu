// tridma.cu: one persistent CUDA kernel moving data among the three processing units at
// once (the host CPU, this GPU and the Xeon Phi's VPU worker), every link in both
// directions, and every wait a spin on a sequence word in memory, so nothing blocks in the
// kernel and no CPU goes into I/O wait. See tridma.md.
//
//   host <-> GPU   lock-free rings in pinned, mapped host memory: host threads produce and
//                  consume, GPU blocks consume and produce, every slot checksummed
//   GPU  <-> card  the worker's matrix service through its host window, double-buffered:
//                  while the card computes request i, the GPU reads the results of i - 1
//                  and writes the activations of i + 1 (vpu_proto.h, vpu_matmul.h)
//   copy engines   both of the GPU's engines, host -> GPU and GPU -> host at the same time
//
// Modes: host (rings), ce (copy engines), card (card channel), all (everything at once).
// host and ce never touch the card; card and all need it to themselves.
#include <cuda_runtime.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <vector>

#define _Static_assert(c, m) static_assert(c, m)
#include "vpu_proto.h"
#include "vpu_matmul.h"
#undef _Static_assert

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "tridma: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

typedef unsigned long long u64;

// ---- shared layout ----

// One ring: the producer owns head, the consumer owns tail, each on its own 128-byte line
// (no false sharing between a GPU block and a host thread); then n slots of slot bytes,
// each a 64-byte header (sequence, checksum) and the payload.
struct ring_hdr { u64 head; char pad0[120]; u64 tail; char pad1[120]; };
struct slot_hdr { u64 seq; u64 sum; char pad[48]; };
static_assert(sizeof(ring_hdr) == 256 && sizeof(slot_hdr) == 64, "ring layout");

// Control block in pinned host memory: the stop flag the host raises and the GPU's tallies.
struct ctl {
    u64 stop;                 char p0[120];
    u64 h2g_slots, h2g_err;   char p1[112];
    u64 g2h_slots;            char p2[120];
    u64 card_reqs, card_err, card_bad_d; u64 card_ns; char p3[96];
};

// ---- device helpers ----

__host__ __device__ __forceinline__ u64 mix(u64 s, u64 j) {
    u64 x = s * 0x9E3779B97F4A7C15ull ^ (j + 1) * 0xC2B2AE3D27D4EB4Full;
    x ^= x >> 31; x *= 0xD6E8FEB86659FD93ull; x ^= x >> 29;
    return x;
}

__device__ __forceinline__ u64 ld_acq(const volatile u64 *p) {
    u64 v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ u64 ld_rlx(const volatile u64 *p) {
    u64 v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_rel(volatile u64 *p, u64 v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ u64 gtimer() {
    u64 t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// block-wide sum of one u64 per thread (blockDim a multiple of 32, at most 1024)
__device__ u64 block_sum(u64 v) {
    __shared__ u64 part[32];
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0;
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    }
    __syncthreads();
    return v;   // valid in thread 0
}

struct kparams {
    unsigned char *rings;     // device pointer of the ring area
    unsigned slot, nslot;     // slot bytes, slots a ring
    int h2g, g2h;             // rings (= blocks) each way
    ctl *c;                   // device pointer of the control block
    // card channel (card / all modes)
    int card;
    unsigned char *win;       // device pointer of the window
    vpu_matmul mm;            // the multiply; b_off / d_off alternate between two halves
    u64 b_half, d_half, threads, d_bytes, d_ref_sum;
    const float *b;           // activations in GPU memory
    float *d;                 // results land here
};

__device__ unsigned char *ring_base(const kparams &k, int r) {
    return k.rings + (size_t) r * (sizeof(ring_hdr) + (size_t) k.slot * k.nslot);
}

// host -> GPU: wait for a full slot, read it with 16-byte uncached loads, check it, free it
__device__ void h2g_block(const kparams &k, int r) {
    __shared__ u64 s_go;
    ring_hdr *h = (ring_hdr *) ring_base(k, r);
    unsigned char *slots = (unsigned char *) (h + 1);
    const unsigned words2 = (k.slot - 64) / 16;
    u64 tail = 0, done = 0, err = 0;
    for (;;) {
        if (threadIdx.x == 0) {
            s_go = 0;
            for (;;) {
                if (ld_acq(&h->head) > tail) { s_go = 1; break; }
                if (ld_rlx(&k.c->stop)) break;
            }
        }
        __syncthreads();
        if (!s_go) break;
        const unsigned char *sl = slots + (size_t) (tail % k.nslot) * k.slot;
        const ulonglong2 *p = (const ulonglong2 *) (sl + 64);
        // four independent loads in flight per thread: a host read is a PCIe round trip, so
        // one load at a time leaves the link idle while it waits
        u64 sum = 0;
        const unsigned stride = blockDim.x;
        unsigned i = threadIdx.x;
        for (; i + 3 * stride < words2; i += 4 * stride) {
            ulonglong2 v0 = __ldcv(p + i), v1 = __ldcv(p + i + stride);
            ulonglong2 v2 = __ldcv(p + i + 2 * stride), v3 = __ldcv(p + i + 3 * stride);
            sum += v0.x + v0.y + v1.x + v1.y + v2.x + v2.y + v3.x + v3.y;
        }
        for (; i < words2; i += stride) {
            ulonglong2 v = __ldcv(p + i);
            sum += v.x + v.y;
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0) {
            const volatile slot_hdr *sh = (const volatile slot_hdr *) sl;
            u64 want_seq = ((u64) r << 40) | tail;
            if (sh->sum != sum || sh->seq != want_seq) err++;
            st_rel(&h->tail, tail + 1);
        }
        tail++; done++;
        __syncthreads();
    }
    if (threadIdx.x == 0) { atomicAdd(&k.c->h2g_slots, done); atomicAdd(&k.c->h2g_err, err); }
}

// GPU -> host: wait for a free slot, fill it (mix pattern), checksum it, fence, publish
__device__ void g2h_block(const kparams &k, int r) {
    __shared__ u64 s_go;
    ring_hdr *h = (ring_hdr *) ring_base(k, k.h2g + r);
    unsigned char *slots = (unsigned char *) (h + 1);
    const unsigned words2 = (k.slot - 64) / 16;
    u64 head = 0;
    for (;;) {
        if (threadIdx.x == 0) {
            s_go = 0;
            for (;;) {
                if (ld_rlx(&k.c->stop)) break;
                if (head - ld_acq(&h->tail) < k.nslot) { s_go = 1; break; }
            }
        }
        __syncthreads();
        if (!s_go) break;
        unsigned char *sl = slots + (size_t) (head % k.nslot) * k.slot;
        ulonglong2 *p = (ulonglong2 *) (sl + 64);
        const u64 seq = ((u64) (64 + r) << 40) | head;
        u64 sum = 0;
        for (unsigned i = threadIdx.x; i < words2; i += blockDim.x) {
            ulonglong2 v = make_ulonglong2(mix(seq, 2ull * i), mix(seq, 2ull * i + 1));
            p[i] = v;
            sum += v.x + v.y;
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0) {
            volatile slot_hdr *sh = (volatile slot_hdr *) sl;
            sh->seq = seq;
            sh->sum = sum;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) st_rel(&h->head, head + 1);
        head++;
    }
    if (threadIdx.x == 0) atomicAdd(&k.c->g2h_slots, head);
}

// GPU <-> card: descriptor, request fields, fence, doorbell (vpu_proto.h: seq last)
__device__ u64 card_ring(const kparams &k, u64 b_off, u64 d_off) {
    vpu_matmul mm = k.mm;
    mm.b_off = b_off;
    mm.d_off = d_off;
    volatile u64 *d = (volatile u64 *) (k.win + VPU_OFF_MATMUL);
    const u64 *s = (const u64 *) &mm;
    for (int i = 0; i < (int) (sizeof(vpu_matmul) / 8); i++) d[i] = s[i];
    volatile vpu_request *rq = (volatile vpu_request *) (k.win + VPU_OFF_REQ);
    rq->kernel = VPU_K_MATMUL;
    rq->threads = (uint32_t) k.threads;
    rq->n = 1;
    u64 seq = rq->seq + 1;
    __threadfence_system();
    st_rel((volatile u64 *) &rq->seq, seq);
    return seq;
}

// 0 ok, else the card's status or -100 (no answer in 10 s) or -101 (stop raised)
__device__ int card_wait(const kparams &k, u64 seq) {
    volatile vpu_reply *rp = (volatile vpu_reply *) (k.win + VPU_OFF_REPLY);
    u64 t0 = gtimer();
    while (ld_acq((const volatile u64 *) &rp->seq) != seq) {
        if (gtimer() - t0 > 10000000000ull) return -100;
    }
    return rp->status;
}

// The pipeline: await i, ring i + 1, read D_i and write B_(i+2) while the card works on i + 1.
__device__ void card_block(const kparams &k) {
    __shared__ int s_stop, s_status;
    __shared__ u64 s_seq;
    const u64 qb = k.mm.k / 4, qd = k.d_bytes / 16;
    u64 done = 0, bad_d = 0;
    // both B halves hold the same activations
    for (int half = 0; half < 2; half++)
        for (u64 c = 0; c < k.mm.n; c++) {
            float4 *o = (float4 *) (k.win + k.mm.b_off + half * k.b_half + c * k.mm.nb_b);
            const float4 *src = (const float4 *) (k.b + c * k.mm.k);
            for (u64 i = threadIdx.x; i < qb; i += blockDim.x) o[i] = src[i];
        }
    __threadfence_system();
    __syncthreads();
    const u64 t_start = gtimer();
    if (threadIdx.x == 0) { s_seq = card_ring(k, k.mm.b_off, k.mm.d_off); s_stop = 0; }
    __syncthreads();
    for (u64 i = 0;; i++) {
        if (threadIdx.x == 0) {
            s_status = card_wait(k, s_seq);
            int stop = (int) ld_rlx(&k.c->stop);
            if (s_status == 0 && !stop) {
                const u64 nx = (i + 1) & 1;   // request i + 1 uses the other halves
                s_seq = card_ring(k, k.mm.b_off + nx * k.b_half, k.mm.d_off + nx * k.d_half);
            }
            s_stop = stop || s_status != 0;
        }
        __syncthreads();
        if (s_status != 0) {
            if (threadIdx.x == 0) atomicAdd(&k.c->card_err, 1);
            break;
        }
        done++;
        // D_i out of its half while the card computes i + 1, checksummed against the first
        const u64 cur = i & 1;
        const ulonglong2 *src = (const ulonglong2 *) (k.win + k.mm.d_off + cur * k.d_half);
        ulonglong2 *dst = (ulonglong2 *) k.d;
        u64 sum = 0;
        for (u64 j = threadIdx.x; j < qd; j += blockDim.x) {
            ulonglong2 v = __ldcv(src + j);
            dst[j] = v;
            sum += v.x ^ v.y;
        }
        sum = block_sum(sum);
        if (threadIdx.x == 0 && sum != k.d_ref_sum) bad_d++;
        __syncthreads();
        if (s_stop) break;
    }
    if (threadIdx.x == 0) {
        k.c->card_ns = gtimer() - t_start;
        atomicAdd(&k.c->card_reqs, done);
        atomicAdd(&k.c->card_bad_d, bad_d);
    }
}

__global__ void k_tri(kparams k) {
    int b = blockIdx.x;
    if (b < k.h2g) { h2g_block(k, b); return; }
    b -= k.h2g;
    if (b < k.g2h) { g2h_block(k, b); return; }
    if (k.card) card_block(k);
}

// one-shot card multiply for the reference checksum and the fp64 check
__global__ void k_fill(float *p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * (size_t) blockDim.x + threadIdx.x; i < n; i += (size_t) gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t) i * 0x9E3779B1u ^ seed;
        h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
        p[i] = (float) (h >> 8) * (1.0f / 8388608.0f) - 1.0f;
    }
}
__global__ void k_ref(double *d, const float *a, const float *b, u64 m, u64 n, u64 kk) {
    u64 idx = blockIdx.x * (u64) blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    u64 c = idx / m, r = idx % m;
    double s = 0;
    for (u64 i = 0; i < kk; i++) s += (double) a[r * kk + i] * (double) b[c * kk + i];
    d[c * m + r] = s;
}
__global__ void k_card_call(unsigned char *win, vpu_matmul mm, uint32_t kernel, u64 threads, int *status) {
    volatile u64 *d = (volatile u64 *) (win + VPU_OFF_MATMUL);
    const u64 *s = (const u64 *) &mm;
    for (int i = 0; i < (int) (sizeof(vpu_matmul) / 8); i++) d[i] = s[i];
    volatile vpu_request *rq = (volatile vpu_request *) (win + VPU_OFF_REQ);
    rq->kernel = kernel; rq->threads = (uint32_t) threads; rq->n = 1;
    u64 seq = rq->seq + 1;
    __threadfence_system();
    st_rel((volatile u64 *) &rq->seq, seq);
    volatile vpu_reply *rp = (volatile vpu_reply *) (win + VPU_OFF_REPLY);
    u64 t0 = gtimer();
    while (ld_acq((const volatile u64 *) &rp->seq) != seq)
        if (gtimer() - t0 > 30000000000ull) { *status = -100; return; }
    *status = rp->status;
}
__global__ void k_dsum(const unsigned char *win, u64 off, u64 bytes, u64 *out) {
    const ulonglong2 *p = (const ulonglong2 *) (win + off);
    u64 s = 0;
    for (u64 j = threadIdx.x; j < bytes / 16; j += blockDim.x) { ulonglong2 v = __ldcv(p + j); s += v.x ^ v.y; }
    s = block_sum(s);
    if (threadIdx.x == 0) *out = s;
}

// ---- host side ----

static double now_s() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void cpu_times(u64 *busy, u64 *iowait, u64 *total) {   // the "cpu" line of /proc/stat
    FILE *f = fopen("/proc/stat", "r");
    u64 v[10] = {0};
    if (f) {
        if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4],
                   &v[5], &v[6], &v[7], &v[8], &v[9]) < 5) memset(v, 0, sizeof v);
        fclose(f);
    }
    *iowait = v[4];
    *total = 0;
    for (int i = 0; i < 8; i++) *total += v[i];
    *busy = *total - v[3] - v[4];
}

struct host_args {
    unsigned char *rings;     // host pointer
    unsigned slot, nslot;
    int first, count;         // rings this thread serves
    int g2h_base;             // index of the first GPU -> host ring (= rings each way)
    int produce;              // 1: host -> GPU producer, 0: GPU -> host consumer
    volatile u64 *stop;
    u64 slots, err;
};

static unsigned char *ring_host(unsigned char *rings, unsigned slot, unsigned nslot, int r) {
    return rings + (size_t) r * (sizeof(ring_hdr) + (size_t) slot * nslot);
}

// Round-robin over its rings, never blocking: a ring that is full (producer) or empty
// (consumer) is skipped; a pass with no work spins with pause.
static void *host_thread(void *arg) {
    host_args *a = (host_args *) arg;
    std::vector<u64> pos(a->count, 0);
    const size_t words = (a->slot - 64) / 8;
    while (!*a->stop) {
        int worked = 0;
        for (int i = 0; i < a->count; i++) {
            const int r = a->first + i;
            ring_hdr *h = (ring_hdr *) ring_host(a->rings, a->slot, a->nslot, r);
            unsigned char *sl = (unsigned char *) (h + 1) + (size_t) (pos[i] % a->nslot) * a->slot;
            u64 *p = (u64 *) (sl + 64);
            slot_hdr *sh = (slot_hdr *) sl;
            if (a->produce) {
                if (pos[i] - __atomic_load_n(&h->tail, __ATOMIC_ACQUIRE) >= a->nslot) continue;
                const u64 seq = ((u64) r << 40) | pos[i];
                u64 sum = 0;
                for (size_t j = 0; j < words; j++) { u64 w = mix(seq, j); p[j] = w; sum += w; }
                sh->seq = seq;
                sh->sum = sum;
                __atomic_store_n(&h->head, pos[i] + 1, __ATOMIC_RELEASE);
            } else {
                if (__atomic_load_n(&h->head, __ATOMIC_ACQUIRE) <= pos[i]) continue;
                u64 sum = 0;
                for (size_t j = 0; j < words; j++) sum += p[j];
                const u64 want = ((u64) (64 + r - a->g2h_base) << 40) | pos[i];
                if (sum != sh->sum || sh->seq != want) a->err++;
                __atomic_store_n(&h->tail, pos[i] + 1, __ATOMIC_RELEASE);
            }
            pos[i]++;
            a->slots++;
            worked = 1;
        }
        if (!worked) __builtin_ia32_pause();
    }
    return NULL;
}

struct ce_args { void *host; void *dev; size_t bytes; int dir; volatile u64 *stop; u64 copies; cudaStream_t st; };

// copy-engine loop: chunked async copies on its own stream, synchronising every 4 chunks
static void *ce_thread(void *arg) {
    ce_args *a = (ce_args *) arg;
    const size_t chunk = 16u << 20, n = a->bytes / chunk;
    while (!*a->stop) {
        for (size_t i = 0; i < n; i++) {
            char *h = (char *) a->host + i * chunk, *d = (char *) a->dev + i * chunk;
            if (a->dir) cudaMemcpyAsync(d, h, chunk, cudaMemcpyHostToDevice, a->st);
            else cudaMemcpyAsync(h, d, chunk, cudaMemcpyDeviceToHost, a->st);
        }
        cudaStreamSynchronize(a->st);
        a->copies += n;
    }
    return NULL;
}

static int card_users(void) {   // scripts/card-users.sh: nobody but phictl/phitop may map the window
    int found = 0;
    DIR *dir = opendir("/proc");
    struct dirent *e;
    char path[300], line[4096];
    while (dir && (e = readdir(dir))) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9' || atoi(e->d_name) == (int) getpid()) continue;
        snprintf(path, sizeof path, "/proc/%s/comm", e->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        char comm[64] = "";
        if (!fgets(comm, sizeof comm, f)) comm[0] = 0;
        fclose(f);
        comm[strcspn(comm, "\n")] = 0;
        if (!strcmp(comm, "phictl") || !strcmp(comm, "phitop")) continue;
        snprintf(path, sizeof path, "/proc/%s/maps", e->d_name);
        if (!(f = fopen(path, "r"))) continue;
        while (fgets(line, sizeof line, f)) {
            size_t L = strlen(line);
            if (L > 21 && !strcmp(line + L - 21, "/dev/shm/phi-hostmem\n")) {
                fprintf(stderr, "tridma: pid %s (%s) maps the card's window\n", e->d_name, comm);
                found = 1;
                break;
            }
        }
        fclose(f);
    }
    if (dir) closedir(dir);
    return found;
}

static u64 num(const char *s, char o) {
    char *e;
    u64 v = strtoull(s, &e, 0);
    if (e == s || *e) { fprintf(stderr, "tridma: -%c wants a number, got \"%s\"\n", o, s); exit(2); }
    return v;
}

int main(int argc, char **argv) {
    const char *mode = "host";
    double secs = 3;
    int nring = 4, hthreads = 1, opt;
    unsigned slot = 1u << 20, nslot = 8;
    u64 m = 4096, kk = 2048, n = 1;
    while ((opt = getopt(argc, argv, "M:s:r:t:S:N:m:k:n:")) != -1) {
        switch (opt) {
        case 'M': mode = optarg; break;
        case 's': secs = (double) num(optarg, 's'); break;
        case 'r': nring = (int) num(optarg, 'r'); break;
        case 't': hthreads = (int) num(optarg, 't'); break;
        case 'S': slot = (unsigned) num(optarg, 'S'); break;
        case 'N': nslot = (unsigned) num(optarg, 'N'); break;
        case 'm': m = num(optarg, 'm'); break;
        case 'k': kk = num(optarg, 'k'); break;
        case 'n': n = num(optarg, 'n'); break;
        default:
            fprintf(stderr, "usage: tridma [-M host|ce|card|all] [-s secs] [-r rings each way] [-t host threads each way]"
                            " [-S slot bytes] [-N slots] [-m rows -k width -n columns]\n");
            return 2;
        }
    }
    const int want_rings = !strcmp(mode, "host") || !strcmp(mode, "all");
    const int want_ce = !strcmp(mode, "ce") || !strcmp(mode, "all");
    const int want_card = !strcmp(mode, "card") || !strcmp(mode, "all");
    if (!want_rings && !want_ce && !want_card) { fprintf(stderr, "tridma: -M host|ce|card|all\n"); return 2; }
    if (slot < 4096 || slot % 4096 || nslot < 2 || nring < 1 || nring > 32 || hthreads < 1 || hthreads > nring
        || m % 4 || kk % 16) { fprintf(stderr, "tridma: bad sizes\n"); return 2; }

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    CK(cudaSetDeviceFlags(cudaDeviceMapHost));
    printf("tridma %s on %s (%d copy engines), %.0f s\n", mode, prop.name, prop.asyncEngineCount, secs);

    // control block + rings, pinned and mapped
    ctl *c_h;
    CK(cudaHostAlloc((void **) &c_h, sizeof(ctl), cudaHostAllocMapped));
    memset(c_h, 0, sizeof(ctl));
    ctl *c_d;
    CK(cudaHostGetDevicePointer((void **) &c_d, c_h, 0));
    const size_t ring_bytes = sizeof(ring_hdr) + (size_t) slot * nslot;
    unsigned char *rings_h = NULL, *rings_d = NULL;
    const int nr_total = want_rings ? 2 * nring : 0;
    if (nr_total) {
        CK(cudaHostAlloc((void **) &rings_h, ring_bytes * nr_total, cudaHostAllocMapped));
        memset(rings_h, 0, ring_bytes * nr_total);
        CK(cudaHostGetDevicePointer((void **) &rings_d, rings_h, 0));
    }

    kparams k = {};
    k.rings = rings_d; k.slot = slot; k.nslot = nslot;
    k.h2g = want_rings ? nring : 0; k.g2h = want_rings ? nring : 0; k.c = c_d;

    // the card's window, only when the card channel runs
    unsigned char *win = NULL;
    const u64 WIN_LEN = 768ull << 20, OFF_A = 128ull << 20, OFF_B = 640ull << 20, OFF_D = 704ull << 20;
    const u64 id = 0x5452494Full;   // "TRIO"
    float *a_dev = NULL, *b_dev = NULL, *d_dev = NULL;
    if (want_card) {
        if (card_users()) { fprintf(stderr, "tridma: the card is in use; one program at a time\n"); return 1; }
        int fd = open("/dev/shm/phi-hostmem", O_RDWR);
        if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
        win = (unsigned char *) mmap(0, WIN_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (win == MAP_FAILED) { perror("mmap"); return 1; }
        if (*(volatile u64 *) (win + VPU_OFF_READY) != VPU_MAGIC) { fprintf(stderr, "tridma: no worker polling\n"); return 1; }
        CK(cudaHostRegister(win, WIN_LEN, cudaHostRegisterMapped | cudaHostRegisterPortable));
        CK(cudaHostGetDevicePointer((void **) &k.win, win, 0));
        const u64 nb_a = kk * 4, nb_b = (kk * 4 + 63) & ~63ull, d_bytes = n * m * 4;
        CK(cudaMalloc(&a_dev, m * kk * 4));
        CK(cudaMalloc(&b_dev, n * kk * 4));
        CK(cudaMalloc(&d_dev, d_bytes));
        k_fill<<<256, 256>>>(a_dev, m * kk, 0x1234567u);
        k_fill<<<256, 256>>>(b_dev, n * kk, 0x7654321u);
        CK(cudaMemcpy(win + OFF_A, a_dev, m * nb_a, cudaMemcpyDeviceToHost));
        int *st_d, st;
        CK(cudaMalloc(&st_d, sizeof(int)));
        vpu_matmul up = {};
        up.a_id = id; up.a_off = OFF_A; up.bytes = (m * nb_a + 4095) & ~4095ull;
        k_card_call<<<1, 1>>>(k.win, up, VPU_K_UPLOAD, 120, st_d);
        CK(cudaMemcpy(&st, st_d, sizeof st, cudaMemcpyDeviceToHost));
        if (st) { fprintf(stderr, "tridma: upload: status %d\n", st); return 1; }
        vpu_matmul mm = {};
        mm.a_id = id; mm.a_type = VPU_MM_F32; mm.b_type = 0; mm.m = m; mm.n = n; mm.k = kk;
        mm.nb_a = nb_a; mm.nb_b = nb_b; mm.b_off = OFF_B; mm.d_off = OFF_D;
        k.card = 1; k.mm = mm; k.b_half = 32ull << 20; k.d_half = 32ull << 20; k.threads = 120;
        k.d_bytes = d_bytes; k.b = b_dev; k.d = d_dev;
        // one plain multiply: the reference checksum of D and the fp64 check
        for (u64 c = 0; c < n; c++) CK(cudaMemcpy(win + OFF_B + c * nb_b, b_dev + c * kk, kk * 4, cudaMemcpyDeviceToHost));
        k_card_call<<<1, 1>>>(k.win, mm, VPU_K_MATMUL, 120, st_d);
        CK(cudaMemcpy(&st, st_d, sizeof st, cudaMemcpyDeviceToHost));
        if (st) { fprintf(stderr, "tridma: matmul: status %d\n", st); return 1; }
        u64 *s_d;
        CK(cudaMalloc(&s_d, 8));
        k_dsum<<<1, 512>>>(k.win, OFF_D, d_bytes, s_d);
        CK(cudaMemcpy(&k.d_ref_sum, s_d, 8, cudaMemcpyDeviceToHost));
        double *ref_d;
        CK(cudaMalloc(&ref_d, n * m * 8));
        k_ref<<<(unsigned) ((m * n + 255) / 256), 256>>>(ref_d, a_dev, b_dev, m, n, kk);
        std::vector<double> ref(n * m);
        std::vector<float> got(n * m);
        CK(cudaMemcpy(ref.data(), ref_d, n * m * 8, cudaMemcpyDeviceToHost));
        memcpy(got.data(), win + OFF_D, d_bytes);
        double ea = 0, mr = 0;
        for (u64 i = 0; i < n * m; i++) { ea = std::max(ea, fabs(got[i] - ref[i])); mr = std::max(mr, fabs(ref[i])); }
        printf("card reference multiply %llux%llu x %llu columns: max |err| / max |ref| = %.3e\n",
               (unsigned long long) m, (unsigned long long) kk, (unsigned long long) n, ea / mr);
    }

    // copy-engine buffers
    void *ce_h = NULL, *ce_d = NULL;
    const size_t ce_bytes = 128u << 20;
    if (want_ce) {
        CK(cudaHostAlloc(&ce_h, 2 * ce_bytes, cudaHostAllocDefault));
        CK(cudaMalloc(&ce_d, 2 * ce_bytes));
        memset(ce_h, 0x5a, 2 * ce_bytes);
    }

    // phase list: ce alone each way first (the duplex baseline), then the main run
    struct phase { const char *name; int ce_h2d, ce_d2h, kernel; };
    std::vector<phase> phases;
    if (want_ce) { phases.push_back({"ce host->GPU alone", 1, 0, 0}); phases.push_back({"ce GPU->host alone", 0, 1, 0}); }
    if (want_rings && !want_card && !want_ce) phases.push_back({"rings both ways", 0, 0, 1});
    if (want_rings && !want_card && !want_ce) {}
    if (!strcmp(mode, "ce")) phases.push_back({"ce both ways at once", 1, 1, 0});
    if (!strcmp(mode, "card")) phases.push_back({"card channel", 0, 0, 1});
    if (!strcmp(mode, "all")) phases.push_back({"everything at once", 1, 1, 1});

    cudaStream_t ks, s1, s2;
    CK(cudaStreamCreateWithFlags(&ks, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    const u64 cd0_in = want_card ? ((volatile u64 *) (win + VPU_OFF_CDMA_STATS))[0] : 0;
    const u64 cd0_out = want_card ? ((volatile u64 *) (win + VPU_OFF_CDMA_STATS))[1] : 0;

    for (const phase &ph : phases) {
        memset(c_h, 0, sizeof(ctl));
        if (nr_total) memset(rings_h, 0, ring_bytes * nr_total);
        u64 b0, w0, t0c, b1, w1, t1c;
        cpu_times(&b0, &w0, &t0c);
        const double t0 = now_s();
        if (ph.kernel) {
            const int blocks = k.h2g + k.g2h + (k.card ? 1 : 0);
            k_tri<<<blocks, 512, 0, ks>>>(k);
            CK(cudaGetLastError());
        }
        std::vector<pthread_t> th;
        std::vector<host_args> ha(ph.kernel && want_rings ? 2 * hthreads : 0);
        for (size_t i = 0; i < ha.size(); i++) {
            const int prod = i < (size_t) hthreads, j = (int) (prod ? i : i - hthreads);
            const int per = (nring + hthreads - 1) / hthreads, first = j * per, cnt = std::min(per, nring - first);
            ha[i] = {rings_h, slot, nslot, (prod ? 0 : nring) + first, cnt, nring, prod, &c_h->stop, 0, 0};
        }
        ce_args ca[2] = {{ce_h, ce_d, ce_bytes, 1, &c_h->stop, 0, s1},
                         {(char *) ce_h + ce_bytes, (char *) ce_d + ce_bytes, ce_bytes, 0, &c_h->stop, 0, s2}};
        for (auto &a : ha) { pthread_t t; pthread_create(&t, NULL, host_thread, &a); th.push_back(t); }
        if (ph.ce_h2d) { pthread_t t; pthread_create(&t, NULL, ce_thread, &ca[0]); th.push_back(t); }
        if (ph.ce_d2h) { pthread_t t; pthread_create(&t, NULL, ce_thread, &ca[1]); th.push_back(t); }
        usleep((useconds_t) (secs * 1e6));
        __atomic_store_n(&c_h->stop, 1, __ATOMIC_RELEASE);
        for (pthread_t t : th) pthread_join(t, NULL);
        if (ph.kernel) CK(cudaStreamSynchronize(ks));
        const double el = now_s() - t0;
        cpu_times(&b1, &w1, &t1c);
        const double pay = (double) (slot - 64);
        printf("\n== %s (%.2f s)\n", ph.name, el);
        if (ph.kernel && want_rings) {
            u64 hp = 0, hc = 0, herr = 0;
            for (size_t i = 0; i < ha.size(); i++) { if (ha[i].produce) hp += ha[i].slots; else { hc += ha[i].slots; herr += ha[i].err; } }
            printf("host -> GPU rings: %llu slots, %.2f GB/s, GPU checksum errors %llu\n",
                   (unsigned long long) c_h->h2g_slots, c_h->h2g_slots * pay / el / 1e9, (unsigned long long) c_h->h2g_err);
            printf("GPU -> host rings: %llu slots, %.2f GB/s, host checksum errors %llu\n",
                   (unsigned long long) hc, hc * pay / el / 1e9, (unsigned long long) herr);
        }
        if (ph.ce_h2d) printf("copy engine host -> GPU: %.2f GB/s\n", ca[0].copies * (16.0 * 1048576) / el / 1e9);
        if (ph.ce_d2h) printf("copy engine GPU -> host: %.2f GB/s\n", ca[1].copies * (16.0 * 1048576) / el / 1e9);
        if (ph.kernel && k.card) {
            const u64 cin = ((volatile u64 *) (win + VPU_OFF_CDMA_STATS))[0] - cd0_in;
            const u64 cout = ((volatile u64 *) (win + VPU_OFF_CDMA_STATS))[1] - cd0_out;
            printf("card channel: %llu requests (%.1f/s, %.1f us each), status errors %llu, D checksum mismatches %llu\n",
                   (unsigned long long) c_h->card_reqs, c_h->card_reqs / (c_h->card_ns * 1e-9),
                   c_h->card_ns * 1e-3 / std::max<u64>(1, c_h->card_reqs), (unsigned long long) c_h->card_err,
                   (unsigned long long) c_h->card_bad_d);
            printf("card DMA channel 7: %.2f MB/s in, %.2f MB/s out\n", cin / el / 1e6, cout / el / 1e6);
        }
        const double tot = (double) (t1c - t0c);
        printf("host CPUs: busy %.1f %%, iowait %.2f %% (%llu jiffies)\n", 100.0 * (b1 - b0) / tot,
               100.0 * (w1 - w0) / tot, (unsigned long long) (w1 - w0));
    }

    if (want_card) {
        int *st_d, st;
        CK(cudaMalloc(&st_d, sizeof(int)));
        vpu_matmul fr = {};
        fr.a_id = id;
        k_card_call<<<1, 1>>>(k.win, fr, VPU_K_FREE, 120, st_d);
        CK(cudaMemcpy(&st, st_d, sizeof st, cudaMemcpyDeviceToHost));
        printf("\nfree of its own upload: status %d\n", st);
        CK(cudaHostUnregister(win));
        munmap(win, WIN_LEN);
    }
    return 0;
}
