// nopopulate: an LD_PRELOAD shim that keeps a model file from being read whole into memory
// at load. llama.cpp maps a GGUF with MAP_POPULATE and advises POSIX_MADV_WILLNEED over it
// (src/llama-mmap.cpp), which reads every byte from the disk into the page cache up front.
// With the weights split between the host (pinned by tools/pin.c) and the Xeon Phi (uploaded
// by libggml_phi.so, then paged out), that full read is exactly what must not happen. For
// mappings of files whose name ends in ".gguf" only, this shim clears MAP_POPULATE and turns
// MADV_WILLNEED / POSIX_MADV_WILLNEED into no-ops; every other mapping and advice passes
// through unchanged. See nopopulate.md.
//
// build: gcc -O2 -Wall -Wextra -shared -fPIC -o libnopopulate.so nopopulate.c -ldl
// use:   LD_PRELOAD=/path/libnopopulate.so llama-server ...   (NOPOPULATE_VERBOSE=1 logs)
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MAX_MAPS 64

static struct { uintptr_t a, b; } maps[MAX_MAPS];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_madvise)(void *, size_t, int);
static int (*real_posix_madvise)(void *, size_t, int);
static int (*real_munmap)(void *, size_t);

static int verbose(void) { const char *v = getenv("NOPOPULATE_VERBOSE"); return v && *v == '1'; }

static int is_gguf(int fd) {
    char link[64], path[4096];
    snprintf(link, sizeof link, "/proc/self/fd/%d", fd);
    ssize_t n = readlink(link, path, sizeof path - 1);
    if (n < 5) return 0;
    path[n] = 0;
    return !strcmp(path + n - 5, ".gguf");
}

static int in_gguf(const void *p, size_t len) {
    uintptr_t a = (uintptr_t) p, b = a + len;
    int hit = 0;
    pthread_mutex_lock(&mu);
    for (int i = 0; i < MAX_MAPS && !hit; i++)
        if (maps[i].b && a < maps[i].b && b > maps[i].a) hit = 1;
    pthread_mutex_unlock(&mu);
    return hit;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = (void *(*)(void *, size_t, int, int, int, off_t)) dlsym(RTLD_NEXT, "mmap");
    const int gguf = fd >= 0 && is_gguf(fd);
    if (gguf && (flags & MAP_POPULATE)) {
        flags &= ~MAP_POPULATE;
        if (verbose()) fprintf(stderr, "nopopulate: MAP_POPULATE cleared on a %zu-byte GGUF mapping\n", len);
    }
    void *r = real_mmap(addr, len, prot, flags, fd, off);
    if (gguf && r != MAP_FAILED) {
        pthread_mutex_lock(&mu);
        for (int i = 0; i < MAX_MAPS; i++)
            if (!maps[i].b) { maps[i].a = (uintptr_t) r; maps[i].b = (uintptr_t) r + len; break; }
        pthread_mutex_unlock(&mu);
    }
    return r;
}
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off_t off) __attribute__((alias("mmap")));

int munmap(void *addr, size_t len) {
    if (!real_munmap) real_munmap = (int (*)(void *, size_t)) dlsym(RTLD_NEXT, "munmap");
    pthread_mutex_lock(&mu);
    for (int i = 0; i < MAX_MAPS; i++)
        if (maps[i].b && maps[i].a == (uintptr_t) addr) maps[i].a = maps[i].b = 0;
    pthread_mutex_unlock(&mu);
    return real_munmap(addr, len);
}

int madvise(void *addr, size_t len, int advice) {
    if (!real_madvise) real_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "madvise");
    if (advice == MADV_WILLNEED && in_gguf(addr, len)) {
        if (verbose()) fprintf(stderr, "nopopulate: MADV_WILLNEED dropped (%zu bytes of a GGUF)\n", len);
        return 0;
    }
    return real_madvise(addr, len, advice);
}

int posix_madvise(void *addr, size_t len, int advice) {
    if (!real_posix_madvise) real_posix_madvise = (int (*)(void *, size_t, int)) dlsym(RTLD_NEXT, "posix_madvise");
    if (advice == POSIX_MADV_WILLNEED && in_gguf(addr, len)) {
        if (verbose()) fprintf(stderr, "nopopulate: POSIX_MADV_WILLNEED dropped (%zu bytes of a GGUF)\n", len);
        return 0;
    }
    return real_posix_madvise(addr, len, advice);
}
