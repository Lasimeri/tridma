// evict FILE...: drop each file's pages from the page cache (posix_fadvise
// POSIX_FADV_DONTNEED over the whole file). No root needed; pages another
// process still maps or has dirty stay. Prints the cached size before and
// after, from mincore over a read-only mapping. See evict.md.
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static long long cached(int fd, off_t size) {
    if (size == 0) return 0;
    void *p = mmap(NULL, (size_t) size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) return -1;
    long pg = sysconf(_SC_PAGESIZE);
    size_t n = ((size_t) size + (size_t) pg - 1) / (size_t) pg;
    unsigned char *v = malloc(n);
    long long res = -1;
    if (v && mincore(p, (size_t) size, v) == 0) {
        res = 0;
        for (size_t i = 0; i < n; i++) res += v[i] & 1;
        res *= pg;
    }
    free(v);
    munmap(p, (size_t) size);
    return res;
}

int main(int argc, char **argv) {
    int rc = 0;
    if (argc < 2) { fprintf(stderr, "usage: evict FILE...\n"); return 2; }
    for (int i = 1; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) < 0) { perror(argv[i]); rc = 1; if (fd >= 0) close(fd); continue; }
        long long before = cached(fd, st.st_size);
        int e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        long long after = cached(fd, st.st_size);
        if (e) { fprintf(stderr, "%s: posix_fadvise: %s\n", argv[i], strerror(e)); rc = 1; }
        printf("%s: cached %.2f GB -> %.2f GB of %.2f GB\n", argv[i], before / 1e9, after / 1e9, st.st_size / 1e9);
        close(fd);
    }
    return rc;
}
