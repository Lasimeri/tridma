# evict.c

`evict FILE...` drops each file's pages from the host's page cache with
`posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)`, without root (no
`drop_caches`). Used to unload a model file from host RAM after its
server stops: the server's process memory goes when it exits, but the
mmap'd model stays cached until the kernel needs the RAM.

It prints the cached bytes before and after, counted with `mincore` over a
read-only shared mapping of the file. Pages that another process still maps
or that are dirty are not dropped, so a non-zero "after" means someone still
uses the file.

Run: `tcc -run tools/evict.c FILE...` (or compile it with gcc).
