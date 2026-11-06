# nopopulate.c

An `LD_PRELOAD` shim that keeps llama.cpp from reading a whole GGUF into memory at
load. Superseded by [`tridma_router.c`](tridma_router.md), which contains the same
code; kept as the minimal form.

llama.cpp's loader maps the model with `MAP_POPULATE` and advises
`POSIX_MADV_WILLNEED` over it (`src/llama-mmap.cpp`, the mapping's constructor;
`init_mappings(true, ...)` from `src/llama-model.cpp`). The "lazy" ranges that
would skip the populate apply only to tensors an architecture marks, and the NUMA
path only on more than one node, so with an ordinary model every byte is read
from the disk into the page cache. When the weights are split between pinned host
memory and a Xeon Phi, that read is exactly what must not happen.

For mappings of files whose names end in `.gguf` (found by reading
`/proc/self/fd/N`), the shim clears `MAP_POPULATE` and turns `MADV_WILLNEED` and
`POSIX_MADV_WILLNEED` into no-ops; every other mapping and advice passes through.
`NOPOPULATE_VERBOSE=1` logs what it changed.

Measured on a real GGUF (the 1.37 GB MTP draft model), 1 GiB mapped with
`MAP_POPULATE` and `WILLNEED`, resident bytes counted with `mincore`: 1024.0 MiB
without the shim, 0.0 MiB with it.

Build: `gcc -O2 -Wall -Wextra -shared -fPIC -o libnopopulate.so nopopulate.c -ldl`.
