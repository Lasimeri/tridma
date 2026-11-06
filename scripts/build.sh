#!/usr/bin/env bash
# build.sh WORK_DIR: build tridma's programs against a llama-phi work directory
# (github.com/Lasimeri/llama-phi scripts/setup.sh: llama.cpp, the patched
# Intel-Phi-AVX-512-C and its card worker). The CUDA programs include the worker's own
# protocol headers (card/vpu/vpu_proto.h, vpu_matmul.h), which need the llama-phi patch
# (VPU_OFF_CDMA_STATS). Binaries land where the scripts look for them:
#   WORK/runs/tri/tridma   WORK/runs/cuda/vpu_dma (serve.sh stop)   WORK/runs/pd/bin/pd
#   WORK/runs/tools/evict
# NVCC (default nvcc on PATH, else /opt/cuda/bin/nvcc), ARCH (default sm_86).
set -euo pipefail
work=${1:?usage: build.sh WORK_DIR}
here=$(cd "$(dirname "$0")/.." && pwd)
vpu=$work/Intel-Phi-AVX-512-C/card/vpu
[ -f "$vpu/vpu_matmul.h" ] || { echo "build.sh: $vpu/vpu_matmul.h not found (run llama-phi's setup.sh first)" >&2; exit 1; }
grep -q VPU_OFF_CDMA_STATS "$vpu/vpu_proto.h" || { echo "build.sh: $vpu/vpu_proto.h lacks VPU_OFF_CDMA_STATS (the llama-phi patch)" >&2; exit 1; }
nvcc=${NVCC:-$(command -v nvcc || echo /opt/cuda/bin/nvcc)}
arch=${ARCH:-sm_86}
mkdir -p "$work/runs/tri" "$work/runs/cuda" "$work/runs/pd/bin" "$work/runs/tools"
flags=(-O3 -arch="$arch" -std=c++17 -Xcompiler -Wall,-Wextra -I"$vpu")
"$nvcc" "${flags[@]}" -o "$work/runs/tri/tridma" "$here/cuda/tridma.cu" -lpthread
"$nvcc" "${flags[@]}" -o "$work/runs/cuda/vpu_dma" "$here/cuda/vpu_dma.cu"
gcc -O2 -Wall -Wextra -Wpedantic -std=gnu11 -pthread -o "$work/runs/pd/bin/pd" "$here/tools/pd.c"
gcc -O2 -Wall -Wextra -o "$work/runs/tools/evict" "$here/tools/evict.c"
mkdir -p "$work/runs/serve"
gcc -O2 -Wall -Wextra -Wpedantic -std=gnu11 -o "$work/runs/serve/pin" "$here/tools/pin.c" -lm
gcc -O2 -Wall -Wextra -shared -fPIC -I"$vpu" -o "$work/runs/serve/libtridma.so" "$here/tools/tridma_router.c" -ldl -lpthread
for b in tri/tridma cuda/vpu_dma pd/bin/pd tools/evict serve/pin serve/libtridma.so; do
    printf '%-28s %8s bytes\n' "runs/$b" "$(wc -c < "$work/runs/$b")"
done
