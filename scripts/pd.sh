#!/usr/bin/env bash
# pd.sh WORK_DIR MODEL.gguf start|stop|status
#
# Two llama-servers over one model file:
#   prefill  :PD_PREFILL_PORT (8091)  the GPU (CUDA), --fit places what fits in
#            its memory, the rest stays in host RAM and is offloaded per batch;
#            2 host threads; no Xeon Phi backend
#   decode   :PD_DECODE_PORT (8092)   design A exactly as bench.sh mode a: the
#            host's 14 threads + the card's rows through libggml_phi.so, the
#            GPU hidden
# Both save and restore slots in PD_SLOTS (/dev/shm/llama-phi-pd), which is
# how tools/pd.c hands each prompt's state from the first to the second.
# The card worker must be running (scripts/worker.sh). Logs and pid files go
# to WORK_DIR/runs/pd.
set -euo pipefail
work=${1:?usage: pd.sh WORK_DIR MODEL.gguf start|stop|status}
model=${2:?usage: pd.sh WORK_DIR MODEL.gguf start|stop|status}
cmd=${3:-status}
pp=${PD_PREFILL_PORT:-8091}
dp=${PD_DECODE_PORT:-8092}
slots=${PD_SLOTS:-/dev/shm/llama-phi-pd}
ctx=${PD_CTX:-8192}
runs=$work/runs/pd
bin=$work/llama.cpp/build-native-cuda/bin/llama-server
mkdir -p "$runs"

wait_up() {   # name pid log
    for _ in $(seq 1 600); do
        grep -q 'listening on' "$3" && { echo "$1 server up (pid $2)"; return 0; }
        kill -0 "$2" 2>/dev/null || { echo "$1 server exited, see $3" >&2; return 1; }
        sleep 1
    done
    echo "$1 server not up after 600 s, see $3" >&2
    return 1
}

case "$cmd" in
    start)
        mkdir -p "$slots"
        env CUDA_VISIBLE_DEVICES= \
            GGML_BACKEND_PATH="$work/Intel-Phi-AVX-512-C/host/asm/out/libggml_phi.so" \
            PHI_GGML_CARDS=${PHI_GGML_CARDS:-0} PHI_GGML_THREADS=120 \
            PHI_GGML_CARD_BYTES=${CARD_BYTES:-10500000000} PHI_GGML_OFFLOAD=1 \
            PHI_GGML_HOST_THREADS=${HOST_THREADS:-14} \
            nohup "$bin" -m "$model" -t "${HOST_THREADS:-14}" -ngl 0 -ot no_such_tensor=CPU \
                -c "$ctx" -np 1 --slot-save-path "$slots" --host 127.0.0.1 --port "$dp" \
                > "$runs/decode.log" 2>&1 &
        echo $! > "$runs/decode.pid"
        wait_up decode "$(cat "$runs/decode.pid")" "$runs/decode.log"
        nohup "$bin" -m "$model" -t 2 -tb 2 -b 2048 -ub 2048 -c "$ctx" -np 1 \
            --slot-save-path "$slots" --host 127.0.0.1 --port "$pp" \
            > "$runs/prefill.log" 2>&1 &
        echo $! > "$runs/prefill.pid"
        wait_up prefill "$(cat "$runs/prefill.pid")" "$runs/prefill.log" ;;
    stop)
        for s in prefill decode; do
            [ -f "$runs/$s.pid" ] || continue
            pid=$(cat "$runs/$s.pid")
            if kill -0 "$pid" 2>/dev/null && grep -q llama-server "/proc/$pid/cmdline"; then
                kill "$pid" && echo "$s server stopped (pid $pid)"
            fi
            mv "$runs/$s.pid" "$runs/$s.pid.old"
        done ;;
    status)
        for s in prefill decode; do
            p=$pp; [ $s = decode ] && p=$dp
            if curl -sf "http://127.0.0.1:$p/health" > /dev/null; then echo "$s :$p up"; else echo "$s :$p down"; fi
        done ;;
    *)
        echo "pd.sh: start, stop or status" >&2; exit 2 ;;
esac
