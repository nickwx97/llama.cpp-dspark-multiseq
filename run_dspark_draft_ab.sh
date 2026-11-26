
#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# DSpark draft-count A/B test
# Order: 2 -> 4 -> 3
# Uses the existing GPU Top-K runtime bundle.
# ============================================================

REPO="$HOME/llama.cpp"
AB_ROOT="$HOME/dspark-topk-ab.2h9EHe"
RUNTIME="$AB_ROOT/gpu-topk"
SERVER="$RUNTIME/llama-server"

MODEL="/mnt/nfs/llm/models/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf"

DRAFT="/mnt/nfs/llm/models/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"

PORT=8080
HOST=127.0.0.1
HEALTH_URL="http://${HOST}:${PORT}/health"

BENCHMARK="$REPO/multi_navid_bench.py"
RESULT_DIR="$AB_ROOT/results/draft-count-$(date +%Y%m%d-%H%M%S)"

VARIANTS=(2 4 3)

SERVER_PID=""

mkdir -p "$RESULT_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server PID $SERVER_PID..."
        kill -TERM "$SERVER_PID" 2>/dev/null || true

        for _ in $(seq 1 60); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done

        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server did not stop gracefully. Sending SIGKILL."
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi

        wait "$SERVER_PID" 2>/dev/null || true
    fi

    SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

check_port_free() {
    python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    print(f"Port {port} is already occupied.")
    sys.exit(1)
finally:
    s.close()
PY
}

wait_for_server() {
    local server_log="$1"

    log "Waiting for server health endpoint..."

    for _ in $(seq 1 900); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Server exited during startup."
            tail -n 100 "$server_log" || true
            return 1
        fi

        if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
            log "Server is healthy."
            return 0
        fi

        sleep 1
    done

    log "Server startup timed out."
    tail -n 100 "$server_log" || true
    return 1
}

check_config() {
    local server_log="$1"
    local expected="$2"

    for _ in $(seq 1 30); do
        if grep -q "n_max=${expected}" "$server_log" &&
           grep -q "block_size=" "$server_log"; then
            break
        fi
        sleep 1
    done

    local config_line block_line
    config_line="$(grep -m1 'n_max=' "$server_log" || true)"
    block_line="$(grep -m1 'block_size=' "$server_log" || true)"

    [[ -n "$config_line" ]] ||
        die "Could not find effective n_max in server log."

    [[ -n "$block_line" ]] ||
        die "Could not find block_size in server log."

    local actual_max block_size sample_from_anchor

    actual_max="$(sed -nE 's/.*n_max=([0-9]+).*/\1/p' <<< "$config_line")"
    block_size="$(sed -nE 's/.*block_size=([0-9]+).*/\1/p' <<< "$block_line")"
    sample_from_anchor="$(sed -nE 's/.*sample_from_anchor=(true|false).*/\1/p' <<< "$block_line")"

    log "Effective speculative configuration:"
    log "$config_line"
    log "$block_line"

    [[ "$actual_max" == "$expected" ]] ||
        die "Expected n_max=$expected, got ${actual_max:-unknown}"

    [[ "$block_size" =~ ^[0-9]+$ ]] ||
        die "Could not parse block_size."

    [[ "$sample_from_anchor" == "true" ]] ||
        die "Unexpected sample_from_anchor=$sample_from_anchor. Review draft limit before benchmarking."

    (( expected <= block_size )) ||
        die "Requested n_max=$expected exceeds block_size=$block_size."

    if grep -q 'requested draft size.*clamping' "$server_log"; then
        die "Server reported draft-size clamping. Refusing to benchmark."
    fi

    log "Configuration verified: n_max=$actual_max, block_size=$block_size, sample_from_anchor=true"
}
# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------

cd "$REPO"

[[ -x "$SERVER" ]] || die "GPU Top-K server binary not found: $SERVER"
[[ -f "$MODEL" ]] || die "Target model not found: $MODEL"
[[ -f "$DRAFT" ]] || die "Draft model not found: $DRAFT"
[[ -f "$BENCHMARK" ]] || die "Benchmark script not found: $BENCHMARK"

command -v curl >/dev/null || die "curl is required."
command -v python3 >/dev/null || die "python3 is required."

check_port_free || die "Refusing to interfere with an existing server."

log "Checking runtime library resolution..."

LD_LIBRARY_PATH="$RUNTIME:${LD_LIBRARY_PATH:-}" ldd "$SERVER" \
    > "$RESULT_DIR/ldd.log" 2>&1

if grep -q 'not found' "$RESULT_DIR/ldd.log"; then
    cat "$RESULT_DIR/ldd.log"
    die "Unresolved runtime libraries."
fi

for lib in libllama.so libggml.so; do
    resolved="$(awk -v lib="$lib" '$1 == lib && $2 == "=>" {print $3; exit}' "$RESULT_DIR/ldd.log")"

    if [[ -n "$resolved" && "$resolved" != "$RUNTIME/"* ]]; then
        die "$lib resolves outside the selected runtime bundle: $resolved"
    fi
done

log "Runtime checks passed."
log "Results directory: $RESULT_DIR"
log "Draft-count order: ${VARIANTS[*]}"

# ------------------------------------------------------------
# Run each variant
# ------------------------------------------------------------

for N in "${VARIANTS[@]}"; do
    VARIANT="n${N}"
    VARIANT_DIR="$RESULT_DIR/$VARIANT"

    mkdir -p "$VARIANT_DIR"

    SERVER_LOG="$VARIANT_DIR/server.log"
    BENCH_LOG="$VARIANT_DIR/benchmark.log"

    log "============================================================"
    log "Starting variant: $VARIANT"
    log "============================================================"

    check_port_free || die "Port $PORT became occupied before $VARIANT."

    (
        export LD_LIBRARY_PATH="$RUNTIME:${LD_LIBRARY_PATH:-}"

        exec "$SERVER" \
            --model "$MODEL" \
            --alias unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
            --host "$HOST" \
            --port "$PORT" \
            --temp 1.0 \
            --top-p 1.0 \
            --top-k 0 \
            --min-p 0.0 \
            --chat-template-kwargs '{"reasoning_effort":75}' \
            --load-mode mmap \
            --lazy-mode on \
            --override-tensor "per_layer_token_embd.weight=CPU" \
            -md "$DRAFT" \
            --spec-type draft-dspark \
            --spec-draft-n-max "$N" \
            -ngld 99 \
            -ngl 999 \
            -c 1048576 \
            --parallel 4 \
            -n 32768 \
            --jinja
    ) > "$SERVER_LOG" 2>&1 &

    SERVER_PID=$!

    wait_for_server "$SERVER_LOG" ||
        die "Server failed to become healthy for $VARIANT."

    check_config "$SERVER_LOG" "$N"

    log "Running 4-user benchmark for $VARIANT..."

    set +e
    python3 "$BENCHMARK" \
        --url "http://${HOST}:${PORT}" \
        --levels "4" \
        --warmup 0 \
        --skip-long \
        2>&1 | tee "$BENCH_LOG"

    BENCH_STATUS=${PIPESTATUS[0]}
    set -e

    log "Benchmark exit status: $BENCH_STATUS"

    # Preserve status and continue only when the benchmark succeeded.
    if [[ "$BENCH_STATUS" -ne 0 ]]; then
        cleanup_server
        die "Benchmark failed for $VARIANT. See $BENCH_LOG"
    fi

    cleanup_server

    log "Completed $VARIANT."
    log "Server log: $SERVER_LOG"
    log "Benchmark log: $BENCH_LOG"

    # Give CUDA/runtime resources a moment to settle.
    sleep 5
done

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

log "============================================================"
log "All draft-count experiments completed."
log "============================================================"

for N in "${VARIANTS[@]}"; do
    VARIANT_DIR="$RESULT_DIR/n${N}"

    echo
    echo "===== n_max=$N ====="

    grep -Ei \
        'SUMMARY TABLE|Test[[:space:]]+Category|TotTPS|AvgTPS|errors|error' \
        "$VARIANT_DIR/benchmark.log" | tail -n 40 || true

    echo
    echo "Draft acceptance:"
    grep -E 'draft acceptance =' "$VARIANT_DIR/server.log" |
        python3 -c '
import re, sys
accepted = generated = 0
lengths = []
tasks = 0
for line in sys.stdin:
    m = re.search(r"draft acceptance\s*=\s*[\d.]+\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated\s*\),\s*mean len\s*=\s*([\d.]+)", line)
    if m:
        a, g, length = m.groups()
        accepted += int(a)
        generated += int(g)
        lengths.append(float(length))
        tasks += 1
if tasks:
    print(f"  Tasks: {tasks}")
    print(f"  Weighted acceptance: {accepted / generated:.2%}" if generated else "  Weighted acceptance: N/A")
    print(f"  Mean accepted length: {sum(lengths) / len(lengths):.2f}")
else:
    print("  No acceptance records found.")
'
done

log "Full results saved under: $RESULT_DIR"
