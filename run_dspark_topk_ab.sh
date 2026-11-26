#!/usr/bin/env bash
set -Eeuo pipefail

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

URL="http://localhost:8080"
HOST="127.0.0.1"
PORT="8080"

MODEL="/mnt/nfs/llm/models/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf"

DRAFT="/mnt/nfs/llm/models/huggingface/hub/models--unsloth--DeepSeek-V4.1-Flash-1017-GGUF/snapshots/dfcee0ec8d6b874c6663eca0fc55d90ab32534c8/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf"

STARTUP_TIMEOUT=900
STOP_TIMEOUT=60

# Optional override:
# AB_ROOT="$HOME/dspark-topk-ab.xxxxxx" ./run_dspark_topk_ab.sh
if [[ -z "${AB_ROOT:-}" ]]; then
    AB_ROOT="$(
        find "$HOME" -maxdepth 1 -type d -name 'dspark-topk-ab.*' \
            -printf '%T@ %p\n' |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
    )"
fi

if [[ -z "$AB_ROOT" ]]; then
    echo "ERROR: Could not find dspark-topk-ab.* directory."
    exit 1
fi

GPU_DIR="$AB_ROOT/gpu-topk"
CPU_DIR="$AB_ROOT/cpu-topk"

STAMP="$(date +%Y%m%d-%H%M%S)"
RESULT_DIR="$AB_ROOT/results/$STAMP"

mkdir -p "$RESULT_DIR"

SERVER_PID=""

# ------------------------------------------------------------
# Runtime validation
# ------------------------------------------------------------

validate_bundle() {
    local variant="$1"
    local directory="$2"
    local binary="$directory/llama-server"

    echo
    echo "Checking $variant bundle: $directory"

    if [[ ! -x "$binary" ]]; then
        echo "ERROR: Missing executable: $binary"
        exit 1
    fi

    local deps
    deps="$(LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$binary")"

    if grep -q 'not found' <<< "$deps"; then
        echo "$deps"
        echo "ERROR: Missing runtime libraries for $variant."
        exit 1
    fi

    while IFS= read -r dependency; do
        [[ -z "$dependency" ]] && continue

        local resolved
        resolved="$(readlink -f "$dependency")"

        case "$resolved" in
            "$(readlink -f "$directory")"/*)
                ;;
            *)
                echo "ERROR: $variant is resolving a GGML/llama library outside its bundle:"
                echo "  $resolved"
                exit 1
                ;;
        esac
    done < <(
        awk '
            $1 ~ /^lib(ggml|llama)/ && $2 == "=>" && $3 ~ /^\// {
                print $3
            }
        ' <<< "$deps"
    )

    echo "Bundle executable and GGML/llama libraries verified."
}

# ------------------------------------------------------------
# Server lifecycle
# ------------------------------------------------------------

check_port_free() {
    if ss -ltnH 2>/dev/null | awk -v port=":$PORT" '$4 ~ port "$" {found=1} END {exit !found}'; then
        echo "ERROR: Port $PORT is already occupied."
        echo "Stop the existing server before running this test."
        exit 1
    fi
}

start_server() {
    local variant="$1"
    local directory="$2"
    local log="$RESULT_DIR/${variant}-server.log"

    local binary="$directory/llama-server"

    check_port_free

    echo
    echo "============================================================"
    echo " Starting $variant Top-K server"
    echo "============================================================"
    echo "Binary: $binary"
    echo "Log:    $log"

    (
        export LD_LIBRARY_PATH="$directory${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        exec "$binary" \
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
            --spec-draft-n-max 3 \
            -ngld 99 \
            -ngl 999 \
            -c 1048576 \
            --parallel 4 \
            -n 32768 \
            --jinja
    ) > "$log" 2>&1 &

    SERVER_PID=$!

    echo "Server PID: $SERVER_PID"
    echo "Waiting for readiness..."

    local ready=0

    for ((i=1; i<=STARTUP_TIMEOUT; i++)); do
        if curl -fsS --max-time 2 "$URL/health" >/dev/null 2>&1; then
            ready=1
            break
        fi

        local state
        state="$(ps -p "$SERVER_PID" -o stat= 2>/dev/null || true)"

        if [[ -z "$state" || "$state" == Z* ]]; then
            echo "ERROR: Server exited before becoming ready."
            tail -n 80 "$log"
            stop_server
            exit 1
        fi

        if (( i % 30 == 0 )); then
            echo "Still loading... (${i}s)"
        fi

        sleep 1
    done

    if (( ready == 0 )); then
        echo "ERROR: Server startup timed out."
        tail -n 80 "$log"
        stop_server
        exit 1
    fi

    echo "Server is ready."
}

stop_server() {
    if [[ -z "$SERVER_PID" ]]; then
        return
    fi

    echo
    echo "Stopping server PID $SERVER_PID..."

    kill -TERM "$SERVER_PID" 2>/dev/null || true

    for ((i=1; i<=STOP_TIMEOUT; i++)); do
        local state
        state="$(ps -p "$SERVER_PID" -o stat= 2>/dev/null || true)"

        if [[ -z "$state" || "$state" == Z* ]]; then
            break
        fi

        sleep 1
    done

    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Server did not exit gracefully. Sending SIGKILL."
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi

    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""

    echo "Server stopped."
}

# Ensure the server is stopped if the script exits or receives Ctrl+C.
cleanup() {
    local rc=$?
    trap - EXIT
    stop_server
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------
# Benchmark execution
# ------------------------------------------------------------

run_benchmark() {
    local variant="$1"
    local output="$RESULT_DIR/${variant}-benchmark.log"

    echo
    echo "============================================================"
    echo " Benchmark: $variant Top-K"
    echo "============================================================"
    echo "Command:"
    echo "python multi_navid_bench.py --url $URL --levels 4 --warmup 0 --skip-long"
    echo
    echo "Output: $output"
    echo

    set +e
    python multi_navid_bench.py \
        --url "$URL" \
        --levels "4" \
        --warmup 0 \
        --skip-long 2>&1 | tee "$output"
    local rc=${PIPESTATUS[0]}
    set -e

    if (( rc != 0 )); then
        echo "ERROR: $variant benchmark exited with status $rc."
        return "$rc"
    fi

    echo
    echo "$variant benchmark completed."
}

# ------------------------------------------------------------
# Run A/B
# ------------------------------------------------------------

echo "============================================================"
echo " DSpark GPU vs CPU Top-K benchmark"
echo "============================================================"
echo "Bundle root: $AB_ROOT"
echo "Results:     $RESULT_DIR"
echo "Model:       $MODEL"
echo "Draft:       $DRAFT"
echo "Users:       4"
echo

validate_bundle "GPU" "$GPU_DIR"
validate_bundle "CPU" "$CPU_DIR"

# GPU Top-K first
start_server "gpu" "$GPU_DIR"
run_benchmark "gpu"
stop_server

echo
echo "Allowing GPU to settle before the next run..."
sleep 10

# CPU Top-K second
start_server "cpu" "$CPU_DIR"
run_benchmark "cpu"
stop_server

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo " A/B TEST COMPLETED"
echo "============================================================"
echo "GPU benchmark: $RESULT_DIR/gpu-benchmark.log"
echo "CPU benchmark: $RESULT_DIR/cpu-benchmark.log"
echo "GPU server:    $RESULT_DIR/gpu-server.log"
echo "CPU server:    $RESULT_DIR/cpu-server.log"
echo
echo "Results directory:"
echo "$RESULT_DIR"
echo
echo "Compare the OVERALL RESULTS sections in both benchmark logs."
