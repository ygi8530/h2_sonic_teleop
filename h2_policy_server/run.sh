#!/usr/bin/env bash
# Lifecycle helper for the H2 EDU policy server.
#
#   ./run.sh build     build the image
#   ./run.sh up        start it detached (host network, 127.0.0.1:5555)
#   ./run.sh down      stop and remove it
#   ./run.sh logs      follow its log
#   ./run.sh verify    re-derive the joint tables from wbc_h12 (after an upstream swap)
#   ./run.sh smoke     end-to-end request check against a running server
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -f "$HERE/docker/docker-compose.yaml")
WBC_ROOT="$(cd "$HERE/../wbc_h12" && pwd)"

require_wbc() {
    local model_dir="$WBC_ROOT/h2_tools/models"
    if [ ! -f "$model_dir/observation_config.yaml" ]; then
        echo "[error] $model_dir is missing. Expected the wbc_h12 checkout at $WBC_ROOT." >&2
        exit 1
    fi
    # LFS pointers are ~130 byte text files; the real decoder is ~41 MB.
    local onnx
    onnx="$(find "$model_dir" -name '*_decoder.onnx' | head -1)"
    if [ -z "$onnx" ] || [ "$(stat -c%s "$onnx")" -lt 1000000 ]; then
        echo "[error] $onnx looks like a Git LFS pointer, not a model." >&2
        echo "        Run: (cd $WBC_ROOT && git lfs pull)" >&2
        exit 1
    fi
}

case "${1:-}" in
    build)  require_wbc; "${COMPOSE[@]}" build ;;
    up)     require_wbc
            "${COMPOSE[@]}" up -d
            echo "[ok] policy server on ${H2_POLICY_BIND:-tcp://0.0.0.0:5555} (reachable as tcp://127.0.0.1:5555)"
            "${COMPOSE[@]}" logs --tail 20 ;;
    down)   "${COMPOSE[@]}" down ;;
    logs)   "${COMPOSE[@]}" logs -f ;;
    verify) docker run --rm \
                -v "$WBC_ROOT:/opt/wbc_h12:ro" \
                -v "$HERE/h2_policy_if:/srv/h2_policy_if:ro" \
                -v "$HERE/tools:/srv/tools:ro" \
                --entrypoint python h2-policy-server:latest \
                tools/verify_constants.py --wbc-root /opt/wbc_h12 ;;
    smoke)  docker run --rm --network host \
                -v "$HERE/h2_policy_if:/srv/h2_policy_if:ro" \
                -v "$HERE/tools:/srv/tools:ro" \
                --entrypoint python h2-policy-server:latest \
                tools/smoke_test.py "${@:2}" ;;
    *)      sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
