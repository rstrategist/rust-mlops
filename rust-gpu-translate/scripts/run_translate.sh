#!/usr/bin/env bash
# Simple example runner for rust-gpu-translate (WSL / Linux / macOS)
#
# New: Use a local LibTorch install with `--local-libtorch` or set the env var
# `LOCAL_LIBTORCH` (or `LIBTORCH`) before running. Example:
#   ./run_translate.sh --local-libtorch=/home/phantom/libtorch -T "Hello" -s en -t de
# or set env and run:
#   LOCAL_LIBTORCH=/home/phantom/libtorch ./run_translate.sh -T "Hello" -s en -t de
#
# Usage examples:
#   Translate English -> German (file):
#     ./run_translate.sh --file examples/sample_sentences_en.txt --source en --target de
#   Translate English -> German (single sentence):
#     ./run_translate.sh -T "Hello world" -s en -t de
#   Translate English -> French (single sentence):
#     ./run_translate.sh -T "Hello world" -s en -t fr

set -euo pipefail

# Parse options - extract --local-libtorch[=PATH]
LOCAL_LIBTORCH="${LOCAL_LIBTORCH:-}"
ARGS=()

for arg in "$@"; do
    case "$arg" in
        --local-libtorch=*)
            LOCAL_LIBTORCH="${arg#*=}"
            ;;
        --local-libtorch)
            LOCAL_LIBTORCH="${LOCAL_LIBTORCH:-/home/phantom/libtorch}"
            ;;
        *) 
            ARGS+=("$arg")
            ;;
    esac
done

# Allow setting via env var for convenience
LOCAL_LIBTORCH="${LOCAL_LIBTORCH:-${LIBTORCH:-}}"

if [[ -n "${LOCAL_LIBTORCH}" ]]; then
    echo "Using local libtorch at ${LOCAL_LIBTORCH}"
    export LIBTORCH="${LOCAL_LIBTORCH}"
    export CXX="${CXX:-c++}"
    # Append include flags (preserve any existing CXXFLAGS)
    export CXXFLAGS="${CXXFLAGS:-}"" -I${LIBTORCH}/include -I${LIBTORCH}/include/torch/csrc/api/include"
    # Add linker search path and rpath so runtime finds .so files
    export RUSTFLAGS="${RUSTFLAGS:-}"" -L native=${LIBTORCH}/lib -C link-args=-Wl,-rpath,${LIBTORCH}/lib"
    export LD_LIBRARY_PATH="${LIBTORCH}/lib:${LD_LIBRARY_PATH:-}"
fi

# pass through remaining args to cargo run
cargo run -- translate "${ARGS[@]}"
