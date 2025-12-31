#!/usr/bin/env bash
set -ex

# Explicitly point to the libtorch we downloaded
export LIBTORCH=/home/phantom/libtorch
export LIBTORCH_LIB="$LIBTORCH/lib"
export LIBTORCH_INCLUDE="$LIBTORCH"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"

echo "LIBTORCH=$LIBTORCH"
echo "LIBTORCH_INCLUDE=$LIBTORCH_INCLUDE"
echo "LIBTORCH_LIB=$LIBTORCH_LIB"

cargo clean

# Capture verbose build output to a log
LIBTORCH=$LIBTORCH LIBTORCH_LIB=$LIBTORCH_LIB LIBTORCH_INCLUDE=$LIBTORCH_INCLUDE cargo build -vv 2>&1 | tee /tmp/cargo_build_libtorch.log
