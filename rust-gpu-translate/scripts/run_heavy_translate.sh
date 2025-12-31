#!/usr/bin/env bash
# Create a large temporary input (default 1000 lines) and translate to saturate GPU
# Usage: ./run_heavy_translate.sh [NUM_LINES] [BATCH_SIZE]
# Example: ./run_heavy_translate.sh 2000 64

set -euo pipefail
NUM_LINES=${1:-1000}
BATCH_SIZE=${2:-64}
TMPFILE=$(mktemp /tmp/rust_gpu_translate_input.XXXXXX)

for i in $(seq 1 $NUM_LINES); do
  echo "This is a sentence to be translated that exercises the model and GPU (line $i)." >> "$TMPFILE"
done

echo "Wrote $NUM_LINES lines to $TMPFILE"

echo "Translating in batches of $BATCH_SIZE..."
# We ask the model to translate the whole file; the pipeline will process all lines. If memory is tight,
# reduce NUM_LINES or BATCH_SIZE.
cargo run -- translate --file "$TMPFILE" --source en --target de

rm -f "$TMPFILE"

echo "Done."