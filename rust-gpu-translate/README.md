# rust-gpu-translate 🚀

A small CLI tool that uses rust-bert to translate text. It uses LibTorch (via `tch`) and will run on GPU automatically if LibTorch with CUDA is available; pass `--no-gpu` to force CPU.

---

## Features ✅
- Translate single sentences or files (one sentence per line)
- Uses `rust-bert`'s `TranslationModelBuilder` to pick an appropriate pretrained model
- GPU-aware: uses `tch::Device::cuda_if_available()` by default
- `languages` subcommand prints a full table of supported languages and their ISO codes

---

## Quick start 🧭

Build the project (WSL / Linux / macOS):

```bash
cargo build
```

Example commands:

- Translate a single sentence:

```bash
cargo run -- translate --text "Hello world!"
# short: -T "Hello world!"
```

- Translate a file (one sentence per line):

```bash
cargo run -- translate --file /path/to/sentences.txt
# short: -f /path/to/sentences.txt
```

- Print the full languages table:

```bash
cargo run -- languages
```

### Interactive REPL

The `translate` subcommand now supports an interactive REPL when no file is specified. Behavior:

- If you run `translate --text "..."` the CLI will translate the provided sentence and then enter the interactive REPL.
- If you run `translate --file <PATH>` the tool translates all lines in the file and exits (no REPL).
- If you run `translate` with no `--text` or `--file` it will enter the interactive REPL immediately.
- To exit the REPL, enter an empty line or send EOF (Ctrl+D on Linux/macOS, Ctrl+Z Enter on Windows).

Notes:

- The application builds a single `TranslationSession` (model) once per run and reuses it for all translations in the session. This avoids rebuilding the model for every line and improves interactive performance.

### Translate subcommand options

- `--text <TEXT>` / `-T <TEXT>` : single sentence to translate
- `--file <PATH>` / `-f <PATH>` : file with one sentence per line
- `--source <LANG>` / `-s <LANG>` : source language (default: **English**). Shortcuts: **EN, DE, FR, ES, AR**
- `--target <LANG>` / `-t <LANG>` : target language (default: **German**). Shortcuts: **EN, DE, FR, ES, AR**
- `--no-gpu` : force CPU even if CUDA is available

---

## Examples ✨

- Translate file text English → German (explicitly):

```bash
cargo run -- translate --file examples/sample_sentences_en.txt --source en --target de
```

- Translate inline text (default English → German):

```bash
cargo run -- translate --text "How are you?"
```

- Translate Spanish/English text to French/Urdu (example of different language pairs):

```bash
cargo run -- translate --text "Hola" --source ES --target FR
cargo run -- translate --text "When will you visit again?" --source EN --target UR
```

- Print languages table:

```bash
cargo run -- languages
```

You can also use the provided helper scripts:

- Bash (WSL / Linux / macOS):
  - `chmod +x scripts/run_translate.sh`
  - `./scripts/run_translate.sh --file examples/sample_sentences_en.txt --source en --target de`
  - **Local LibTorch:** Use `--local-libtorch=/path/to/libtorch` or set the `LOCAL_LIBTORCH` env var to point to a local LibTorch install (useful on WSL where you have a CUDA-enabled libtorch). Example:

    ```bash
    # Use a local LibTorch install and run an inline translation
    ./scripts/run_translate.sh --local-libtorch=/home/phantom/libtorch -T "Hello, where shall we go tonight?" -s English -t German

    # Or set env var and run the heavy translate script
    LOCAL_LIBTORCH=/home/phantom/libtorch ./scripts/run_heavy_translate.sh 2000 64
    ```

- PowerShell (Windows):
  - `./scripts/run_translate.ps1 -File examples\\sample_sentences_en.txt -Source en -Target de`

---

## Implementation details 🔍

- The translation pipeline uses `rust-bert`'s `TranslationModelBuilder` and selects a pretrained model that supports the requested language pair.
- The tool sets device using `tch::Device::cuda_if_available()` unless `--no-gpu` is passed (so it runs on GPU when LibTorch + CUDA is present). Device diagnostics are printed once when the translation session is created (not on every translation).
- The CLI creates a `TranslationSession` that builds the model once for the chosen language pair and device; the session is reused for subsequent translations (interactive and file modes) to improve performance and avoid repeated model initialization.
- `language_table()` collects each `Language` variant's display name and optional ISO-639-1 code (via `Language::get_iso_639_1_code()`), and the `languages` subcommand prints a simple table with that information.

### Monitoring GPU / system usage

To monitor GPU usage dynamically while the model is running, use one of these commands:

- `watch -n 1 nvidia-smi` — refreshes the `nvidia-smi` output every second (works on NVIDIA GPUs)
- `nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv -l 1` — prints just GPU/memory utilization in a loop
- `gpustat -i 1` — (Python) a compact display; install with `pip install gpustat`
- `nvtop` — interactive GPU monitor (installable on many distros)

To monitor CPU/memory usage, use:

- `htop` — interactive process viewer
- `top` — built-in system monitor

Example (run in one terminal):

```bash
watch -n 1 nvidia-smi
```

Then run the translation in another terminal to observe GPU usage.

### Heavy work example (saturate GPU a bit)

A helper script is provided to create a larger test workload that can exercise the GPU more:

```bash
chmod +x scripts/run_heavy_translate.sh
# default: 1000 lines
./scripts/run_heavy_translate.sh 2000 64
```

This will create a temporary file with repeated sample sentences and translate it. Increase the number of lines to push GPU/CPU utilization higher; reduce it if you run out of memory.

---

## Notes & troubleshooting ⚠️

- First run may download model artifacts and take some time and network bandwidth.
- If `cargo` is not available in your shell (WSL), install Rust toolchain with `rustup`.
- If you want to force CPU execution for repeatable benchmarks or low-memory environments, use `--no-gpu`.

---

## Contributing

Contributions welcome — open an issue or PR. Keep changes small and include tests where useful.

---

License: MIT (see `LICENSE`)
