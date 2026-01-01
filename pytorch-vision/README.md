# pytorch-vision

Small example that demonstrates how to run a pretrained ResNet18 from Rust using the `tch` crate (LibTorch bindings).

This example shows how to build and run a minimal Rust program that:

- loads an image and preprocesses it for ImageNet (224x224, normalized),
- creates a ResNet18 model, loads provided weights (state dict `.ot`) into a `VarStore`,
- falls back to loading a TorchScript module if the state-dict load fails,
- runs inference and prints the top-5 ImageNet classes.

## Prerequisites

- Rust toolchain (stable)
- A PyTorch / LibTorch install. Two common ways:
  - Use a Python virtualenv with `torch` installed (recommended for WSL): set `LIBTORCH_USE_PYTORCH=1` in the environment so the build script picks it up.
  - Or provide a local LibTorch directory and set `LIBTORCH` to point at it.

Notes:
- Tested with Python venv `torch==2.4.0+cu118` (CUDA 11.8) in WSL.
- If you don't have a GPU or want to force CPU, set `FORCE_CPU=1` before running.

## Build

Example (WSL + Python venv):

```bash
# Activate your venv that has torch installed
export VIRTUAL_ENV=/home/phantom/.venv
export PATH="$VIRTUAL_ENV/bin:$HOME/.cargo/bin:$PATH"
export LIBTORCH_USE_PYTORCH=1
cargo build --release
```

## Run

Example using the included `dog.jpg` and `resnet18.ot` (default weight filename):

```bash
# Make sure the venv libs are found at runtime
export LD_LIBRARY_PATH=/home/phantom/.venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
./target/release/pytorch-vision dog.jpg resnet18.ot
```

Sample output from a run on GPU (your output may differ slightly):

```
Using device: Cuda(0)
Loaded weights into VarStore from 'resnet18.ot'
Bernese mountain dog                               85.03%
Appenzeller                                         8.52%
EntleBucher                                         2.28%
Greater Swiss Mountain dog                          1.93%
Border collie                                       0.61%
```

## Usage

```
pytorch-vision [IMAGE_FILE] [WEIGHT_FILE]

Defaults: IMAGE_FILE=dog.jpg WEIGHT_FILE=resnet18.ot
```

If the weights file is a PyTorch state dict (the usual `.ot` / `.pth`), the example will try to load it into the `VarStore` and run the `resnet18` defined in the code. If the file cannot be loaded that way, the binary attempts to load it as a TorchScript module (saved with `torch.jit.trace`/`torch.jit.script`).

## Troubleshooting

- If you see linker errors, ensure `LD_LIBRARY_PATH` includes the path to the `torch/lib` directory of your Python venv, or set `LIBTORCH` to a local LibTorch install and re-run `cargo build`.
- To bypass the strict version check in the build script (only recommended for advanced users), set `LIBTORCH_BYPASS_VERSION_CHECK=1`.

---

Contributions welcome — this is a small playground to explore `tch` and running PyTorch models from Rust.