# rust-mlops
Rust for MLOPs, examples and learnings from my studies and experience with ML tools.
- Workflows that go beyond Jupyter, Conda, Pandas, Numpy, Sklearn stack for mlops.
-  Docker + pip + virtualenv
-  Microservices

```
rustc --version
```

![image](https://github.com/user-attachments/assets/da663ab9-907d-4141-93d9-8f6dce1984af)
Image sourced from: https://github.com/noahgift/rust-mlops-template
# Run with GitHub Actions
GitHub Actions uses a Makefile to simplify automation
To run everything locally do: make all.
```
name: Rust CI/CD Pipeline
on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]
env:
  CARGO_TERM_COLOR: always
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v1
    - uses: actions-rs/toolchain@v1
      with:
          toolchain: stable
          profile: minimal
          components: clippy, rustfmt
          override: true
    - name: update linux
      run: sudo apt update 
    - name: update Rust
      run: make install
    - name: Check Rust versions
      run: make rust-version
    - name: Format
      run: make format
    - name: Lint
      run: make lint
    - name: Test
      run: make test
```

# MLOps project ideas
- CLI query Hugging Face dataset
- CLI summarise News
- Microservice Web Framework - actix
- Microservice Web Framework deploys pre-trained model
-CLI with descriptive statistics on a well known dataset using https://www.pola.rs/[Polars]
- Train a model with PyTorch (via Rust bindings)
- Explore use-cases in Financial Analysis, trading and DeFi


## 🚀 Example: Rust + CUDA/cuBLAS Integration

This project demonstrates how to call NVIDIA GPU libraries (CUDA Runtime + cuBLAS)
directly from Rust for high-performance linear algebra.

- **What it does:** Implements matrix multiplication (`SGEMM`) on the GPU.
- **Why it matters:** Shows Rust’s potential in HPC/ML workloads, combining safety with raw performance.
- **Skills highlighted:**
  - FFI bindings (`cuda-runtime-sys`, `cublas-sys`)
  - GPU memory management (malloc, free, memcpy)
  - Handling column-major storage (as cuBLAS expects)
  - Wrapping C error codes safely with Rust’s `Result` and `anyhow`

### Run the Example

Make sure CUDA and cuDNN are installed, and DLLs are available in your `PATH`.

```bash
cargo run -p cublas_matmul
Expected output:

A (row-major original): [ [1 2 3], [4 5 6] ]
B (row-major original): [ [7 8], [9 10], [11 12] ]
C (row-major computed): [58.0, 64.0, 139.0, 154.0]
```

This is a foundational building block for more advanced Rust + GPU ML workflows (deep learning, tensor ops, serverless deployment, etc.).