# Candle CUDA Demo Project

A simple demonstration of GPU-accelerated tensor operations using Hugging Face's Candle library in Rust. This project showcases automatic CUDA detection with graceful CPU fallback.

## 🚀 Features

- Automatic GPU Detection: Attempts to use CUDA-enabled GPU, falls back to CPU if unavailable
- Simple Tensor Operations: Matrix multiplication with random tensors
- Cross-Platform: Works on Windows, macOS, and Linux
- Minimal Setup: Easy to build and run

## 📋 Prerequisites

### System Requirements

- Rust: 1.70+ (install via rustup)
- CUDA Toolkit: 11.8+ (only if using NVIDIA GPU)
- NVIDIA GPU: Compute capability 6.1+ (for CUDA support)

## CUDA Installation (for GPU support)

### On WSL (Windows Subsystem for Linux):

Follow the official NVIDIA guide: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

```bash
# Install CUDA 11.8 (recommended for compatibility)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-11-8

# Set environment variables in ~/.bashrc
nano ~/.bashrc

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

## Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU information
nvidia-smi

# Check compute capability (should be ≥ 6.1)
nvidia-smi --query-gpu=compute_cap --format=csv
```

## 📦 Dependencies

The project uses the following Rust crates:

- candle-core: Hugging Face's tensor library with CUDA support
  - Features: cuda (enables GPU acceleration)
  - Version: 0.9.1 (stable release tested with CUDA 11.8)

## Explore Candle Examples

```bash
# Clone official examples
git clone https://github.com/huggingface/candle.git
cd candle/candle-examples
```

## Learn More

- Candle Documentation
- Candle GitHub Repository
- Rust CUDA Programming Guide

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgments

- Hugging Face for the Candle library
- Rust Community for excellent tools and documentation
- NVIDIA for CUDA platform
