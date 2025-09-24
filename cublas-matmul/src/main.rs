//! # Rust + CUDA/cuBLAS Matrix Multiplication Example
//!
//! This example demonstrates how to call NVIDIA's CUDA Runtime API
//! and cuBLAS (CUDA Basic Linear Algebra Subprograms) library from Rust
//! via FFI bindings (`cuda-runtime-sys` and `cublas-sys`).
//!
//! The program:
//! - Allocates memory on the GPU for matrices A, B, and C.
//! - Transfers data from host (CPU) to device (GPU).
//! - Performs a matrix multiplication (SGEMM: single-precision general matrix multiply)
//!   using `cublasSgemm_v2`.
//! - Copies the result back to host memory.
//! - Prints the result in row-major order for verification.
//!
//! This project highlights:
//! - How to integrate Rust with NVIDIA GPU libraries via FFI.
//! - Correct use of column-major storage (as cuBLAS expects).
//! - Safe error handling wrappers around CUDA and cuBLAS calls.
//!
//! Expected output for this small 2×3 * 3×2 multiplication is:
//! ```text
//! C (row-major computed): [58.0, 64.0, 139.0, 154.0]
//! ```
//!
//! ## Skills Demonstrated
//! - Low-level GPU programming from Rust
//! - FFI integration with CUDA/cuBLAS
//! - Memory management and data layout handling (row-major ↔ column-major)
//! - Error propagation using `anyhow`
//!
//! This is a building block for larger Rust ML / MLOps workflows.

use anyhow::{Context, Result};
use cublas_sys as cublas;
use cuda_runtime_sys as cuda;
use std::ffi::c_void;
use std::ptr;

/// Convenience wrapper to check CUDA runtime API return codes.
fn check_cuda(status: cuda::cudaError_t) -> Result<()> {
    // Many bindgen-ed enums differ in naming; check numeric success (0).
    if (status as i32) != 0 {
        Err(anyhow::anyhow!("CUDA error: {:?}", status))
    } else {
        Ok(())
    }
}

/// Convenience wrapper to check cuBLAS return codes.
fn check_cublas(status: cublas::cublasStatus_t) -> Result<()> {
    if (status as i32) != 0 {
        Err(anyhow::anyhow!("cuBLAS error: {:?}", status))
    } else {
        Ok(())
    }
}

fn main() -> Result<()> {
    // Matrix dims: (M x K) * (K x N) = (M x N)
    const M: i32 = 2;
    const K: i32 = 3;
    const N: i32 = 2;

    // Host data in **column-major** order to match cuBLAS default expectations.
    // Original A (row-major): [[1,2,3], [4,5,6]]
    let h_a_col: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // length M*K = 6

    // Original B (row-major): [[7,8], [9,10], [11,12]]
    let h_b_col: Vec<f32> = vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0]; // K*N = 6

    // Output buffer for C (column-major, size M*N = 4)
    let mut h_c_col: Vec<f32> = vec![0.0; (M * N) as usize];

    unsafe {
        // 1) Choose device 0 (assumes at least one CUDA-capable GPU).
        check_cuda(cuda::cudaSetDevice(0)).context("cudaSetDevice failed")?;

        // 2) Allocate device memory.
        let bytes_a = (M * K) as usize * std::mem::size_of::<f32>();
        let bytes_b = (K * N) as usize * std::mem::size_of::<f32>();
        let bytes_c = (M * N) as usize * std::mem::size_of::<f32>();

        let mut d_a: *mut c_void = ptr::null_mut();
        let mut d_b: *mut c_void = ptr::null_mut();
        let mut d_c: *mut c_void = ptr::null_mut();

        check_cuda(cuda::cudaMalloc(&mut d_a as *mut *mut c_void, bytes_a))?;
        check_cuda(cuda::cudaMalloc(&mut d_b as *mut *mut c_void, bytes_b))?;
        check_cuda(cuda::cudaMalloc(&mut d_c as *mut *mut c_void, bytes_c))?;

        // 3) Copy host → device.
        check_cuda(cuda::cudaMemcpy(
            d_a,
            h_a_col.as_ptr() as *const c_void,
            bytes_a,
            cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))?;
        check_cuda(cuda::cudaMemcpy(
            d_b,
            h_b_col.as_ptr() as *const c_void,
            bytes_b,
            cuda::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))?;

        // 4) Create cuBLAS handle (context object).
        let mut handle: cublas::cublasHandle_t = std::mem::zeroed();
        check_cublas(cublas::cublasCreate_v2(&mut handle))?;

        // 5) SGEMM: single-precision general matrix multiply.
        // Computes: C = α * A * B + β * C
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Leading dims (column-major).
        let lda: i32 = M; // rows of A
        let ldb: i32 = K; // rows of B
        let ldc: i32 = M; // rows of C

        check_cublas(cublas::cublasSgemm_v2(
            handle,
            cublas::cublasOperation_t::CUBLAS_OP_N, // op(A) = A
            cublas::cublasOperation_t::CUBLAS_OP_N, // op(B) = B
            M,
            N,
            K,
            &alpha as *const f32,
            d_a as *const f32,
            lda,
            d_b as *const f32,
            ldb,
            &beta as *const f32,
            d_c as *mut f32,
            ldc,
        ))?;

        // 6) Copy device → host.
        check_cuda(cuda::cudaMemcpy(
            h_c_col.as_mut_ptr() as *mut c_void,
            d_c,
            bytes_c,
            cuda::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))?;

        // 7) Cleanup.
        check_cublas(cublas::cublasDestroy_v2(handle))?;
        check_cuda(cuda::cudaFree(d_a))?;
        check_cuda(cuda::cudaFree(d_b))?;
        check_cuda(cuda::cudaFree(d_c))?;
    }

    // Convert column-major result back to row-major for pretty printing.
    let mut h_c_row: Vec<f32> = vec![0.0; (M * N) as usize];
    for i in 0..M {
        for j in 0..N {
            let idx_col = (j * M + i) as usize;
            let idx_row = (i * N + j) as usize;
            h_c_row[idx_row] = h_c_col[idx_col];
        }
    }

    println!("A (row-major original): [ [1 2 3], [4 5 6] ]");
    println!("B (row-major original): [ [7 8], [9 10], [11 12] ]");
    println!("C (row-major computed): {:?}", h_c_row);
    // Expected output: [58, 64, 139, 154]

    Ok(())
}
