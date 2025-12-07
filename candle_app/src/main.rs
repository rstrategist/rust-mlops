use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    // 1. Device Detection
    //    Attempts to create a CUDA device (device 0)
    //    If CUDA is unavailable (driver not installed, no GPU, etc.), falls back to CPU
    let device = match Device::new_cuda(0) {
        Ok(cuda_device) => {
            println!("Using CUDA device");
            cuda_device
        }
        Err(_) => {
            println!("CUDA not available, using CPU");
            Device::Cpu
        }
    };

    println!("Device: {:?}", device);

    // 2. Tensor Creation
    //    Creates two 3x3 random tensors with:
    //    - Mean: 0.0
    //    - Standard deviation: 1.0
    //    - Normal distribution
    let a = Tensor::randn(0f32, 1.0, (3, 3), &device)?;
    let b = Tensor::randn(0f32, 1.0, (3, 3), &device)?;

    // 3. Matrix Multiplication
    //    Performs: c = a × b
    //    This operation is accelerated on GPU when using CUDA
    let c = a.matmul(&b)?;

    // 4. Output
    //    Prints the resulting 3x3 tensor
    println!("Result:\n{}", c);
    Ok(())
}
