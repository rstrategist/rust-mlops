// build.rs

fn main() {
    // Link search paths for .lib files
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.2\\lib\\x64");
    println!(
        "cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA\\CUDNN\\v9.13\\lib\\12.9\\x64"
    );

    // Link against the required libraries
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    // (If there are cuBLAS helper libs or versioned names, adjust accordingly.)
}
