use anyhow::Result;
use std::env;
use tch::{
    nn::VarStore,
    vision::{imagenet, resnet::resnet18},
    Device, Kind,
};

fn main() -> Result<()> {
    // Parse args: image_file [weight_file]
    let args: Vec<String> = env::args().collect();
    let image_file = args.get(1).map(|s| s.as_str()).unwrap_or("dog.jpg");
    let weight_file = args.get(2).map(|s| s.as_str()).unwrap_or("resnet18.ot");

    // Create the model and attempt to load the provided weights
    // Allow forcing CPU mode by setting the environment variable FORCE_CPU=1
    let device = if std::env::var("FORCE_CPU").is_ok() {
        println!("FORCE_CPU set — using CPU");
        Device::Cpu
    } else {
        let d = Device::cuda_if_available();
        println!("Using device: {:?}", d);
        d
    };
    let mut vs = VarStore::new(device);
    let model = resnet18(&vs.root(), 1000);

    // Try to load the weights file into the VarStore (state dict compatible with tch)
    let mut loaded_in_varstore = false;
    match vs.load(weight_file) {
        Ok(_) => {
            println!("Loaded weights into VarStore from '{}'", weight_file);
            loaded_in_varstore = true;
        }
        Err(e) => {
            eprintln!("VarStore load failed for '{}': {}", weight_file, e);
            eprintln!("Will try loading as a TorchScript module (CModule) as a fallback.");
        }
    }

    // Load and preprocess the image (resize to 224x224 and normalize as imagenet expects)
    let image = imagenet::load_image_and_resize224(image_file)?.to_device(vs.device());
    let input = image.unsqueeze(0);

    if loaded_in_varstore {
        // Forward pass using the defined model and loaded VarStore
        let output = input.apply_t(&model, false).softmax(-1, Kind::Float);
        for (probability, class) in imagenet::top(&output, 5).iter() {
            println!("{:50} {:5.2}%", class, 100.0 * probability);
        }
    } else {
        // Try to load as a TorchScript module and run it directly
        match tch::CModule::load(weight_file) {
            Ok(module) => {
                println!(
                    "Loaded TorchScript module from '{}', running inference with it.",
                    weight_file
                );
                use tch::IValue;
                let output_ival = module.forward_is(&[IValue::Tensor(input)])?;
                let output = match output_ival {
                    IValue::Tensor(t) => t.softmax(-1, Kind::Float),
                    _ => {
                        return Err(
                            anyhow::anyhow!("TorchScript module did not return a tensor").into(),
                        )
                    }
                };
                for (probability, class) in imagenet::top(&output, 5).iter() {
                    println!("{:50} {:5.2}%", class, 100.0 * probability);
                }
            }
            Err(e) => {
                eprintln!(
                    "Failed to load '{}' as TorchScript module: {}",
                    weight_file, e
                );
                eprintln!("If your weights are a PyTorch state dict, consider exporting a TorchScript model via Python:");
                eprintln!("  model.eval(); torch.jit.trace(model, torch.randn(1,3,224,224)).save('resnet18_scripted.pt')");
                eprintln!("Or provide a safetensors file and we can add a custom loader (nontrivial). Exiting.");
                return Err(e.into());
            }
        }
    }

    Ok(())
}
