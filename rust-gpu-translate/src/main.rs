//! CLI for `rust-gpu-translate` with subcommands and language listing.
//!
//! Subcommands:
//!  - `translate` : translate text (supports `--text` or `--file`), defaults English -> German
//!  - `languages` : print a full table of supported languages and ISO codes

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use rust_bert::pipelines::translation::Language;
use rust_gpu_translate::{language_table, read_file, translate_lines};

#[derive(Parser)]
#[command(
    name = "rust-gpu-translate",
    version = "1.0",
    author = "Rashid Rasul",
    about = "Translate text using rust-bert (uses GPU if available)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate text (single sentence or file with one sentence per line)
    Translate {
        /// Text to translate (a single sentence) (short: -T)
        #[arg(short = 'T', long)]
        text: Option<String>,

        /// Path to a file with one sentence per line
        #[arg(short = 'f', long)]
        file: Option<String>,

        /// Source language (name or code). Default: English. Shortcuts: EN, DE, FR, ES, AR
        #[arg(short = 's', long, default_value = "English")]
        source: String,

        /// Target language (name or code). Default: German. Shortcuts: EN, DE, FR, ES, AR
        #[arg(short = 't', long, default_value = "German")]
        target: String,

        /// Disable GPU usage even if CUDA is available
        #[arg(long)]
        no_gpu: bool,
    },

    /// Print a full table of available languages
    Languages {},
}

/// Parse language names and shortcuts into `Language`.
fn parse_language(s: &str) -> Option<Language> {
    match s.to_lowercase().as_str() {
        "english" | "en" | "eng" => Some(Language::English),
        "german" | "de" | "ger" | "deu" => Some(Language::German),
        "french" | "fr" | "fra" => Some(Language::French),
        "spanish" | "es" | "spa" => Some(Language::Spanish),
        "arabic" | "ar" | "ara" => Some(Language::Arabic),
        _ => None,
    }
}

fn print_languages() {
    let table = language_table();
    println!("{:<30} {:<6}", "Language", "ISO");
    println!("{:-<37}", "");
    for (name, iso) in table {
        println!("{:<30} {:<6}", name, iso.unwrap_or("N/A"));
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Translate {
            text,
            file,
            source,
            target,
            no_gpu,
        } => {
            let source_lang = parse_language(&source)
                .ok_or_else(|| anyhow!("Unknown source language: {}", source))?;
            let target_lang = parse_language(&target)
                .ok_or_else(|| anyhow!("Unknown target language: {}", target))?;
            let use_gpu = !no_gpu;

            if let Some(t) = text {
                let outputs = translate_lines(&[t], source_lang, target_lang, use_gpu)?;
                println!("Translation: {}", outputs.get(0).unwrap_or(&String::new()));
            } else if let Some(path) = file {
                let contents = read_file(path)?;
                let lines: Vec<String> = contents.lines().map(|s| s.to_string()).collect();
                let outputs = translate_lines(&lines, source_lang, target_lang, use_gpu)?;
                for s in outputs {
                    println!("{}", s);
                }
            } else {
                println!("No input provided. Use --text or --file. See --help for details.");
            }
        }
        Commands::Languages {} => print_languages(),
    }

    Ok(())
}
