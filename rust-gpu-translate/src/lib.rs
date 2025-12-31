//! Small helper library that wraps `rust-bert` translation pipelines.
//!
//! This module provides utilities to read text from files and perform translations using
//! `TranslationModelBuilder`. When LibTorch with CUDA is available the model will run on
//! GPU; otherwise it falls back to CPU. Use the CLI (in `main.rs`) for a simple user-facing tool.

use anyhow::Result;
use rust_bert::pipelines::translation::{Language, TranslationModel, TranslationModelBuilder};
use std::fs::File;
use std::io::Read;
use tch::Device;

/// Read an entire file into a single `String`.
/// The function expects UTF-8 encoded files and returns an error on I/O problems.
pub fn read_file(path: String) -> Result<String> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(contents)
}

/// Read a file and return a vector where each element is a single line (no trailing newline).
pub fn read_file_array(path: String) -> Result<Vec<String>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let array = contents.lines().map(|s| s.to_string()).collect();
    Ok(array)
}

/// Session that owns a single translation pipeline (built once) and reuses it for
/// subsequent translations. This avoids rebuilding the model on every call and also
/// centralizes the device detection and diagnostics (printed once at session creation).
pub struct TranslationSession {
    model: TranslationModel,
    target: Language,
}

impl TranslationSession {
    /// Build a new session for the given language pair and device preference.
    pub fn new(source: Language, target: Language, use_gpu: bool) -> Result<Self> {
        let device: Device = if use_gpu {
            Device::cuda_if_available()
        } else {
            Device::Cpu
        };

        // Print available devices and which will be used (only once per session)
        println!("Available devices:");
        println!(" - CPU");
        if tch::Cuda::is_available() {
            let count = tch::Cuda::device_count();
            println!(" - CUDA available (device_count={})", count);
            // Try to get GPU names via nvidia-smi if present
            match std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=name", "--format=csv,noheader"])
                .output()
            {
                Ok(out) if out.status.success() => {
                    let names = String::from_utf8_lossy(&out.stdout);
                    for (i, name) in names.lines().enumerate() {
                        println!("    - [{}] {}", i, name.trim());
                    }
                }
                _ => {
                    for i in 0..tch::Cuda::device_count() {
                        println!("    - CUDA device {}", i);
                    }
                }
            }
        } else {
            println!(" - CUDA not available");
        }
        println!("Selected device: {:?}", device);

        let model = TranslationModelBuilder::new()
            .with_source_languages(vec![source])
            .with_target_languages(vec![target])
            .with_device(device)
            .create_model()?;

        Ok(Self { model, target })
    }

    /// Translate a single sentence.
    pub fn translate<S: AsRef<str>>(&self, sentence: S) -> Result<String> {
        let input = [sentence.as_ref()];
        let out = self.model.translate(&input, None, self.target)?;
        Ok(out.get(0).cloned().unwrap_or_default())
    }

    /// Translate a slice of sentences.
    pub fn translate_lines<S: AsRef<str>>(&self, lines: &[S]) -> Result<Vec<String>> {
        let input_refs: Vec<&str> = lines.iter().map(|s| s.as_ref()).collect();
        let out = self.model.translate(&input_refs, None, self.target)?;
        Ok(out)
    }
}

/// Convenience wrapper that keeps the original API: build a session and translate the lines.
pub fn translate_lines<S: AsRef<str>>(
    lines: &[S],
    source: Language,
    target: Language,
    use_gpu: bool,
) -> Result<Vec<String>> {
    let session = TranslationSession::new(source, target, use_gpu)?;
    session.translate_lines(lines)
}

/// Convenience wrapper: read a file and translate each line from Spanish to English on GPU if available.
pub fn translate_file(path: String) -> Result<()> {
    let text = read_file_array(path)?;
    let outputs = translate_lines(&text, Language::Spanish, Language::English, true)?;
    for s in outputs {
        println!("{}", s);
    }
    Ok(())
}

/// Return a full table of supported languages (Display name and optional ISO 639-1 code).
///
/// The list is constructed from the `Language` enum variants in `rust-bert` so it reflects
/// all languages the translation pipelines are aware of. For languages without a short
/// ISO 639-1 code the code will be `None`.
pub fn language_table() -> Vec<(String, Option<&'static str>)> {
    use rust_bert::pipelines::translation::Language::*;

    let languages = vec![
        Latvian,
        Achinese,
        MesopotamianArabic,
        TaizziAdeniArabic,
        TunisianArabic,
        Afrikaans,
        SouthLevantineArabic,
        Akan,
        Amharic,
        NorthLevantineArabic,
        NajdiArabic,
        MoroccanArabic,
        EgyptianArabic,
        Assamese,
        Asturian,
        Awadhi,
        CentralAymara,
        SouthAzerbaijani,
        NorthAzerbaijani,
        Bashkir,
        Bambara,
        Balinese,
        Belarusian,
        Bemba,
        Bengali,
        Bhojpuri,
        Banjar,
        Tibetan,
        Bosnian,
        Buginese,
        Bulgarian,
        Catalan,
        Cebuano,
        Czech,
        Chokwe,
        CentralKurdish,
        CrimeanTatar,
        Welsh,
        Danish,
        German,
        SouthwesternDinka,
        Dyula,
        Dzongkha,
        Greek,
        English,
        Esperanto,
        Estonian,
        Basque,
        Ewe,
        Faroese,
        Fijian,
        Finnish,
        Fon,
        French,
        Friulian,
        NigerianFulfulde,
        WestCentralOromo,
        ScottishGaelic,
        Irish,
        Galician,
        Guarani,
        Gujarati,
        Haitian,
        Hausa,
        Hebrew,
        Hindi,
        Chhattisgarhi,
        Croatian,
        Hungarian,
        Armenian,
        Igbo,
        Iloko,
        Indonesian,
        Icelandic,
        Italian,
        Javanese,
        Japanese,
        Kabyle,
        Kachin,
        Kamba,
        Kannada,
        Kashmiri,
        Georgian,
        Kazakh,
        Kabiye,
        Kabuverdianu,
        HalhMongolian,
        Khmer,
        Kikuyu,
        Kinyarwanda,
        Kirghiz,
        Kimbundu,
        NorthernKurdish,
        CentralKanuri,
        Kongo,
        Korean,
        Lao,
        Ligurian,
        Limburgan,
        Lingala,
        Lithuanian,
        Lombard,
        Latgalian,
        Luxembourgish,
        LubaLulua,
        Ganda,
        Luo,
        Lushai,
        Magahi,
        Maithili,
        Malayalam,
        Marathi,
        Minangkabau,
        Macedonian,
        Maltese,
        Manipuri,
        Mossi,
        Maori,
        Burmese,
        Dutch,
        Norwegian,
        NorwegianNynorsk,
        NorwegianBokmal,
        Nepali,
        Pedi,
        Nuer,
        Nyanja,
        Occitan,
        Odia,
        Pangasinan,
        Panjabi,
        Papiamento,
        SouthernPashto,
        IranianPersian,
        PlateauMalagasy,
        Polish,
        Portuguese,
        Dari,
        AyacuchoQuechua,
        Romanian,
        Rundi,
        Russian,
        Sango,
        Sanskrit,
        Santali,
        Sicilian,
        Shan,
        Sinhala,
        Slovak,
        Slovenian,
        Samoan,
        Shona,
        Sindhi,
        Somali,
        SouthernSotho,
        Spanish,
        Sardinian,
        Serbian,
        Swati,
        Sundanese,
        Swedish,
        Swahili,
        Silesian,
        Tamil,
        Tamasheq,
        Tatar,
        Telugu,
        Tajik,
        Tagalog,
        Thai,
        Tigrinya,
        TokPisin,
        Tswana,
        Tsonga,
        Turkmen,
        Tumbuka,
        Turkish,
        Twi,
        CentralAtlasTamazight,
        Uighur,
        Ukrainian,
        Umbundu,
        Urdu,
        NorthernUzbek,
        Venetian,
        Vietnamese,
        Waray,
        Wolof,
        Xhosa,
        EasternYiddish,
        Yoruba,
        YueChinese,
        Chinese,
        Zulu,
        WesternFrisian,
        Arabic,
        Mongolian,
        Yiddish,
        Pashto,
        Farsi,
        Fulah,
        Uzbek,
        Malagasy,
        Albanian,
        Breton,
        Malay,
        Oriya,
        NorthernSotho,
        Luganda,
        Azerbaijani,
        ChineseMandarin,
        HaitianCreole,
        CentralKhmer,
    ];

    languages
        .into_iter()
        .map(|l| (format!("{}", l), l.get_iso_639_1_code()))
        .collect()
}
