# PowerShell helper for windows users
# Examples:
#   Translate English -> German (file):
#     .\run_translate.ps1 -File examples\sample_sentences_en.txt -Source en -Target de
#   Translate English -> German (single sentence):
#     .\run_translate.ps1 -Text "Hello world" -Source en -Target de
#   Translate English -> French (single sentence):
#     .\run_translate.ps1 -Text "Hello world" -Source en -Target fr

param(
    [string]$Text = $null,
    [string]$File = $null,
    [string]$Source = "en",
    [string]$Target = "de",
    [switch]$NoGpu
)

$argList = @("translate")
if ($Text) { $argList += "--text"; $argList += $Text }
if ($File) { $argList += "--file"; $argList += $File }
$argList += "--source"; $argList += $Source
$argList += "--target"; $argList += $Target
if ($NoGpu) { $argList += "--no-gpu" }

cargo run -- $argList
