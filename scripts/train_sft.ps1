[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot "..\training\configs\sft_qwen_lora.yaml"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command llamafactory-cli -ErrorAction SilentlyContinue)) {
    throw "llamafactory-cli was not found. Install LLaMA-Factory in the GPU environment first."
}

& llamafactory-cli train $Config @ExtraArgs
exit $LASTEXITCODE
