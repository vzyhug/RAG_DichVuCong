[CmdletBinding()]
param(
    [string]$Config = (Join-Path $PSScriptRoot "..\training\configs\dpo_qwen_lora.yaml"),
    [string]$PreferenceDataset = (Join-Path $PSScriptRoot "..\data\training\exports\preference_dataset.jsonl"),
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found. Install the GPU training environment first."
}
& python scripts\validate_dpo_inputs.py --preference $PreferenceDataset
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$HasCuda = & python -c "import torch; print(torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0 -or $HasCuda.Trim() -ne "True") {
    throw "DPO requires a CUDA-enabled PyTorch environment; refusing a CPU training attempt."
}
if (-not (Get-Command llamafactory-cli -ErrorAction SilentlyContinue)) {
    throw "llamafactory-cli was not found. Install LLaMA-Factory in the GPU environment first."
}

& llamafactory-cli train $Config @ExtraArgs
exit $LASTEXITCODE
