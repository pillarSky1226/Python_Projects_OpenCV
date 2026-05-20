# Downloads TensorFlow OCR weights for license-plate character recognition.
$ErrorActionPreference = "Stop"
$modelDir = Join-Path $PSScriptRoot "model"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

$files = @{
    "binary_128_0.50_ver3.pb" = "https://raw.githubusercontent.com/hritik7080/Car-License-Plate-Recognition/master/model/binary_128_0.50_ver3.pb"
    "binary_128_0.50_labels_ver2.txt" = "https://raw.githubusercontent.com/hritik7080/Car-License-Plate-Recognition/master/model/binary_128_0.50_labels_ver2.txt"
}

foreach ($name in $files.Keys) {
    $outFile = Join-Path $modelDir $name
    if (Test-Path $outFile) {
        Write-Host "Already exists: $outFile"
        continue
    }
    Write-Host "Downloading $name ..."
    Invoke-WebRequest -Uri $files[$name] -OutFile $outFile
}

Write-Host "Done. Model files are in $modelDir"
