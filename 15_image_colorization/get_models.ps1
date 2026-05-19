# Downloads the Caffe weights (~123 MB). Prototxt and pts_in_hull.npy are in the repo.
$ErrorActionPreference = "Stop"
$modelDir = Join-Path $PSScriptRoot "Model"
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
$outFile = Join-Path $modelDir "colorization_release_v2.caffemodel"
if (Test-Path $outFile) {
    Write-Host "Already exists: $outFile"
    exit 0
}
$url = "https://github.com/spmallick/learnopencv/releases/download/Colorization/colorization_release_v2.caffemodel"
Write-Host "Downloading colorization_release_v2.caffemodel ..."
Invoke-WebRequest -Uri $url -OutFile $outFile
Write-Host "Done."
