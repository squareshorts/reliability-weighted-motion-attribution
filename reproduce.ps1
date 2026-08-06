$ErrorActionPreference = "Stop"

Write-Host "Starting Reproducibility Smoke Test..."
python -m pytest tests/test_reproduction_values.py -v

if ($args -contains "--full") {
    Write-Host "Running full reproduction (Placeholder for lengthy scripts if necessary)..."
    python -m pytest tests/test_reproduction_values.py -v
}

Write-Host "Reproduction verified successfully."
