#!/bin/bash
set -e

echo "Starting Reproducibility Smoke Test..."
python -m pytest tests/test_reproduction_values.py -v

if [ "$1" == "--full" ]; then
    echo "Running full reproduction (Placeholder for lengthy scripts if necessary)..."
    # Execute full sweeps if requested, currently just re-running tests
    python -m pytest tests/test_reproduction_values.py -v
fi

echo "Reproduction verified successfully."
