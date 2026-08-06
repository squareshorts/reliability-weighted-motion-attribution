# Reliability-Weighted Attribution of Object Motion under Reduced Otolith Reliability: reproducibility code and outputs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

This repository contains the canonical reproducibility code and outputs for the manuscript "Reliability-Weighted Attribution of Object Motion under Reduced Otolith Reliability".

## Reproduction Instructions

To reproduce the analysis, use the provided script:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1
```

Or on Unix-like systems:

```bash
./reproduce.sh
```

A full run will re-verify the following canonical metrics against the archived results:
- angular RMSE change: 4.37%;
- object RMSE change: 48.77%;
- false-positive RMS change: 47.61%;
- reliability-sweep slope: 0.04135;
- 112 robust and 12 attenuated eye-signal settings;
- worst eye-signal condition: 39/50 positive seeds;
- global sensitivity: 992/1000 positive means, 979/1000 robust positive, one robust reversal, zero failures;
- PRCC: baseline otolith noise approximately +0.932, canal noise approximately -0.843;
- main and supplementary figures.

## License

Software is provided under the MIT License. Data and numerical outputs are provided under CC-BY-4.0.
