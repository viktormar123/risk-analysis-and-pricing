# Non-Life Insurance Claim Simulation and Reinsurance Analysis

This repository contains Python code for modeling and analyzing fire insurance claims using Danish historical data. The focus is on comparing different approaches to simulate total claims and evaluate the financial impact of reinsurance contracts.

## Overview

The analysis includes:

* Empirical, theoretical, and hybrid claim modeling techniques.
* Evaluation of model fit and accuracy using statistical metrics.
* Simulation-based analysis of reinsurance contracts with different caps.
* Quantitative and visual comparison of tail behavior, revenue, and risk.

## Main Files

### `compare_claim_models.py`

Simulates insurance claims using various severity modeling strategies:

* **Empirical**: Pure resampling from data
* **Hybrid**: Mixture of empirical and theoretical (e.g. 97/3 split)
* **Theoretical**: Full parametric distribution fit

Also includes:

* Train/test stratified split for evaluation
* Visualization of claim distributions and tails
* KL divergence, Wasserstein distance, KS/AD tests
* Full comparison table of statistical metrics

### `compare_simulations.py`

Compares simulation strategies in the context of reinsurance impact:

* **Empirical model**
* **Composite model** (95% and 99% thresholds)

Also includes:

* Reinsurance contract analysis (with and without limits)
* Visualization of risk and revenue distributions
* CSV outputs of simulation and reinsurance metrics

### `analysis.py`

Utility functions for:

* Loading and cleaning data
* Generating histograms, QQ plots, and CDF comparisons
* Applying reinsurance formulas
* Producing CSVs and figures for reporting

## Data

The analysis is based on a cleaned CSV file: `danishDataFinal.csv`. Claim amounts are in millions of DKK.

## Outputs

Generated plots and tables include:

* Train/test distribution histograms
* Fitted PDF and CDF plots
* Model performance comparisons
* Revenue and risk metrics under different reinsurance structures
* Summary CSVs for LaTeX reports or further analysis

## Requirements

* Python 3.8+
* pandas, numpy, matplotlib, seaborn
* scipy, scikit-learn

Install requirements:

```bash
pip install -r requirements.txt
```

## Usage

To run the main simulation and evaluation scripts:

```bash
python compare_claim_models.py
python compare_simulations.py
python analysis.py
```

Plots and CSVs will be saved to the current directory.

## Notes

* All code is designed to be reproducible with fixed seeds.
* Visualizations are saved in `ClaimDensityPlots/` and the root directory.
* The models focus on aggregate claims per year, not policy-level data.

---

For questions or suggestions, feel free to open an issue or contact the author.
