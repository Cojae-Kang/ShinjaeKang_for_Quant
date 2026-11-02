# Real Options Valuation under Stochastic Carbon Prices (LSMC)

Code for the quantitative core of an M.S. thesis (KAIST, 2025–2026; journal manuscript in preparation):
the value of the option to time an investment in a biochar carbon-removal project when carbon-credit
prices follow different stochastic processes. Authors: Shinjae Kang, Jihyo Kim.

## Methods

| Step | What | Script |
|---|---|---|
| Price processes | GBM, generalized Ornstein–Uhlenbeck (AR(1) on log price), Markov-switching diffusion (MS-DR) | `code/lsmc_main_simulation.py` |
| Estimation | GBM and AR(1) fits; MS-DR by maximum likelihood with a hand-coded Hamilton filter (L-BFGS-B, three starting points); model comparison by log-likelihood, AIC and BIC | `code/Data_processfit__fit_diagnostics_and_synthetic_test.py` |
| Valuation | Four-factor least-squares Monte Carlo (Longstaff–Schwartz) for an annual Bermudan deferral option; 14-term polynomial basis; separate training and evaluation paths (N = 100,000 each); 27 scenarios | `code/lsmc_main_simulation.py` |
| Misspecification test | All three models re-estimated on 200 synthetic 47-month samples generated from the calibrated GBM and G-OU, to check whether cross-model divergence comes from the process families or from fitting them to data they do not describe | `code/Data_processfit__fit_diagnostics_and_synthetic_test.py` |
| Convergence | Path counts 10,000 / 25,000 / 50,000 / 100,000, five seeds per level | `code/lsmc_convergence_test.py` |

## Results included

```
results/lsmc_27_scenarios_results.csv          ENPV, option premium and exercise timing, 27 scenarios
results/lsmc_convergence_raw.csv / _summary.csv Path-count convergence
results/msci_biochar_price_descriptive_stats.csv Descriptive statistics and calibrated parameters
results/process_fit/                            Fit statistics and synthetic re-estimation results
```

## Running

Python >= 3.11 with numpy, pandas, scipy and scikit-learn. Seed 42 throughout.

```
python code/lsmc_main_simulation.py      # writes results/lsmc_27_scenarios_0705.csv and results/tau/
python code/lsmc_convergence_test.py     # writes results/lsmc_convergence_*_0705.csv
```

All model parameters are hard-coded, so the valuation scripts run without the licensed price series.

## Data policy

The weekly biochar carbon-credit price series is licensed from MSCI and is not redistributed. The real-sample
part of `Data_processfit__...py` expects it at `data_licensed/` and cannot run without it; its outputs and the
calibrated parameters are provided in `results/`.

## License

MIT — see `LICENSE`.
