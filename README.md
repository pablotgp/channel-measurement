# When channels measure differently

[![DOI](https://zenodo.org/badge/1433058780.svg)](https://doi.org/10.5281/zenodo.23639437)

Code, data and results for the paper

> **When channels measure differently: counterfactual evaluation of channel switching via change of measure**
> Pablo Antonio García Pastor

The paper estimates what would happen if a group of units (customers, patients, calls) were served by another channel, without a randomized experiment. Identification is cast as a change of measure. Its focus is covariates that each channel records differently: a variable recorded by only one channel, or measured with a different instrument in each one.

All empirical material is either **synthetic** (the simulation) or **public** (NHANES 2017–2018; the NSW/CPS-1 data of LaLonde, 1986). No proprietary data are used.

## Contents

| Folder | What it holds |
| --- | --- |
| `paper/en/` | Official LaTeX source (English), compiled with XeLaTeX. |
| `paper/es/` | Spanish translation with the same content. |
| `paper/arxiv/` | `build_arxiv.py`, which builds a single pdfLaTeX-compatible `main.tex` for arXiv. |
| `simulacion/` | Simulation (Sections 9–10): generator, truths, estimators, registered predictions and raw results. |
| `ilustracion/` | Public-data illustrations (Section 11): NHANES and NSW/CPS, with data, code, registered predictions and results. |

File and variable names are in Spanish.

## Registered predictions

Predictions that do not follow directly from the theory were written to `prediccion*.json` / `predicciones.json` and **committed before the corresponding run**. The commit timestamps in this repository's history are the record. This is a time-stamped self-registration, not a third-party registry. Failed predictions are reported in the paper.

## Simulation (`simulacion/`)

- `gen.py`: data generator (customers, calls, channel, channel-recorded history, outcomes).
- `truth.py`: true θ and theoretical limits, computed on a large population with Gauss–Hermite quadrature.
- `scenarios.py`: scenarios. Run as a script, it writes the predictions to `predicciones.json`.
- `est.py`: Hájek and doubly robust estimators with customer-level cross-fitting (LightGBM).
- `run.py`: runs the replications of one scenario.
- `analyze.py`: compares the results with the predictions.
- `corte_conocido.py`: known-cutoff weights (Proposition 2).
- `bootstrap.py`: unit-level bootstrap, with replacement and Bayesian.
- `ejecutar_todo.py`: resumable runner for a local machine.
- `res_*.json` and `resultados_pc/`: raw results.

```bash
pip install -r requirements.txt
cd simulacion
python scenarios.py               # predictions
N_JOBS=-1 python run.py base 200 1000
python analyze.py
```

## Illustrations (`ilustracion/`)

- **NHANES** (self-reported vs measured BMI):
  - `nhanes_comun.py` (data and channel assignment rule);
  - `nhanes_prediccion.py` (registered predictions);
  - `nhanes_ilustracion.py` (base run);
  - `nhanes_robustez.py` (robustness runs);
  - `analisis_nhanes.md` (results against the predictions).
  
  Data: CDC/NCHS public-use files `DEMO_J`, `BMX_J`, `WHQ_J` and `BPX_J` in `datos/nhanes/`.
- **NSW/CPS-1**: `lalonde.py`. Data: the Dehejia–Wahba NSW subsample and the CPS-1 comparison group in `datos/`, as distributed in the `causaldata` package.

## License

Code: MIT (see `LICENSE`). The NHANES files are public-domain U.S. government data. The NSW/CPS-1 extracts are redistributed as in the `causaldata` package.
