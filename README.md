# SFRC Fatigue-Life Prediction — Core Modeling Reference

This repository provides the original dataset and a **concise reference implementation** of the core modeling workflow described in the associated study on steel fiber-reinforced concrete (SFRC) fatigue-life prediction. The reference code is not the original research script and is **not intended to reproduce the paper's exact numerical results**.

## Repository contents

- `README.md` — Description of the repository, data, and reference workflow.
- `core_modeling.py` — Reference implementation of the main modeling stages.
- `requirements.txt` — Software dependencies for the reference code.
- Original SFRC fatigue dataset (Excel file) — 117 observations collected from the literature.

## Dataset

The publicly shared Excel dataset contains **117 original observations**. Following the data screening and quality-control procedures described in the paper, **93 observations** were retained for the reported modeling analysis.

The original dataset is provided for transparency and further research. It is **not** the curated input table expected by `core_modeling.py`. The raw-data screening, exclusions, and feature preparation needed to obtain the modeling dataset are not implemented in this reference code. Please refer to the paper for the reported data-selection and preprocessing procedures.

## Reference modeling workflow

The reference code illustrates the following stages:

- Four modeling cases: Case A, Case B, Case A+B, and All-in-One.
- Missing-value imputation and feature scaling fitted using training data only.
- XGBoost hyperparameter optimization using Optuna and cross-validation.
- Cross-validation-based selection of boosting rounds.
- Evaluation across five random training–testing splits using basic prediction metrics.

The hyperparameter information reported in the study is provided in **Appendix Table A1** of the paper.

## Scope and limitations

This repository does **not** include:

- The complete original data-screening and right-censoring exclusion scripts or geometric feature derivation routines.
- SHAP analysis and figure-generation scripts.
- Residual diagnostics, trained model exports, or publication-specific reporting scripts.

Accordingly, the reference implementation should **not** be treated as an independently executable reproduction of all results, figures, or statistics reported in the paper.

## Environment

The reference workflow was documented for **Python 3.10.9**. Selected package dependencies are listed in `requirements.txt`. Compatibility should be checked in the user's environment.

## Running the reference code

The script expects a **separately prepared, curated** Excel file named `SFRC_curated_dataset.xlsx` in the same directory. The expected input includes `log_Nf`, the five model features, fiber type, substrate type, and data-quality labels.

Once a compatible curated dataset has been prepared and the required dependencies installed, run:

```bash
python core_modeling.py
```

**Important:** The repository's 117-observation original dataset cannot be used directly as the script input without the necessary data-preparation steps. The curated 93-observation modeling table is not included in this repository.
