# Insurance Cost Prediction

An end-to-end machine learning project for predicting individual medical
insurance charges from demographic, health, and location-related features.
The workflow is implemented in a Jupyter notebook and is designed to be
reproducible and leakage-safe.

## Project files

```text
.
├── insurance.csv               # Insurance dataset
├── insurance_regression.ipynb  # Data analysis, training, and evaluation
├── requirements.txt            # Python dependencies
├── .gitignore                  # Ignored local and generated files
└── README.md                   # Project documentation
```

## Dataset

The dataset is [`insurance.csv`](./insurance.csv). It contains the following
columns:

| Column | Description |
| --- | --- |
| `age` | Beneficiary age |
| `sex` | Beneficiary sex |
| `bmi` | Body mass index |
| `children` | Number of children covered |
| `smoker` | Whether the beneficiary smokes |
| `region` | US residential region |
| `charges` | Medical insurance cost to predict |

## Project goals

- Explore the insurance dataset and identify useful patterns.
- Build domain-informed features and preprocessing steps.
- Compare several regression algorithms.
- Select and tune a model using cross-validation.
- Evaluate the final model on an untouched test set.
- Inspect errors and feature importance.

## Model results

The recorded final model is a tuned Ridge Regression pipeline:

- **Test R²:** 0.908
- **Test RMSE:** 4,118.08
- **Test MAE:** 2,267.50
- **Data split:** 855 training rows, 214 validation rows, and 268 test rows
- **Best parameters:** `alpha=0.1`, `solver=lsqr`

These metrics are intended for learning and portfolio use, not for making
financial, medical, or insurance underwriting decisions.

## Getting started

### 1. Create an environment

Python 3.11 was used for the recorded model results. Create and activate a
virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install all project dependencies from
[`requirements.txt`](./requirements.txt):

```bash
python -m pip install -r requirements.txt
```

The requirements file includes JupyterLab for running the notebook, NumPy and
Pandas for data processing, SciPy and scikit-learn for modelling, Matplotlib
and Seaborn for visualisation, Joblib and Cloudpickle for model persistence,
and MLflow for experiment tracking and model signatures.

### 2. Run the notebook

Start JupyterLab from the project root:

```bash
jupyter lab
```

Open `insurance_regression.ipynb` and run the cells from top to bottom. The
notebook uses a fixed random seed (`42`), reserves 20% of the data for final
testing, and performs five-fold cross-validation.

## Reproducibility notes

- Run the notebook from the project root so `insurance.csv` resolves correctly.
- Do not use the test set for model selection or hyperparameter tuning.
- Generated experiment output, notebook checkpoints, and serialized model
  binaries are excluded by `.gitignore`.
- Serialized scikit-learn models should only be loaded from trusted sources.

## Limitations

The dataset contains a limited set of predictors. The model does not account
for policy details, medical history, geographic cost differences, or changes in
healthcare pricing. Performance on new populations or future pricing
conditions may differ from the recorded test metrics.
