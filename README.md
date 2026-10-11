# Insurance Cost Prediction

An end-to-end machine learning project for predicting individual medical
insurance charges from demographic, health, and location-related features.
The workflow is implemented in a Jupyter notebook and is designed to be
reproducible, leakage-safe, and suitable for exporting a deployment pipeline.

## Project goals

- Explore the insurance dataset and identify useful patterns.
- Build domain-informed features and preprocessing steps.
- Compare several regression algorithms.
- Select and tune a model using cross-validation.
- Evaluate the final model once on an untouched test set.
- Inspect errors and feature importance.
- Track experiments with MLflow.
- Export the trained pipeline and its metadata.

## Dataset

The main dataset is [`Dataset/insurance.csv`](./Dataset/insurance.csv). It
contains the following columns:

| Column | Description |
| --- | --- |
| `age` | Beneficiary age |
| `sex` | Beneficiary sex |
| `bmi` | Body mass index |
| `children` | Number of children covered |
| `smoker` | Whether the beneficiary smokes |
| `region` | US residential region |
| `charges` | Medical insurance cost to predict |

The repository also contains a Cars24 dataset under `Dataset/` as a separate
modelling artifact. It is not used by the insurance regression notebook.

## Model results

The recorded final model is a tuned Ridge Regression pipeline:

- **Test R²:** 0.908
- **Test RMSE:** 4,118.08
- **Test MAE:** 2,267.50
- **Data split:** 855 training rows, 214 validation rows, and 268 test rows
- **Best parameters:** `alpha=0.1`, `solver=lsqr`

These results are documented in
[`models/insurance_model_card.json`](./models/insurance_model_card.json).
They are intended for learning and portfolio use, not for making financial,
medical, or insurance underwriting decisions.

## Repository structure

```text
.
├── Dataset/
│   ├── insurance.csv
│   └── cars24-car-price-clean2.csv
├── experiments/                 # MLflow tracking data (generated)
├── models/
│   ├── insurance_model_card.json
│   ├── model_metadata.json
│   └── catalog.json
├── insurance_regression.ipynb   # Main analysis and training workflow
├── .gitignore
└── README.md
```

## Getting started

### 1. Create an environment

Python 3.11 is used by the recorded model metadata. Create and activate a
virtual environment, then install the notebook dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyterlab numpy pandas scipy scikit-learn \
  matplotlib seaborn joblib mlflow
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run the notebook

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Open `insurance_regression.ipynb` and run the cells from top to bottom. The
notebook uses a fixed random seed (`42`), reserves 20% of the data for final
testing, performs five-fold cross-validation, and stores MLflow run data under
`experiments/`.

## Reproducibility notes

- Run the notebook from the repository root so relative dataset and output
  paths resolve correctly.
- Do not use the test set for model selection or hyperparameter tuning.
- Generated MLflow runs and serialized model binaries are ignored by Git; keep
  them locally or store them in an external artifact repository when sharing
  experiments.
- Serialized scikit-learn models should only be loaded from trusted sources.

