# 🏦 Loan Approval Predictor (KNN + Streamlit)

A Streamlit app that predicts whether a loan application will be **Approved** or **Rejected**
using a K-Nearest Neighbors classifier. It reproduces the pipeline from
[`notebooks/KNN.ipynb`](notebooks/KNN.ipynb): drop missing values, one-hot encode the
categorical columns, 67/33 train/test split (`random_state=42`), standardize features, then KNN.

## Features
- **Predict** – enter applicant details and get a prediction with class probabilities
- **Model performance** – accuracy, precision/recall, confusion matrix, and a K-vs-accuracy chart
- **Data** – preview, class distribution and summary statistics
- Adjustable **K** in the sidebar (default 5, as in the notebook)

## Project structure
```
.
├── app.py                # Streamlit UI
├── model_utils.py        # data loading, training, evaluation
├── requirements.txt
├── data/
│   └── loan_data.csv     # <- put your dataset here (see below)
├── notebooks/
│   └── KNN.ipynb         # original notebook
└── .streamlit/config.toml
```

## Dataset
Place your CSV at **`data/loan_data.csv`** (rename `loan_data (1).csv` from the notebook).
Required columns:

`age, annual_income, loan_amount, monthly_emi, credit_score, existing_loans, has_property, city, education, employment_type, loan_status`

`applicant_id` is optional and ignored. `loan_status` must contain `Approved` / `Rejected`.

If the file is missing, the app shows an upload box so it never crashes.

> ⚠️ If your repo is public, the CSV will be public too. For private data, don't commit it —
> use the upload box in the app instead.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Push this folder to a GitHub repository.
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → pick your repo and branch.
4. Set **Main file path** to `app.py` and click **Deploy**.

The model trains in a second or two on startup, so there is no pickle file to keep in sync
with library versions.

## Notes
- The dataset is imbalanced, so KNN is weak at spotting the minority class (low recall).
  Consider class balancing (e.g. SMOTE), other models, or threshold tuning for real use.
- This is a learning/demo project, not a real credit-decision system.
