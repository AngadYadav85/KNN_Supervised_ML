"""Loan Approval Predictor - Streamlit app (KNN classifier)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from model_utils import TARGET, load_data, predict_one, sweep_k, train_model

DATA_PATH = Path(__file__).parent / "data" / "loan_data.csv"

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")


# --------------------------------------------------------------------------- #
# Cached helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def read_default_data(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_resource(show_spinner="Training KNN model...")
def get_model(_df: pd.DataFrame, data_key: int, k: int):
    return train_model(_df, k)


@st.cache_data(show_spinner="Comparing values of K...")
def get_k_sweep(_df: pd.DataFrame, data_key: int) -> pd.DataFrame:
    return sweep_k(_df, list(range(1, 26, 2)))


def clamp(value, low, high):
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# Sidebar + data loading
# --------------------------------------------------------------------------- #
st.title("🏦 Loan Approval Predictor")
st.caption("K-Nearest Neighbors classifier trained on historical loan applications.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Number of neighbors (K)", min_value=1, max_value=25, value=5, step=1)
    st.caption("The original notebook uses K = 5.")

df = None
try:
    if DATA_PATH.exists():
        df = read_default_data(str(DATA_PATH))
    else:
        st.warning(
            f"Dataset not found at `data/{DATA_PATH.name}`. "
            "Upload your loan CSV below, or add it to the repo at that path."
        )
        uploaded = st.file_uploader("Upload loan_data.csv", type="csv")
        if uploaded is not None:
            df = load_data(uploaded)
except ValueError as err:
    st.error(str(err))
    st.stop()
except Exception as err:  # unreadable / malformed file
    st.error(f"Could not read the dataset: {err}")
    st.stop()

if df is None:
    st.info("Waiting for a dataset to get started.")
    st.stop()

data_key = int(pd.util.hash_pandas_object(df, index=True).sum() % (2**31))
model = get_model(df, data_key, k)

tab_predict, tab_perf, tab_data = st.tabs(
    ["🔮 Predict", "📊 Model performance", "🗂️ Data"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Applicant details")
    with st.form("applicant_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age", min_value=18, max_value=100,
                value=clamp(int(df["age"].median()), 18, 100), step=1,
            )
            annual_income = st.number_input(
                "Annual income", min_value=0,
                value=int(df["annual_income"].median()), step=10000,
            )
            loan_amount = st.number_input(
                "Loan amount", min_value=0,
                value=int(df["loan_amount"].median()), step=10000,
            )
            monthly_emi = st.number_input(
                "Monthly EMI", min_value=0.0,
                value=float(df["monthly_emi"].median()), step=1000.0,
            )

        with c2:
            cs_low = min(300, int(df["credit_score"].min()))
            cs_high = max(900, int(df["credit_score"].max()))
            credit_score = st.slider(
                "Credit score", min_value=cs_low, max_value=cs_high,
                value=clamp(int(df["credit_score"].median()), cs_low, cs_high),
            )
            existing_loans = st.number_input(
                "Existing loans", min_value=0, max_value=30,
                value=clamp(int(df["existing_loans"].median()), 0, 30), step=1,
            )
            has_property = st.selectbox("Owns property?", ["Yes", "No"])

        with c3:
            city = st.selectbox("City", sorted(df["city"].unique()))
            education = st.selectbox("Education", sorted(df["education"].unique()))
            employment_type = st.selectbox(
                "Employment type", sorted(df["employment_type"].unique())
            )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        result = predict_one(
            model,
            {
                "age": age,
                "annual_income": annual_income,
                "loan_amount": loan_amount,
                "monthly_emi": monthly_emi,
                "credit_score": credit_score,
                "existing_loans": existing_loans,
                "has_property": 1 if has_property == "Yes" else 0,
                "city": city,
                "education": education,
                "employment_type": employment_type,
            },
        )
        label = result["label"]
        if label == "Approved":
            st.success(f"✅ Prediction: **{label}**")
        else:
            st.error(f"❌ Prediction: **{label}**")

        cols = st.columns(len(result["proba"]))
        for col, (name, p) in zip(cols, sorted(result["proba"].items())):
            col.metric(f"P({name})", f"{p:.0%}")
        st.caption(
            f"Based on the {model.k} most similar past applicants. With a small K, "
            "probabilities move in coarse steps (e.g. 0%, 20%, 40% for K = 5). "
            "This is a demo model and not financial advice."
        )

# --------------------------------------------------------------------------- #
# Tab 2: Model performance
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader(f"Hold-out test results (K = {model.k})")
    report = model.report
    minority = min(model.class_names, key=lambda c: report[c]["support"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{model.accuracy:.1%}")
    m2.metric(f"Precision ({minority})", f"{report[minority]['precision']:.1%}")
    m3.metric(f"Recall ({minority})", f"{report[minority]['recall']:.1%}")
    m4.metric("Train / test rows", f"{model.n_train} / {model.n_test}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        im = ax.imshow(model.confusion, cmap="Blues")
        ax.set_xticks(range(len(model.class_names)), model.class_names)
        ax.set_yticks(range(len(model.class_names)), model.class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        threshold = model.confusion.max() / 2
        for i in range(model.confusion.shape[0]):
            for j in range(model.confusion.shape[1]):
                ax.text(
                    j, i, int(model.confusion[i, j]), ha="center", va="center",
                    color="white" if model.confusion[i, j] > threshold else "black",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Classification report**")
        rows = {
            name: report[name]
            for name in model.class_names + ["macro avg", "weighted avg"]
        }
        report_df = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
        report_df = report_df.round(2)
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df)

    counts = df[TARGET].value_counts()
    st.info(
        f"The dataset is imbalanced ({', '.join(f'{n}: {c}' for n, c in counts.items())}). "
        f"When one class is much rarer, KNN tends to favour the majority class, so recall "
        f"for **{minority}** can be low even when overall accuracy looks fine."
    )

    st.markdown("**Effect of K on test performance**")
    sweep = get_k_sweep(df, data_key)
    st.line_chart(sweep)
    st.caption("For exploration only: picking K by test score slightly overfits the test set.")

# --------------------------------------------------------------------------- #
# Tab 3: Data
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Dataset preview")
    st.write(f"{len(df):,} rows after dropping missing values.")
    st.dataframe(df.head(50))

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Loan status distribution**")
        st.bar_chart(df[TARGET].value_counts())
    with d2:
        st.markdown("**Numeric summary**")
        st.dataframe(df.describe().T)
