import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------ #
# 1. Load data – same helper the other pages use
# ------------------------------------------------------------------ #
from data_loader import load_data


@st.cache_resource  # trains the tree once per session / parameter set
def train_tree(df: pd.DataFrame, predictors, max_depth=6):
    """Return a pruned Decision-Tree pipeline and the list of categorical cols."""
    X = df[predictors].dropna()
    y = df["smoking"].astype(str).eq("1").astype(int).loc[X.index]

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    preproc = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    pipe = Pipeline([("prep", preproc), ("clf", base_tree)])

    # cost-complexity pruning grid
    alphas = np.linspace(0.0, 0.02, 21)
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__ccp_alpha": alphas},
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid.fit(X, y)
    return grid.best_estimator_, list(cat_cols)


def predict():
    st.title("Prediction – Decision Tree Classifier")

    # -------------------------------------------------------------- #
    # 2. Data + model training (cached)
    # -------------------------------------------------------------- #
    file_path = st.sidebar.text_input(
        "CSV path (Prediction)", "train_dataset.csv", key="pred_csv"
    )
    try:
        df = load_data(file_path).to_pandas()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        st.stop()

    if "smoking" not in df.columns:
        st.error("Column 'smoking' not found in the dataset!")
        st.stop()

    predictors = [c for c in df.columns if c != "smoking"]
    max_depth = st.sidebar.slider("Max tree depth", 2, 20, 6, 1)

    model, cat_cols = train_tree(df, predictors, max_depth)

    # -------------------------------------------------------------- #
    # 3. User input form
    # -------------------------------------------------------------- #
    st.subheader("Input predictor values")

    with st.form("user_inputs"):
        user_vals = {}
        for col in predictors:
            if col in cat_cols:
                opts = sorted(df[col].dropna().unique())
                user_vals[col] = st.selectbox(col, opts, index=0)
            else:
                mn, mx, md = (
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].median()),
                )
                user_vals[col] = st.number_input(col, value=md, min_value=mn, max_value=mx)
        submitted = st.form_submit_button("Classify")

    # -------------------------------------------------------------- #
    # 4. Prediction
    # -------------------------------------------------------------- #
    if submitted:
        user_df = pd.DataFrame([user_vals])
        prob = model.predict_proba(user_df)[0, 1]
        pred = int(prob >= 0.5)

        st.success(
            f"**Predicted class:** {pred} "
            f"({'smoker' if pred else 'non-smoker'})\n\n"
            f"**Probability of being smoker:** {prob:.3f}"
        )
