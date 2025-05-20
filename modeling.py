import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import altair as alt  # Solo Altair

def show_modeling():
    st.title("Modellazione – Classificazione fumatore vs non-fumatore")

    from data_loader import load_data
    file_path = st.sidebar.text_input("CSV path (Modeling)", "train_dataset.csv")

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Errore nel caricamento {file_path}: {e}")
        st.stop()

    df_pd = df.to_pandas()
    if "smoking" not in df.columns:
        st.error("La variabile di destinazione 'smoking' non è presente nel dataset.")
        st.stop()

    model_df = df_pd.dropna(subset=["smoking"])
    X = model_df.drop(columns=["smoking"])
    y = model_df["smoking"].astype("category").cat.codes

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()

    algo = st.selectbox("Algoritmo", ["Logistic Regression"])
    test_size = st.slider("Test size", 0.1, 0.5, 0.25, 0.05)

    if algo == "Logistic Regression":
        C = st.number_input("Inverse regularization strength (C)", 0.01, 10.0, 1.0, 0.1)
    else:
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        max_depth = st.slider("max_depth", 2, 20, 10, 1)

    if st.button("Train model"):
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ]
        )
        if algo == "Logistic Regression":
            classifier = LogisticRegression(max_iter=500, C=C)
        else:
            classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        pipe = Pipeline([("prep", preprocessor), ("clf", classifier)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        st.success(f"Accuracy: {acc:.3f} | ROC-AUC: {roc_auc:.3f}")

        with st.expander("Classification report"):
            st.text(classification_report(y_test, y_pred))

        # Generazione dati per ROC con Altair
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        
        # Linea della classificazione casuale
        random_line = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
        
        # Creazione del grafico ROC con Altair
        roc_chart = (
            alt.Chart(roc_data)
            .mark_line(strokeWidth=3)
            .encode(
                x=alt.X("FPR:Q", title="False Positive Rate"),
                y=alt.Y("TPR:Q", title="True Positive Rate"),
                tooltip=["FPR", "TPR"]
            )
            .properties(title=f"ROC Curve (AUC = {roc_auc:.2f})")
        )
        
        # Aggiunta linea casuale
        random_chart = (
            alt.Chart(random_line)
            .mark_line(color="red", strokeDash=[5, 5])
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q")
            )
        )
        
        # Combinazione dei grafici
        final_chart = (roc_chart + random_chart).configure_axis(
            grid=False
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(final_chart, use_container_width=True)
        