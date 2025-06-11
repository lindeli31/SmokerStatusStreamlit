from typing import List
import seaborn as sns #to plot the confusion matrix 
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------- Helper functions ----------------------------- #

def sensitivity_specificity(cm: np.ndarray):
    """Return sensitivity (recall for positive class 1) and specificity."""
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return sens, spec
  


def show_confusion_matrix(cm, classes):
    """Visualizza una matrice di confusione compatta con matplotlib/seaborn."""
    plt.figure(figsize=(2, 2))  # Dimensione ridotta
    ax = sns.heatmap(cm, 
                     annot=True, 
                     fmt='d', 
                     cmap='Blues', 
                     square=True,
                     xticklabels=classes, 
                     yticklabels=classes,
                     cbar=False,
                     annot_kws={'size': 8})  # Testo annotazione più piccolo
    
    # Impostazioni assi e label
    ax.set_xlabel('Predicted', fontsize=6)
    ax.set_ylabel('Actual', fontsize=6)
    ax.tick_params(axis='both', labelsize=6)  # Riduce dimensione label degli assi
    
    # Ottimizzazione layout
    plt.tight_layout(pad=0.5)
    
    # Visualizzazione in Streamlit
    st.pyplot(plt.gcf(), use_container_width=False)
    plt.close()  # Chiude la figura per evitare sovrapposizioni


def plot_roc(y_true, y_prob):
    roc_auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_data = pd.DataFrame({"FPR": fpr, "TPR": tpr})
    random_line = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    roc_chart = (
        alt.Chart(roc_data)
        .mark_line(strokeWidth=3)
        .encode(x="FPR:Q", y="TPR:Q", tooltip=["FPR", "TPR"])
        .properties(title=f"ROC Curve (AUC = {roc_auc:.2f})")
    )
    rand_chart = (
        alt.Chart(random_line)
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )
    st.altair_chart(roc_chart + rand_chart, use_container_width=True)


# --------------------------- Streamlit UI -------------------------------- #

def show_modeling():
    st.title("Models for classification smoking status")
    
    st.write("""
    In this section three models can be choose to train to classify the smoking status, 
    in this situation our main concern is to not misclassify smokers, so sensitivity is
    shown as an important metric to compare the models.  Predictors can be choose for the 
    logistic regression, and the classification tree though classification tree 
    are themselves predictor selectors. In the training of LDA and QDA,
    all variable that violet inherently the assumption of normality, which is necessary 
    for these  models, were excluded. Cross validation is used to train the classification tree
    to choose the cost complexity parameter. More models can be added in the future. 
    """)

    from data_loader import load_data  # lazy import per compatibilità

    file_path = st.sidebar.text_input("CSV path (Modeling)", "train_dataset.csv")
    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Errore nel caricamento {file_path}: {e}")
        st.stop()

    df_pd = df.to_pandas()

    # ----------- Target binario: 1 = fumatore (positivo) ----------------- #
    y_bin = df_pd["smoking"].astype(str).eq("1").astype(int)
    label_names = ["0", "1"]  # non‑smoker, smoker

    # -------------------- Variabili predittive --------------------------- #
    available_vars: List[str] = [c for c in df_pd.columns if c != "smoking"]
    selected_vars = st.multiselect(
        "Predictors for logistic regression/classification Tree", available_vars, default=available_vars
    )

    exclude_cols = [
        "hearing(right)",
        "hearing(left)",
        "eyesight(left)",
        "eyesight(right)",
        "Urine protein",
        "dental caries",
    ]
    discriminant_cols = [c for c in df_pd.columns if c not in exclude_cols and c != "smoking"]

    algo = st.selectbox(
        "Model",
        [
            "Logistic Regression",
            "Linear Discriminant Analysis (LDA)",
            "Quadratic Discriminant Analysis (QDA)",
            "Decision Tree",
        ],
    )

    # Hyper‑parametri solo per Albero
    if algo == "Decision Tree":
        max_depth = st.slider("max_depth", 2, 20, 6, 1)

    if st.button("Train model"):
        if algo == "Logistic Regression":
            st.write("""
Automatic selection can lead to the exclusion from the model 
of explanatory variables that are essential for interpretation.

Moreover, the entry and exit tests for variables are conducted by comparing 
the minimum or maximum value of statistics with the quantiles of nominal 
reference distributions (normal, t, F, ...) that are not appropriate, 
as they are valid, if at all, only for a single analysis.

As a result, inference in the final model is highly inaccurate. 
The observed significance levels for individual coefficients, 
calculated ignoring the selection process, will, for example, 
be smaller than they should be and, consequently, 
the confidence intervals narrower than they should be.
""")
            X_full = df_pd[selected_vars]
            data_logit = pd.concat([y_bin.rename("smoking"), X_full], axis=1).dropna()
            y_logit = data_logit["smoking"]
            X_logit = data_logit[selected_vars]

            X_enc = pd.get_dummies(X_logit, drop_first=True).astype(float)
            X_enc = sm.add_constant(X_enc)

            try:
                res = sm.Logit(y_logit, X_enc).fit(disp=0)
            except Exception as e:
                st.error(f"Errore nell'adattamento del modello: {e}")
                st.stop()

            st.subheader("Model summary")
            st.dataframe(res.summary2().tables[1].round(4))

            y_prob = res.predict(X_enc)
            y_pred = (y_prob >= 0.5).astype(int)

            cm = confusion_matrix(y_logit, y_pred, labels=[0, 1])
            sens, spec = sensitivity_specificity(cm)

            st.write(f"**Sensitivity (classe 1):** {sens:.3f} | **Specificity:** {spec:.3f}")
            show_confusion_matrix(cm, label_names)
            plot_roc(y_logit, y_prob)

        elif algo in [
            "Linear Discriminant Analysis (LDA)",
            "Quadratic Discriminant Analysis (QDA)",
        ]:
            st.write("""
            QDA has a lower sensitivity than LDA probably because QDA penalizes groups with higher variances, 
            so as result we have good classification of non smokers.  We emphasize that in this case it could be limiting to look at the discriminant
function since its estimation depends on the quality of the data. An option would
be to adjust the threshold and use the posterior probabilities estimated by the model for classification.        
                     
            """)
            X_disc = df_pd[discriminant_cols].dropna()
            y_disc = y_bin.loc[X_disc.index]

            cat_features = X_disc.select_dtypes(include=["object", "category"]).columns
            preprocessor = ColumnTransformer(
                transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
                remainder="passthrough",
            )

            clf = LinearDiscriminantAnalysis() if algo.startswith("Linear") else QuadraticDiscriminantAnalysis(store_covariance=True)
            pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
            pipe.fit(X_disc, y_disc)

            y_pred = pipe.predict(X_disc)
            y_prob = pipe.predict_proba(X_disc)[:, 1]

            cm = confusion_matrix(y_disc, y_pred, labels=[0, 1])
            sens, spec = sensitivity_specificity(cm)

            st.write(f"**Sensitività (classe 1):** {sens:.3f} | **Specificità:** {spec:.3f}")
            show_confusion_matrix(cm, label_names)
            plot_roc(y_disc, y_prob)

        else:  # Decision Tree
            st.write("""
The decision tree is grown on the full data set with a user-defined maximum depth to keep the structure interpretable.
Its cost-complexity pruning parameter is selected via 5-fold cross-validation, producing a pruned tree that balances accuracy and simplicity.
In the multiselect predictors can bee chose although classification tree work themselves as predictors selectors""")
    # --- Predittori e target (niente split train/test) ------------------ #
            X_tree_full = df_pd[selected_vars].dropna()
            y_tree_full = y_bin.loc[X_tree_full.index]

    # One-hot dei categorici
            cat_features = X_tree_full.select_dtypes(include=["object", "category"]).columns
            preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)],
            remainder="passthrough",
        )

    # Albero con profondità massima scelta a priori per interpretabilità
            base_tree = DecisionTreeClassifier(
                max_depth=max_depth,      # slider nello sidebar
                random_state=42
            )
            pipe = Pipeline([("prep", preprocessor), ("clf", base_tree)])

        # GridSearchCV su cost-complexity pruning (post-potatura)
                
            alphas = np.linspace(0.0, 0.02, 21)
            grid = GridSearchCV(
            pipe,
            param_grid={"clf__ccp_alpha": alphas},
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            )
            grid.fit(X_tree_full, y_tree_full)
            best_pipe = grid.best_estimator_

    # Valutazione sullo stesso set (coerente con altri modelli)
            y_pred = best_pipe.predict(X_tree_full)
            y_prob = best_pipe.predict_proba(X_tree_full)[:, 1]

            cm = confusion_matrix(y_tree_full, y_pred, labels=[0, 1])
            sens, spec = sensitivity_specificity(cm)

            st.write(f"**Sensitività (classe 1):** {sens:.3f} | **Specificità:** {spec:.3f}")
            show_confusion_matrix(cm, label_names)
            plot_roc(y_tree_full, y_prob)

            # Tree visualizazion
            clf_best: DecisionTreeClassifier = best_pipe.named_steps["clf"]
            feature_names = best_pipe.named_steps["prep"].get_feature_names_out()
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_tree(
                clf_best,
                feature_names=feature_names,
                class_names=["non-smoker", "smoker"],
                max_depth=3,
                filled=True,
                impurity=False,
                ax=ax,
            )
            st.pyplot(fig)
    st.write("""
    In conclusion classification trees seems to have better perfomances for our purpose.          
    """)


