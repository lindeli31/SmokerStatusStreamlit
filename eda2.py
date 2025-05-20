import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

@st.cache_resource(show_spinner=False)
def to_pandas_cached(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas()

def show_eda():
    from data_loader import load_data
    file_path = st.sidebar.text_input("CSV path (EDA)", "train_dataset.csv")

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Errore nel caricamento {file_path}: {e}")
        st.stop()

    df_pd = to_pandas_cached(df)

    PALETTE_SMOKING = ["#648FFF", "#FE6100"]

    st.title("Exploratory Data Analysis")

    with st.expander("👁 Preview & info"):
        st.dataframe(df.head(10))
        st.write("**dtypes:**")
        st.write(df.dtypes)

    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
    num_pd = df_pd[numeric_cols].dropna()

    # Correlation ellipse plot
    if len(num_pd.columns) > 1:
        corr = num_pd.corr()
        st.subheader("Correlation matrix (ellipse style)")
        thr = st.slider("|r| threshold", 0.0, 1.0, 0.7, 0.05)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, len(numeric_cols))
        ax.set_ylim(0, len(numeric_cols))
        ax.set_xticks(np.arange(len(numeric_cols)) + 0.5)
        ax.set_xticklabels(numeric_cols, rotation=90)
        ax.set_yticks(np.arange(len(numeric_cols)) + 0.5)
        ax.set_yticklabels(numeric_cols[::-1])
        ax.set_aspect("equal")

        for i, row in enumerate(numeric_cols[::-1]):
            for j, col in enumerate(numeric_cols):
                if(j>i):
                    continue
                r = corr.loc[row, col]
                if np.isnan(r):
                    continue
                width = height = 0.9
                height *= (1 - abs(r))
                angle = 45 if r > 0 else -45
                colour = plt.cm.coolwarm((r + 1) / 2)
                ax.add_patch(Ellipse((j + 0.5, i + 0.5), width, height, angle=angle, color=colour))
                if abs(r) >= thr:
                    ax.text(j + 0.5, i + 0.5, f"{r:.2f}", ha="center", va="center", fontsize=7)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)

        
    # Scatter Height vs Hemoglobin (Altair)
    if {"height", "height.cm."}.intersection(df.columns) and "hemoglobin" in df.columns:
        xcol = "height" if "height" in df.columns else "height.cm."
        chart = (
            alt.Chart(df_pd)
            .mark_point(filled=True, size=60)
            .encode(
                x=f"{xcol}:Q",
                y="hemoglobin:Q",
                color=alt.Color("smoking:N", scale=alt.Scale(domain=["0", "1"], range=["#648FFF", "#FE6100"]), legend=alt.Legend(title="Smoking")),
                tooltip=[xcol, "hemoglobin", "smoking"],
            )
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)


    if "smoking" in df.columns:
    # Menu di selezione variabili
        all_vars = [c for c in num_pd.columns if c != "smoking"]
        selected_vars = st.multiselect(
        "Seleziona variabili",
        options=["Tutti"] + all_vars,
        default="Tutti"
        )

    # Gestione selezione
    if "Tutti" in selected_vars:
        filtered_vars = all_vars
    else:
        filtered_vars = selected_vars

    # Creazione boxplot dinamici
    def create_boxplot(data, var):
        return (
            alt.Chart(data)
            .mark_boxplot(extent="min-max", size=40)
            .encode(
                x=alt.X("smoking:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
                color=alt.Color("smoking:N", 
                    scale=alt.Scale(domain=["0", "1"], range=PALETTE_SMOKING),
                    legend=None if var != filtered_vars[0] else alt.Legend(title="Smoking")
                )
            )
            .properties(
                title=var,
                width=250,
                height=200
            )
        )

    long_df = df.unpivot(
        index="smoking",
        on=filtered_vars,
        variable_name="variable",
        value_name="value"
    ).to_pandas()

    # Creazione griglia dinamica
    if len(filtered_vars) > 0:
        if "Tutti" in selected_vars:
            # Layout a griglia 3 colonne
            charts = []
            for i in range(0, len(filtered_vars), 3):
                row = alt.hconcat()
                for var in filtered_vars[i:i+3]:
                    row |= create_boxplot(long_df[long_df["variable"] == var], var)
                charts.append(row)
            final_chart = alt.vconcat(*charts)
        else:
            # Layout verticale
            charts = [create_boxplot(long_df[long_df["variable"] == var], var) for var in filtered_vars]
            final_chart = alt.vconcat(*charts)

        st.altair_chart(final_chart, use_container_width=False)
