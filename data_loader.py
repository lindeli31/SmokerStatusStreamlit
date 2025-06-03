import polars as pl
import streamlit as st

@st.cache_resource(show_spinner=True)
def load_data(path: str) -> pl.DataFrame:
    df = pl.read_csv(path)
    df = df.rename({c: c.strip() for c in df.columns})


    if "dental caries" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("dental caries") == 1)
            .then(1)
            .when(pl.col("dental caries") == 0)
            .then(-1)
            .otherwise(None)
            .alias("dental caries")
        )

    cat_cols = [c for c in ["hearing(left)", "hearing(right)", "dental caries", "smoking"] if c in df.columns]
    for c in cat_cols:
        df = df.with_columns(pl.col(c).cast(pl.Utf8).cast(pl.Categorical))

    if "smoking" in df.columns:
        order = [c for c in df.columns if c != "smoking"] + ["smoking"]
        df = df.select(order)

    return df
