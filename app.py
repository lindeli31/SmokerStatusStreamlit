import streamlit as st
from eda2 import show_eda
from modeling import show_modeling

st.set_page_config(page_title="Health Dataset – EDA & Modeling", layout="wide")

page = st.sidebar.radio("Sezione", ["EDA", "Modellazione"])

if page == "EDA":
    show_eda()
elif page == "Modellazione":
    show_modeling()
