import streamlit as st
from eda import show_eda
from modeling import show_modeling
from prediction import predict 

st.set_page_config(page_title="Health Dataset – EDA & Modeling", layout="wide")

page = st.sidebar.radio("Sec", ["EDA", "Models", "Prediction"])

if page == "EDA":
    show_eda()
elif page == "Models":
    show_modeling()
elif page == "Prediction": 
    predict()