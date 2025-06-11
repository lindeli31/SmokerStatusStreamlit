import polars as pl
import streamlit as st
import altair as alt
from eda_utils import plot_categorical
from eda_utils import scatter_plot
from eda_utils import plot_corrplot

def show_eda():
    from data_loader import load_data
    file_path = st.sidebar.text_input("CSV path (EDA)", "train_dataset.csv")

    PALETTE_SMOKING = ["#648FFF", "#FE6100"]

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Errore nel caricamento {file_path}: {e}")
        st.stop()

    df_pd = df.to_pandas()
    st.title("Introduction and Exploratory Data Analysis")
# Sezione introduttiva
    st.markdown("""
    In clinical and research contexts, reliably determining whether a patient is a smoker is crucial for both:
    - Prevention and management of chronic diseases
    - Ensuring response reliability in health surveys
    
    Smoking status can be a sensitive topic, and patients might provide inaccurate self-reports. 
    Our solution analyzes biological markers to objectively predict smoking status.
    The working dataset contains N = 38984 adult observations and p = 15 biological variables drawn from routine clinical
    examinations. The target variable Smoking is binary (1 = current smoker, 0 = non-smoker). 
    There are no missing data. The sample size of smokers is 14,318, and non-smokers is 24,666.
    """)

    with st.expander("🧪 Predictor Variables", expanded=True):
        st.markdown("""
        We consider these biological measurements:
        - **Demographics**: Age
        - **Body metrics**: Height, Weight, Waist
        - **Sensory**: Eyesight (both eyes), Hearing (both ears)
        - **Cardiovascular**: Systolic BP, Diastolic BP
        - **Metabolic**: Fasting Blood Sugar
        - **Lipid Panel**: Total Cholesterol, HDL, LDL, Triglycerides
        - **Hematology**: Hemoglobin
        - **Renal**: Urine Protein, Serum Creatinine
        - **Hepatic**: AST, ALT, GGT
        - **Oral Health**: Dental Caries
        """)

    with st.expander("📚 Variable Details", expanded=False):
        st.markdown("""
        | Variable | Clinical Significance |
        |----------|----------------------|
        | Waist | Abdominal fat indicator |
        | Eyesight | Visual acuity (10/10 = normal) |
        | Hearing | Binary classification (normal 1/abnormal 2) |
        | Systolic BP | Pressure during heartbeats |
        | Diastolic BP | Pressure between heartbeats |
        | Fasting Blood Sugar | Diabetes risk indicator (mg/dL) |
        | HDL | 'Good' cholesterol (artery protection) |
        | LDL | 'Bad' cholesterol (artery clogging) |
        | Triglycerides | Heart disease risk marker |
        | Hemoglobin | Oxygen-carrying protein (anemia indicator) |
        | Urine Protein | Kidney health marker |
        | Serum Creatinine | Kidney filtration indicator |
        | AST/ALT | Liver inflammation markers |
        | GGT | Alcohol/tobacco effects on liver |
        | Dental Caries | Presence of tooth cavities |
        """)

    with st.expander("👁 Preview & info", expanded=False):
        st.dataframe(df.head(10))
        st.markdown("**Data types:**")
        # Opzione più pulita: mostra nome variabile e tipo
        st.write(df_pd.dtypes.to_frame('Type'))
        
    #dropping al categorical variables 
    numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]  
    num_pd = df_pd[numeric_cols].dropna() 



    ######## CORRPLOT 
    
    st.write("## Correlation Plot")
    if len(num_pd.columns) > 1:
        corr = num_pd.corr().reset_index().melt(id_vars='index')
        corr.columns = ['var1', 'var2', 'correlation']
        st.write("""For a better visualization only coefficients with values 
                 above the specified threshold are shown in the corrplot. """)
        thr = st.slider("**Correlation coefficient threshold**", 0.0, 1.0, 0.7, 0.05)
    
    
        corr['size'] = 1 - abs(corr['correlation'])
    
   
        chart = plot_corrplot(corr, thr)
        st.altair_chart(chart, use_container_width=True)
    
    #### BOXPLOTS
    all_vars = [c for c in num_pd.columns if c != "smoking"]

    st.write("## Boxplots Continuous variables – Smoking status")

    default_vars = ["age", "height(cm)", "hemoglobin", "waist(cm)", "weight(kg)"]
    boxplot_selected_vars = st.multiselect(
        "Select variables",
        options=all_vars,         
        default=default_vars      
    )

    boxplot_filtered_vars = boxplot_selected_vars or default_vars


    def create_boxplot(data, var):
        return (
            alt.Chart(data)
            .mark_boxplot(size=40)
            .encode(
                x=alt.X("smoking:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value:Q", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "smoking:N",
                    scale=alt.Scale(domain=["0", "1"], range=PALETTE_SMOKING),
                    legend=None if var != boxplot_filtered_vars[0] else alt.Legend(title="Smoking")
                )
            )
            .properties(title=var, width=250, height=200)
        )

    if boxplot_filtered_vars:      # evita errori se la lista è vuota
        long_df = (
            df.unpivot(
                index="smoking",
                on=boxplot_filtered_vars,
                variable_name="variable",
                value_name="value",
            )
            .to_pandas()
        )

    # Layout: 3 colonne se più variabili, verticale se poche
    if len(boxplot_filtered_vars) > 3:
        rows = []
        for i in range(0, len(boxplot_filtered_vars), 3):
            row = alt.hconcat(
                *[create_boxplot(long_df[long_df["variable"] == var], var)
                  for var in boxplot_filtered_vars[i:i + 3]]
            )
            rows.append(row)
        final_chart = alt.vconcat(*rows)
    else:
        final_chart = alt.vconcat(
            *[create_boxplot(long_df[long_df["variable"] == var], var)
              for var in boxplot_filtered_vars]
        )

    st.altair_chart(final_chart, use_container_width=False)


    with st.expander("📊 Comments on boxplots", expanded=True):
        st.markdown("""
                    
    **Patterns observed in the dataset:**
    
    - **Younger age distributions**: Smokers are typically younger.
    - **Elevated hemoglobin levels**: Higher median values in smokers. \
    Smoking causes CO to bind to hemoglobin, forming carboxyhemoglobin (HbCO) and reducing oxygen transport.
    - **Larger waist circumference**: Suggests potential weight-related impacts.
    - **Higher body weight**: Consistent with abdominal fat patterns; smoking habits may contribute to weight gain.,
    - **Higher relaxation times**: (example comment) Possible cardiovascular adaptations.
    **Height-Smoking Association:** this finding might be coincidental or could reflect demographic/socioeconomic patterns in the
    dataset rather than a direct biological relationship. It's also possible that the dataset includes specific population subgroups where these characteristics co-occur.
        """)

        st.markdown("""
    **Additional observations:**
    - Numerous upper-tail outliers present across nearly all variables.
    """)
        st.info("Note: These observations are based on aggregate dataset patterns. Individual variations may occur.")

    #Dicrete variable- Smoking status 
                
    st.write("## Barplots Discrete variables-Smoking Status")
    
    categorical_cols = ["Urine protein", "dental caries", "hearing(left)", "hearing(right)"]
    barplot_selected_var = st.selectbox("Select categorical variable", categorical_cols)
    
    # Selezione tipo di marginalizzazione
    margin_type = st.radio("Marginalization direction", 
                          ["Within category levels (margin=1)", 
                           "Within smoking groups (margin=2)"],
                          horizontal=True)
    
    margin = 1 if margin_type.startswith("Within category") else 2
    
    # Mostra tabella delle frequenze
    with st.expander("View frequency table"):
        freq_table = df[barplot_selected_var].value_counts(normalize=True)
        freq_table.columns = [barplot_selected_var, 'Proportion']
        st.dataframe(freq_table)
    
    # Genera e mostra il grafico
    chart = plot_categorical(df_pd, barplot_selected_var, margin=margin)
    st.altair_chart(chart, use_container_width=True)
    
    #########################àà
    with st.expander("📊 Comments on barplots", expanded = True):
        st.markdown("""
    **Key observations from categorical variable analysis:**
    
    ##### Urine Protein (Margin 1)
    - For protein levels 1-4, higher levels show increasing proportions of smokers
    - Level 5 shows an unexpected reversal with higher non-smoker proportion
    - Levels 5-6 show irregular patterns due to low sample size:
    
    """)
    
    # Show protein distribution if available
        if 'Urine protein' in df.columns:
            protein_dist = df["Urine protein"].value_counts()
            protein_dist.columns = ["Level", "Count"]
            st.dataframe(protein_dist)
        st.markdown("""
    ##### Dental Caries (Margin 2)
    - Smokers show approximately 5% higher prevalence of dental caries
    - Suggests potential association between tobacco use and dental health
    - Observational finding - requires causal investigation

    ##### Hearing Capability (Margin2 )
    - No substantial differences in hearing impairment frequencies between smokers and non-smokers
    - Similar patterns observed for both left and right ears
    - Recommendation: Combine hearing variables into a single dichotomous feature:
        - 1 = Hearing impairment present (either ear)
        - 0 = Normal hearing (both ears)
    """)
    
    st.header("Scatterplot Analysis")
    st.markdown("""
    Explore relationships between numerical variables with interactive scatterplots.
    Points are color-coded by smoking status to identify potential patterns.
    This scatterplot analysis provides limited new insights, functioning mainly as
    a visual synthesis of the relationships already identified through the correlation matrix and boxplot comparisons.
    """)
    
    # Get numerical columns (excluding smoking status)
    num_cols = num_pd.columns.tolist()
    if 'smoking' in num_cols:
        num_cols.remove('smoking')
    
    # Create columns for selection
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        x_var = st.selectbox("X-axis Variable", num_cols, index=num_cols.index('height') if 'height' in num_cols else 0)
    with col2:
        y_var = st.selectbox("Y-axis Variable", num_cols, index=num_cols.index('weight') if 'weight' in num_cols else 1)
    with col3:
        color_var = st.selectbox("Color Variable", ['smoking', 'gender'], index=0)
    
    # Options panel
    with st.expander("⚙️ Visualization Options", expanded=True):
        jitter = st.checkbox("Add jitter for discrete values", value=True)
        if jitter:
            jitter_width = st.slider("Jitter Intensity", 0.01, 0.5, 0.2, 0.01)
        seed = st.number_input("Random Seed", min_value=0, value=42, step=1)
    # Generate and display plot
    try:
        fig = scatter_plot(
            data = df_pd,
            x_var=x_var,
            y_var=y_var,
            color_var=color_var,
            jitter=jitter,
            jitter_width=jitter_width if jitter else 0.01,
            seed=seed
        )
        st.altair_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error generating scatterplot: {str(e)}")
    
    with st.expander("📊 Comments on scatterplots", expanded=True):
        st.markdown("""
    
    **Key observations:**
    - **Height-Emoglobin**: These variables also exhibit moderate correlation, 
    potentially because taller individuals may have a larger blood volume
    requiring higher hemoglobin levels to meet oxygen demands
    and smoking itself can increase hemoglobin concentrations 
    (via carbon monoxide exposure, which stimulates red blood cell production).
    - **Waist-Weight**: Smokers exhibit a higher values of waist circumferences and weights, reinforcing
    the “central obesity” signal seen earlier.

    """)