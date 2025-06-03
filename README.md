# Smoker-Status Prediction from Biosignals

An interactive Streamlit application that lets users  

1. **Explore** a health dataset (EDA)  
2. **Train & compare** classification models (Models)  
3. **Predict** a person’s smoker status with a Decision-Tree classifier (Prediction)

---

## 1 · Project Goals

* Provide an intuitive dashboard where anyone can:
  * Visualise and interrogate the biosignal data on their own.
  * Re-train several machine-learning models and benchmark their performance.
  * Enter custom feature values and instantly obtain a smoker/non-smoker prediction.

---

## 2 · Dataset

| Item | Details |
|------|---------|
| **Source** | Kaggle – “Smoker Status Prediction Using Biosignals” <https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals> |
| **Files used** | `train_dataset.csv` |
| **Target** | `smoking_status` (binary: smoker = 1, non-smoker = 0) |
| **Main features** | Age, Height, Weight, BMI, Blood Pressure (sys/dia), Heart-rate, Body-Temp, Oxygen-Sat, etc. |

> **No preprocessing required:** the dataset is already clean, well-formatted, and free of missing values or obvious outliers, so we load it directly into the app.

---


If you **just want to try the dashboard** and don’t care about creating a permanent virtual-env:

```bash
# ① Install uv (one-time, 5-second download)
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS/Linux
# on Windows:  iwr -useb https://astral.sh/uv/install.ps1 | iex

# ② Grab the code any way you like (clone / ZIP) and cd inside
git clone https://github.com/<your-username>/smokerstatusprediction.git
cd smokerstatusprediction            # or the folder you un-zipped

# ③ Run the code
uv run streamlit run app.py

