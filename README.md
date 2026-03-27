# Heart Disease Risk Predictor

A machine learning web application that predicts the likelihood of heart disease based on clinical health parameters. Built as a BYOP (Bring Your Own Project) submission for the **Fundamentals of AI/ML** course.

---

##  Problem Statement :

Cardiovascular disease is the leading cause of death in India, responsible for over 28% of all fatalities. Most people are unaware of their risk until symptoms appear — often too late. This project provides a simple, accessible tool for early self-assessment using well-established clinical indicators from the UCI Heart Disease dataset.

---

## What It Does :

- Takes 13 clinical inputs (age, cholesterol, blood pressure, etc.)
- Runs them through a trained ML classification model
- Returns a **risk prediction** (Disease / No Disease) with a **confidence score**
- Categorises risk as Low 🟢 / Moderate 🟡 / High 🔴

---

##  Tech Stack :

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Models | scikit-learn (Logistic Regression, Random Forest, SVM) |
| Evaluation | classification_report, ROC-AUC, cross-validation |
| Visualisation | Matplotlib, Seaborn |
| Web App | Streamlit |
| Model Persistence | Joblib |

---

## 📁 Project Structure

```
heart-disease-predictor/
│
├── data/
│   └── heart.csv                   # UCI Heart Disease dataset
│
├── models/
│   ├── model.pkl                   # Saved best model (after training)
│   ├── scaler.pkl                  # Saved StandardScaler
│   ├── confusion_matrix.png        # Generated after training
│   └── feature_importance.png      # Generated if Random Forest wins
│
├── notebooks/
│   └── eda_and_training.ipynb      # Exploratory Data Analysis notebook
│
├── src/
│   ├── preprocess.py               # Data loading and preprocessing
│   └── train.py                    # Model training and evaluation
│
├── app/
│   └── app.py                      # Streamlit web application
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate (Mac/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `heart.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Place it in the `data/` folder:
```
data/heart.csv
```

---

## 🚀 How to Run

### Step 1 — Train the model

```bash
python src/train.py
```

This will:
- Load and preprocess the dataset
- Train and compare Logistic Regression, Random Forest, and SVM
- Save the best model to `models/model.pkl`
- Save the scaler to `models/scaler.pkl`
- Generate evaluation plots in `models/`

### Step 2 — Launch the web app

```bash
streamlit run app/app.py
```

Then open your browser at `http://localhost:8501`

---

## 📊 Dataset

**Source:** UCI Heart Disease Dataset (Cleveland subset)  
**Samples:** 303 patients  
**Features:** 13 clinical attributes  
**Target:** 0 = No Disease, 1 = Disease

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = Male, 0 = Female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels coloured by fluoroscopy (0–3) |
| thal | Thalassemia type (0–3) |

---

## 📈 Model Performance

After training, you'll see a comparison like:

```
Logistic Regression   → Acc: 0.8525 | AUC: 0.9201 | CV: 0.8416
Random Forest         → Acc: 0.8689 | AUC: 0.9312 | CV: 0.8350
SVM                   → Acc: 0.8361 | AUC: 0.9104 | CV: 0.8317
```

The best model is automatically selected and saved.

---

## 🖥️ App Preview

The Streamlit app features:
- Input sliders and dropdowns for all 13 features
- One-click prediction
- Probability score with visual progress bar
- Risk level indicator (Low / Moderate / High)
- Expandable input summary table

---

## ⚠️ Disclaimer

This tool is built for **educational purposes only** as part of a course project. It is **not a medical device** and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 👩‍💻 Author

**[SARAH AHMED]**  
**[REG NO: 25BCE11375]**
**[CSE (CORE)]**
Fundamentals of AI/ML — BYOP Submission  
VIT | VITyarthi Platform
