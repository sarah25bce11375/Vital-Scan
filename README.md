# VitalScan - Heart Disease Risk Predictor

A machine learning project that predicts the likelihood of heart disease based on clinical health parameters. Built as a BYOP (Bring Your Own Project) submission for the Fundamentals of AI/ML course.

This project runs entirely from the terminal. There is no web interface, no browser, and no localhost server. All input is taken from the shell and all output is printed to the shell.

---

## Problem Statement

Cardiovascular disease is the leading cause of death in India, responsible for over 28% of all fatalities. Most people are unaware of their risk until symptoms appear, often too late for early intervention. This project provides a simple, accessible command-line tool for early self-assessment using well-established clinical indicators from the UCI Heart Disease dataset.

---

## What It Does

- Accepts 13 clinical inputs one by one through the terminal
- Passes them through a trained machine learning classification model
- Prints a risk prediction with a probability score directly in the terminal
- Categorises the result as Low, Moderate, or High risk

---

## Tech Stack

| Layer             | Tools                                              |
|-------------------|----------------------------------------------------|
| Language          | Python 3.10+                                       |
| Data Processing   | Pandas, NumPy                                      |
| ML Models         | scikit-learn (Logistic Regression, Random Forest, SVM) |
| Evaluation        | classification_report, ROC-AUC, cross-validation  |
| Visualisation     | Matplotlib, Seaborn                                |
| Model Persistence | Joblib                                             |
| Interface         | Pure terminal (stdin / stdout)                     |

---

## Project Structure

```
VitalScan/
|
|-- app.py                  # Main entry point (CLI application)
|-- train.py                # Model training and evaluation script
|-- requirements.txt        # All dependencies
|-- README.md
|-- .gitignore
|
|-- data/
|   `-- heart.csv           # UCI Heart Disease dataset
|
|-- models/
|   |-- model.pkl           # Saved best model (generated after training)
|   |-- scaler.pkl          # Saved StandardScaler (generated after training)
|   |-- confusion_matrix.png
|   `-- feature_importance.png
|
`-- src/
    |-- __init__.py
    `-- preprocess.py       # Data loading and preprocessing logic
```

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vitalscan.git
cd vitalscan
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Git Bash)
source .venv/Scripts/activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This must be done once before running the app. It generates the model and scaler files inside the models/ folder.

```bash
python train.py
```

What this does:
- Loads and preprocesses heart.csv
- Trains and compares Logistic Regression, Random Forest, and SVM
- Selects the best model automatically
- Saves model.pkl and scaler.pkl to models/
- Generates confusion_matrix.png and feature_importance.png

---

## How to Run

```bash
python app.py
```

The app will ask you 13 clinical questions one by one in the terminal. After you answer all of them, it prints the prediction result directly in the terminal.

To retrain the model:

```bash
python app.py --retrain
```

To check the version:

```bash
python app.py --version
```

---

## User Flow

```
Run: python app.py
        |
        v
Answer Question 1 of 13  (e.g. Age)
        |
        v
Answer Question 2 of 13  (e.g. Sex)
        |
       ...
        |
        v
Answer Question 13 of 13  (e.g. Thalassemia)
        |
        v
Model runs prediction
        |
        v
Result printed in terminal:
  - Probability percentage
  - Visual risk bar
  - Risk level: LOW / MODERATE / HIGH
  - Advice message
        |
        v
Option to run another prediction or exit
```

Each question includes a hint showing valid values. If an invalid value is entered, the app rejects it and asks again. No input is sent to the model until all 13 answers are validated.

---

## Dataset

**Source:** UCI Heart Disease Dataset (Cleveland subset)
**Samples:** 303 patients
**Features:** 13 clinical attributes
**Target:** 0 = No Disease, 1 = Disease

| Feature  | Description                                      |
|----------|--------------------------------------------------|
| age      | Age in years                                     |
| sex      | 1 = Male, 0 = Female                             |
| cp       | Chest pain type (0-3)                            |
| trestbps | Resting blood pressure (mm Hg)                   |
| chol     | Serum cholesterol (mg/dl)                        |
| fbs      | Fasting blood sugar > 120 mg/dl (1 = true)       |
| restecg  | Resting ECG results (0-2)                        |
| thalach  | Maximum heart rate achieved                      |
| exang    | Exercise induced angina (1 = yes)                |
| oldpeak  | ST depression induced by exercise                |
| slope    | Slope of peak exercise ST segment (0-2)          |
| ca       | Number of major vessels coloured by fluoroscopy (0-3) |
| thal     | Thalassemia type (0-2)                           |

---

## Model Performance

After training, the terminal prints a comparison like:

```
Logistic Regression   -> Acc: 0.8033 | AUC: 0.8690 | CV: 0.8349
Random Forest         -> Acc: 0.8361 | AUC: 0.9091 | CV: 0.8382
SVM                   -> Acc: 0.8361 | AUC: 0.8864 | CV: 0.8183

Best model: Random Forest
```

The best model is automatically selected and saved to models/model.pkl.

---

## Disclaimer

This tool is built for educational purposes only as part of a course project. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## Author

**Sarah Ahmed**
Registration No: 25BCE11375
CSE (Core)
Fundamentals of AI/ML — BYOP Submission
VIT | VITyarthi Platform