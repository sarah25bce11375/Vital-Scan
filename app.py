#!/usr/bin/env python3
"""
VitalScan - Heart Disease Risk Predictor
Pure terminal. Input one by one. Output in terminal. No localhost. No GUI.

Run:
    python cli.py
    python cli.py --retrain
    python cli.py --help
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colors ────────────────────────────────────────────────────────────────────
R  = "\033[91m"   # red
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
B  = "\033[94m"   # blue
C  = "\033[96m"   # cyan
W  = "\033[1m"    # bold
X  = "\033[0m"    # reset

MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "scaler.pkl")

# ── Questions asked one by one ────────────────────────────────────────────────
QUESTIONS = [
    {
        "key":     "age",
        "prompt":  "Age (years)",
        "type":    int,
        "range":   (1, 120),
        "options": None,
    },
    {
        "key":     "sex",
        "prompt":  "Sex",
        "hint":    "0 = Female    1 = Male",
        "type":    int,
        "range":   None,
        "options": [0, 1],
    },
    {
        "key":     "cp",
        "prompt":  "Chest Pain Type",
        "hint":    "0 = Typical Angina    1 = Atypical Angina    2 = Non-anginal Pain    3 = Asymptomatic",
        "type":    int,
        "range":   None,
        "options": [0, 1, 2, 3],
    },
    {
        "key":     "trestbps",
        "prompt":  "Resting Blood Pressure (mm Hg)",
        "type":    int,
        "range":   (80, 200),
        "options": None,
    },
    {
        "key":     "chol",
        "prompt":  "Cholesterol (mg/dl)",
        "type":    int,
        "range":   (100, 600),
        "options": None,
    },
    {
        "key":     "fbs",
        "prompt":  "Fasting Blood Sugar > 120 mg/dl",
        "hint":    "0 = No    1 = Yes",
        "type":    int,
        "range":   None,
        "options": [0, 1],
    },
    {
        "key":     "restecg",
        "prompt":  "Resting ECG Result",
        "hint":    "0 = Normal    1 = ST-T Abnormality    2 = Left Ventricular Hypertrophy",
        "type":    int,
        "range":   None,
        "options": [0, 1, 2],
    },
    {
        "key":     "thalach",
        "prompt":  "Maximum Heart Rate Achieved",
        "type":    int,
        "range":   (60, 220),
        "options": None,
    },
    {
        "key":     "exang",
        "prompt":  "Exercise Induced Angina",
        "hint":    "0 = No    1 = Yes",
        "type":    int,
        "range":   None,
        "options": [0, 1],
    },
    {
        "key":     "oldpeak",
        "prompt":  "ST Depression (Oldpeak)",
        "type":    float,
        "range":   (0.0, 6.0),
        "options": None,
    },
    {
        "key":     "slope",
        "prompt":  "Slope of ST Segment",
        "hint":    "0 = Upsloping    1 = Flat    2 = Downsloping",
        "type":    int,
        "range":   None,
        "options": [0, 1, 2],
    },
    {
        "key":     "ca",
        "prompt":  "Number of Major Vessels (0-3)",
        "type":    int,
        "range":   None,
        "options": [0, 1, 2, 3],
    },
    {
        "key":     "thal",
        "prompt":  "Thalassemia",
        "hint":    "0 = Normal    1 = Fixed Defect    2 = Reversible Defect",
        "type":    int,
        "range":   None,
        "options": [0, 1, 2],
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def divider():
    print(f"{B}{'─' * 52}{X}")

def ask(q, number, total):
    """Ask a single question and return validated value."""
    print(f"\n  {W}[{number}/{total}] {q['prompt']}{X}")
    if q.get("hint"):
        print(f"  {Y}{q['hint']}{X}")
    if q.get("range"):
        print(f"  {Y}Valid range: {q['range'][0]} to {q['range'][1]}{X}")
    if q.get("options"):
        print(f"  {Y}Valid values: {q['options']}{X}")

    while True:
        try:
            raw = input(f"  {C}→ Enter value: {X}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{Y}Cancelled. Goodbye!{X}\n")
            sys.exit(0)

        # Validate type
        try:
            val = q["type"](raw)
        except ValueError:
            print(f"  {R}✗ Please enter a valid number.{X}")
            continue

        # Validate range
        if q["range"] and not (q["range"][0] <= val <= q["range"][1]):
            print(f"  {R}✗ Must be between {q['range'][0]} and {q['range'][1]}.{X}")
            continue

        # Validate options
        if q["options"] and val not in q["options"]:
            print(f"  {R}✗ Must be one of {q['options']}.{X}")
            continue

        print(f"  {G}✓ Recorded: {val}{X}")
        return val

def collect_inputs():
    """Ask all questions one by one and return list of values."""
    total = len(QUESTIONS)
    values = []
    for i, q in enumerate(QUESTIONS, 1):
        val = ask(q, i, total)
        values.append(val)
    return values

def predict(values):
    """Run the model and return probability."""
    import joblib
    import numpy as np

    if not os.path.exists(MODEL_PATH):
        print(f"\n{R}ERROR: Model file not found.{X}")
        print(f"{Y}Run first: python cli.py --retrain{X}\n")
        sys.exit(1)

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X      = np.array(values, dtype=float).reshape(1, -1)
    X_sc   = scaler.transform(X)
    prob   = model.predict_proba(X_sc)[0][1]
    return prob

def show_result(prob):
    """Print final result to terminal."""
    pct = round(prob * 100, 1)

    filled = int(pct / 5)
    bar    = "█" * filled + "░" * (20 - filled)

    if pct < 35:
        color, risk, advice = G, "LOW RISK",      "Low likelihood of heart disease. Keep up the healthy lifestyle!"
    elif pct < 65:
        color, risk, advice = Y, "MODERATE RISK", "Moderate risk detected. Consider consulting a doctor."
    else:
        color, risk, advice = R, "HIGH RISK",     "High risk detected. Please consult a doctor immediately."

    print(f"\n")
    divider()
    print(f"  {W}PREDICTION RESULT{X}")
    divider()
    print(f"  Probability  :  {W}{pct}%{X}")
    print(f"  Risk Bar     :  [{color}{bar}{X}]")
    print(f"  Risk Level   :  {color}{W}{risk}{X}")
    print(f"  Advice       :  {advice}")
    divider()
    print(f"\n  {Y}This is for educational purposes only.{X}")
    print(f"  {Y}Always consult a qualified doctor.{X}\n")

def retrain():
    """Retrain model via terminal."""
    print(f"\n{Y}Retraining model...{X}\n")
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    result = subprocess.run([sys.executable, train_script])
    if result.returncode == 0:
        print(f"\n{G}✓ Model retrained and saved successfully.{X}\n")
    else:
        print(f"\n{R}✗ Retraining failed. Check train.py for errors.{X}\n")
        sys.exit(1)

def banner():
    print(f"""
{R}{W}
  ██╗   ██╗██╗████████╗ █████╗ ██╗      ███████╗ ██████╗ █████╗ ███╗   ██╗
  ██║   ██║██║╚══██╔══╝██╔══██╗██║      ██╔════╝██╔════╝██╔══██╗████╗  ██║
  ██║   ██║██║   ██║   ███████║██║      ███████╗██║     ███████║██╔██╗ ██║
  ╚██╗ ██╔╝██║   ██║   ██╔══██║██║      ╚════██║██║     ██╔══██║██║╚██╗██║
   ╚████╔╝ ██║   ██║   ██║  ██║███████╗ ███████║╚██████╗██║  ██║██║ ╚████║
    ╚═══╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝
{X}
  {W}Heart Disease Risk Predictor  |  Pure Terminal  |  No GUI  |  No Browser{X}
""")
    divider()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="VitalScan - Heart Disease Risk Predictor (Terminal Only, No GUI)",
    )
    parser.add_argument("--retrain", action="store_true", help="Retrain the ML model")
    parser.add_argument("--version", action="store_true", help="Show version")
    args = parser.parse_args()

    if args.version:
        print(f"VitalScan CLI v1.0 | Python {sys.version.split()[0]}")
        sys.exit(0)

    if args.retrain:
        retrain()
        sys.exit(0)

    # ── Normal flow ───────────────────────────────────────────────────────────
    banner()
    print(f"  Answer {len(QUESTIONS)} questions below. The model will predict your heart disease risk.\n")

    while True:
        values = collect_inputs()

        print(f"\n{Y}  Analyzing your data...{X}")
        prob = predict(values)
        show_result(prob)

        try:
            again = input(f"  {C}Run another prediction? (y/n): {X}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            again = "n"

        if again != "y":
            print(f"\n{G}  Goodbye! Stay healthy. ❤️{X}\n")
            break
        else:
            print(f"\n{B}{'─'*52}{X}\n")

if __name__ == "__main__":
    main()
