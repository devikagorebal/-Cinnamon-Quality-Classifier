"""Train and compare 5 ML models for cinnamon quality classification.

Usage:  python train_model.py
- If data/cinnamon.csv exists it is used (columns below + 'quality').
- Otherwise a SYNTHETIC demo dataset is generated so the app runs.
  Replace it with your real internship dataset for real results.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

FEATURES = ["moisture", "ash", "volatile_oil", "acid_insoluble_ash", "chromium", "coumarin"]
CSV = "data/cinnamon.csv"


def make_demo_data(n=600, seed=42):
    rng = np.random.default_rng(seed)
    # (moisture, ash, volatile_oil, acid_insoluble_ash, chromium, coumarin) means per grade
    means = {"High": [9, 4.0, 3.0, 0.3, 0.5, 6], "Medium": [12, 5.5, 1.8, 0.8, 1.5, 18], "Low": [15, 7.5, 0.9, 1.5, 3.0, 35]}
    rows = []
    for label, m in means.items():
        x = rng.normal(m, [1.5, 0.8, 0.4, 0.2, 0.5, 5], size=(n // 3, 6)).clip(min=0)
        df = pd.DataFrame(x, columns=FEATURES)
        df["quality"] = label
        rows.append(df)
    return pd.concat(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    os.makedirs("ml", exist_ok=True)
    df = pd.read_csv(CSV) if os.path.exists(CSV) else make_demo_data()
    if not os.path.exists(CSV):
        print("No data/cinnamon.csv found - using SYNTHETIC demo data.")
    df = df.dropna().drop_duplicates()                      # basic cleaning
    X, y = df[FEATURES], df["quality"]

    enc = LabelEncoder().fit(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, enc.transform(y), test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    accuracies = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        accuracies[name] = round(accuracy_score(y_te, model.predict(X_te)) * 100, 2)
        print(f"{name:20s} {accuracies[name]}%")

    best = max(accuracies, key=accuracies.get)
    joblib.dump({"models": models, "scaler": scaler, "encoder": enc,
                 "accuracies": accuracies, "best": best, "features": FEATURES}, "ml/artifacts.joblib")
    print(f"\nBest model: {best}. Saved to ml/artifacts.joblib")


if __name__ == "__main__":
    main()
