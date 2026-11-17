"""Model training + prediction helpers for the Sonar Rock vs Mine project."""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path(__file__).parent / "sonar_data.csv"
N_FEATURES = 60
LABELS = {"R": "ROCK", "M": "MINE"}


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, header=None)


def build_models() -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=1),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=1)
        ),
    }


def train_all():
    """Train every model; return (fitted_models, metrics_df, confusions, test_df)."""
    df = load_data()
    X = df.drop(columns=60)
    y = df[60]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1
    )

    models, rows, confusions = {}, [], {}
    for name, model in build_models().items():
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        cv = cross_val_score(model, X, y, cv=skf).mean()  # on the full dataset
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        rows.append(
            {
                "Model": name,
                "Train accuracy": accuracy_score(y_tr, model.predict(X_tr)),
                "Test accuracy": accuracy_score(y_te, pred),
                "5-fold CV accuracy": cv,
            }
        )
        confusions[name] = confusion_matrix(y_te, pred, labels=["R", "M"])
        models[name] = model

    metrics = pd.DataFrame(rows).set_index("Model")
    test_df = X_te.copy()
    test_df["label"] = y_te
    return models, metrics, confusions, test_df


def parse_readings(text: str) -> np.ndarray:
    """Turn pasted text (commas / spaces / new lines) into a float array of 60."""
    cleaned = text.replace("\n", ",").replace("\t", ",").replace(";", ",")
    parts = [p for p in cleaned.replace(" ", ",").split(",") if p.strip()]
    try:
        values = np.array([float(p) for p in parts])
    except ValueError:
        raise ValueError("Only numbers are allowed (separated by commas or spaces).")
    if len(values) != N_FEATURES:
        raise ValueError(
            f"Expected exactly {N_FEATURES} readings, but got {len(values)}."
        )
    return values


def predict(model, values: np.ndarray):
    """Return (label_code, confidence, {code: probability})."""
    x = np.asarray(values, dtype=float).reshape(1, -1)
    probs = model.predict_proba(x)[0]
    classes = list(model.classes_)
    prob_map = {c: float(p) for c, p in zip(classes, probs)}
    code = max(prob_map, key=prob_map.get)
    return code, prob_map[code], prob_map
