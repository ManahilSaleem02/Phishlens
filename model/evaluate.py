"""
evaluate.py  —  honest, reproducible model evaluation for the report / dashboard.

Runs OFFLINE (not part of the live API). It:
  1. Rebuilds the EXACT same train/test split used in training (seed 42).
  2. Trains two alternative models (Logistic Regression, Gradient Boosting) on
     the same features as the shipped Random Forest.
  3. Evaluates all three on the held-out test set.
  4. Writes model/evaluation.json in PLAIN-LANGUAGE terms (caught / missed /
     false alarms per 100 sites) so the dashboard can show it to non-experts.

Usage:  python -m model.evaluate
"""
import os, json, time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from utils import feature_extractor as fx

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BUNDLE = os.path.join(HERE, "phishing_model.joblib")
CSV = os.path.join(ROOT, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")
OUT = os.path.join(HERE, "evaluation.json")


def featurize(urls, tld_prob, char_logprob):
    rows = [fx.to_vector(fx.extract(u, tld_prob=tld_prob, char_logprob=char_logprob))
            for u in urls]
    return np.array(rows, dtype=float)


def friendly(y_true, y_pred, phish_label):
    """Translate raw results into 'per 100 sites' numbers anyone can read."""
    acc = accuracy_score(y_true, y_pred) * 100.0
    # phishing treated as the positive case we care about catching
    is_phish = (y_true == phish_label)
    is_legit = ~is_phish
    pred_phish = (y_pred == phish_label)

    caught = int(np.sum(is_phish & pred_phish))
    total_phish = int(np.sum(is_phish))
    false_alarms = int(np.sum(is_legit & pred_phish))
    total_legit = int(np.sum(is_legit))

    catch_rate = caught / total_phish * 100 if total_phish else 0
    false_alarm_rate = false_alarms / total_legit * 100 if total_legit else 0
    return {
        "accuracy": round(acc, 1),
        "phishing_caught_per_100": round(catch_rate, 1),
        "false_alarms_per_100_safe": round(false_alarm_rate, 1),
    }


def main():
    print("Loading data + model bundle ...")
    b = joblib.load(BUNDLE)
    tld_prob, char_logprob = b["tld_prob"], b["char_logprob"]
    phish_label = b["phish_label"]

    df = pd.read_csv(CSV, usecols=["URL", "label"])
    urls = df["URL"].astype(str).tolist()
    y = df["label"].to_numpy()

    # identical split to training, so the test set is genuinely held out for all
    u_tr, u_te, y_tr, y_te = train_test_split(
        urls, y, test_size=0.30, random_state=42, stratify=y)

    print(f"Featurizing {len(u_tr)} train + {len(u_te)} test URLs (this takes a few minutes) ...")
    t0 = time.time()
    X_tr = featurize(u_tr, tld_prob, char_logprob)
    X_te = featurize(u_te, tld_prob, char_logprob)
    print(f"  done in {time.time()-t0:.0f}s")

    results = []

    # 1) The model the app actually ships (already trained, in the bundle)
    rf = b["model"]
    results.append({
        "name": "Random Forest",
        "tag": "the method this app uses",
        "blurb": "Asks hundreds of small yes/no questions about the address and votes on the answer.",
        **friendly(y_te, rf.predict(X_te), phish_label),
    })

    # 2) Logistic Regression — simple, fast baseline
    print("Training Logistic Regression ...")
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(X_tr, y_tr)
    results.append({
        "name": "Logistic Regression",
        "tag": "a simple baseline",
        "blurb": "A straightforward formula that weighs each clue once. Fast, but less flexible.",
        **friendly(y_te, lr.predict(X_te), phish_label),
    })

    # 3) Gradient Boosting — strong modern alternative
    print("Training Gradient Boosting ...")
    gb = HistGradientBoostingClassifier(random_state=42)
    gb.fit(X_tr, y_tr)
    results.append({
        "name": "Gradient Boosting",
        "tag": "an advanced alternative",
        "blurb": "Builds on its own mistakes step by step. Powerful, a bit slower to train.",
        **friendly(y_te, gb.predict(X_te), phish_label),
    })

    # mark the top scorer by accuracy
    best_i = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    for i, r in enumerate(results):
        r["is_best"] = (i == best_i)

    # plain-language breakdown for the app's own model (Random Forest)
    app = results[0]
    caught = round(app["phishing_caught_per_100"])
    fa = round(app["false_alarms_per_100_safe"])
    breakdown = {
        "of_100_phishing": {"caught": caught, "missed": 100 - caught},
        "of_100_safe": {"passed": 100 - fa, "false_alarm": fa},
    }

    out = {
        "generated_at": time.strftime("%Y-%m-%d"),
        "test_site_count": int(len(u_te)),
        "models": results,
        "app_model": app["name"],
        "breakdown": breakdown,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
