"""
Train the phishing-URL classifier.

Run from the project root:
    python -m model.train_model

What it does:
  1. Loads the PhiUSIIL CSV (URL + label only — we recompute every feature
     ourselves so the model never sees a feature we can't reproduce live).
  2. Splits into train/test on raw URLs.
  3. Learns two lookup tables FROM THE TRAIN SPLIT ONLY:
       - TLD -> P(legitimate)        (TLD reputation)
       - char -> log P(char | legit) (character model for URLCharProb)
  4. Extracts features for every URL via utils.feature_extractor.
  5. Trains a regularised RandomForest (depth / leaf limits to curb overfit).
  6. Reports honest metrics and saves a single joblib bundle.
"""

import os
import sys
import math
import time
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import feature_extractor as fx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")
BUNDLE_PATH = os.path.join(ROOT, "model", "phishing_model.joblib")

# In PhiUSIIL: label 1 == legitimate, label 0 == phishing.
LEGIT, PHISH = 1, 0


def build_tld_prob(urls, labels, smoothing=5.0):
    """P(legit | TLD) with additive smoothing toward the global legit rate.
    Smoothing keeps rare TLDs from collapsing to 0 or 1 on a few samples."""
    legit_count = defaultdict(float)
    total_count = defaultdict(float)
    for u, y in zip(urls, labels):
        tld = fx.parse_parts(u)["tld"]
        total_count[tld] += 1
        if y == LEGIT:
            legit_count[tld] += 1
    global_rate = float(np.mean([1.0 if y == LEGIT else 0.0 for y in labels]))
    table = {}
    for tld, n in total_count.items():
        table[tld] = (legit_count[tld] + smoothing * global_rate) / (n + smoothing)
    table["__default__"] = global_rate  # unseen TLD at inference
    return table


def build_char_model(urls, labels):
    """log P(char) estimated from legitimate URLs (Laplace-smoothed).
    Used by URLCharProb to flag character mixes unlike normal sites."""
    counts = Counter()
    for u, y in zip(urls, labels):
        if y == LEGIT:
            counts.update(u.lower())
    total = sum(counts.values())
    vocab = len(counts) + 1
    logprob = {}
    for ch, c in counts.items():
        logprob[ch] = math.log((c + 1) / (total + vocab))
    logprob["__floor__"] = math.log(1 / (total + vocab))  # unseen char
    return logprob


def featurize(urls, tld_prob, char_logprob):
    rows = []
    for u in urls:
        feats = fx.extract(u, tld_prob=tld_prob, char_logprob=char_logprob)
        rows.append(fx.to_vector(feats))
    return np.asarray(rows, dtype=float)


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"Dataset not found at {CSV_PATH}. Put the CSV in data/.")

    print("Loading dataset ...")
    df = pd.read_csv(CSV_PATH, usecols=["URL", "label"])
    df = df.dropna(subset=["URL", "label"])
    urls = df["URL"].astype(str).tolist()
    y = df["label"].astype(int).values
    print(f"  {len(urls):,} rows | legit={int((y==LEGIT).sum()):,} "
          f"phish={int((y==PHISH).sum()):,}")

    # Split on raw URLs first so lookup tables only see training data.
    u_tr, u_te, y_tr, y_te = train_test_split(
        urls, y, test_size=0.30, random_state=42, stratify=y)

    print("Learning TLD reputation + character model from train split ...")
    tld_prob = build_tld_prob(u_tr, y_tr)
    char_logprob = build_char_model(u_tr, y_tr)

    print("Extracting features (train) ...")
    t0 = time.time()
    X_tr = featurize(u_tr, tld_prob, char_logprob)
    print(f"  {X_tr.shape} in {time.time()-t0:.1f}s")
    print("Extracting features (test) ...")
    X_te = featurize(u_te, tld_prob, char_logprob)

    print("Training RandomForest (regularised) ...")
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,            # cap depth -> less overfit
        min_samples_leaf=4,      # leaves must generalise
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    # ---- honest evaluation ----
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, list(model.classes_).index(LEGIT)]
    acc = accuracy_score(y_te, pred)
    prec = precision_score(y_te, pred, pos_label=PHISH)
    rec = recall_score(y_te, pred, pos_label=PHISH)
    f1 = f1_score(y_te, pred, pos_label=PHISH)
    auc = roc_auc_score((y_te == LEGIT).astype(int), proba)

    print("\n================ TEST METRICS ================")
    print(f"Accuracy            : {acc:.4f}")
    print(f"Phishing precision  : {prec:.4f}")
    print(f"Phishing recall     : {rec:.4f}")
    print(f"Phishing F1         : {f1:.4f}")
    print(f"ROC-AUC             : {auc:.4f}")
    print("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred,
          target_names=["phishing(0)", "legit(1)"]))

    # Quick overfit check via 3-fold CV on a subsample (keeps it fast).
    print("Cross-val (3-fold, 40k subsample) for overfit sanity ...")
    idx = np.random.RandomState(0).choice(len(X_tr), size=min(40000, len(X_tr)),
                                          replace=False)
    cv = cross_val_score(
        RandomForestClassifier(n_estimators=120, max_depth=18,
                               min_samples_leaf=4, max_features="sqrt",
                               class_weight="balanced", n_jobs=-1,
                               random_state=42),
        X_tr[idx], y_tr[idx], cv=3, scoring="accuracy")
    print(f"  CV accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")

    importances = dict(sorted(
        zip(fx.FEATURE_ORDER, model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True))

    bundle = {
        "model": model,
        "feature_order": fx.FEATURE_ORDER,
        "tld_prob": tld_prob,
        "char_logprob": char_logprob,
        "legit_label": LEGIT,
        "phish_label": PHISH,
        "feature_importances": importances,
        "metrics": {"accuracy": acc, "phish_precision": prec,
                    "phish_recall": rec, "phish_f1": f1, "roc_auc": auc,
                    "cv_accuracy_mean": float(cv.mean()),
                    "cv_accuracy_std": float(cv.std())},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": len(u_tr), "n_test": len(u_te),
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    print(f"\nSaved model bundle -> {BUNDLE_PATH}")
    print("Top features:", json.dumps(
        {k: round(v, 4) for k, v in list(importances.items())[:8]}, indent=2))


if __name__ == "__main__":
    main()
