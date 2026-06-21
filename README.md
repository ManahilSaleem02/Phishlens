# PhishLens — AI-Powered URL Phishing Detection System

A production-style, full-stack phishing URL detector. Paste a URL → get a phishing
risk score (0–100%), a clear verdict (legitimate / suspicious / phishing), and a
**human-readable explanation** of *why* — both from the ML model and from a
rule-based reasoning layer.

```
User → Frontend Dashboard → FastAPI → Feature Extraction → ML Model
                                              ↓                 ↓
                                        Rule Explainer ←────────┘
                                              ↓
                                   Blended Score + Reasons → JSON → UI
```

---

## What's inside

```
phishing-detector/
├── backend/
│   ├── main.py              # FastAPI app: /api/predict, /api/history, /api/stats, /health
│   └── database.py          # sqlite3 scan history + stats (stdlib only)
├── frontend/
│   └── index.html           # single-file dashboard (no build step, dark "forensic" UI)
├── model/
│   ├── train_model.py       # retrains the model from the CSV
│   └── phishing_model.joblib # trained bundle (model + lookups + metrics)
├── utils/
│   ├── feature_extractor.py # SHARED extractor — used by BOTH training and inference
│   ├── explainer.py         # rule-based reasoning layer + trusted-domain allowlist
│   └── predictor.py         # loads model, blends ML + rules, returns structured result
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv   # Kaggle source (only needed to retrain)
├── requirements.txt
├── run.sh                   # trains if needed, then launches the server
└── test_predict.py          # CLI sanity tester (no server required)
```

---

## Quick start

```bash
pip install -r requirements.txt

# Option A — one command (trains only if the model is missing, then serves)
./run.sh

# Option B — manual
python -m model.train_model        # only if you want to retrain
uvicorn backend.main:app --port 8000
```

Then open **http://localhost:8000** in your browser. The dashboard is served by the
same FastAPI process, so there's no separate frontend server and no CORS hassle.

Want to test without a browser?

```bash
python test_predict.py https://paypal-secure-login.tk/verify
```

---


## Honest model metrics

Held-out test set (30% stratified split, never seen during training):

| Metric                  | Value  |
|-------------------------|--------|
| Accuracy                | 96.3%  |
| Phishing precision      | 98.7%  |
| Phishing recall         | 92.6%  |
| Phishing F1             | 95.6%  |
| ROC-AUC                 | 0.980  |
| 3-fold CV accuracy      | 95.6% ± 0.07% |

The tiny gap between CV and test accuracy means it's **not overfitting**. Top
features by importance: `IsHTTPS` (0.39), `TLDLegitimateProb` (0.19),
`CharContinuationRateHost` (0.10), `NoOfDotsInHost`, `HostCharProb`, `NoOfSubDomain`.

### Model comparison (the "How good is the model?" tab)

`python -m model.evaluate` retrains two alternatives on the **same features and the
same held-out test set** and writes `model/evaluation.json`, which the dashboard's
**"How good is the model?"** tab renders in plain language (caught / missed / false
alarms per 100 sites). Results on 70,739 unseen sites:

| Method | Accuracy | Phishing caught /100 | False alarms /100 |
|--------|----------|----------------------|-------------------|
| **Random Forest (this app)** | **96.3%** | 93 | ~1 |
| Gradient Boosting | 96.0% | 92 | ~1 |
| Logistic Regression | 93.6% | 88 | ~3 |

The Random Forest is the top scorer, which justifies the model choice with evidence
rather than assumption. The JSON ships pre-generated so the tab works out of the box;
re-run the command anytime to regenerate it.

> **Reality check:** that 96% is partly inflated by a dataset artifact — *every*
> legitimate URL in PhiUSIIL is HTTPS, so the model leans hard on `IsHTTPS`. In the
> real world plenty of phishing sites now have valid HTTPS certs (Let's Encrypt is
> free), so expect real-world accuracy to be **lower** than the headline number. The
> rule layer exists partly to compensate for this. Treat the output as a
> **risk advisory, not a verdict.**

---

## API

`POST /api/predict`  body: `{ "url": "https://example.com" }`

```jsonc
{
  "url": "https://paypal-secure-login.tk/verify",
  "prediction": "phishing",          // legitimate | suspicious | phishing
  "risk_score": 100.0,               // blended 0–100
  "ml_risk_score": 88.7,             // model-only, before rules
  "confidence": 100.0,
  "rule_adjustment": 27.0,           // how much the rules moved the score
  "registered_domain": "paypal-secure-login.tk",
  "trusted_domain": false,
  "reasons": [
    { "severity": "medium", "text": "Uses a high-risk TLD (.tk) …" },
    { "severity": "medium", "text": "Contains brand-bait keywords: secure, login, verify" }
  ],
  "top_feature_contributions": [ … ],
  "extracted_features": { … }
}
```

Other endpoints: `GET /health`, `GET /api/history?limit=20`, `GET /api/stats`.

**Built-in security/ops:** in-memory token-bucket **rate limiting** (60 req/min per
IP on `/api/`), per-request **logging** to `backend/predictions.log`, and SQLite
**scan history** in `backend/scans.db` (both created on first run).

---

## Known limitations & extension points

- **Dataset artifacts** (HTTPS / homepage / `www.`) inflate the offline score — see
  the reality check above. Retraining on a fresher, more balanced feed (e.g.
  PhishTank + Tranco) would harden it.
- **Typosquat detection is allowlist-bound.** Impersonations are only caught for
  brands in the protected list (`utils/brands.txt` — ~150 popular sites plus the
  core allow-list). `brain1y.com` is caught because `brainly.com` is listed; a
  look-alike of a brand that *isn't* listed won't be. Just add a line to
  `utils/brands.txt` to widen coverage (a real deployment would load the Tranco
  top-1M here).
- **No live domain-age lookup.** You listed this as optional. It's a clean drop-in:
  add a WHOIS/RDAP call in `explainer.py` and append a reason when a domain is only
  days old. I left it out so the system has no external API dependency or latency.
- **PDF report & React frontend** (both on your optional list) aren't built. The
  frontend is a single self-contained `index.html` so it has no build step — to move
  to React, point a Vite/Next app at the same `/api/*` endpoints; the contract won't
  change. A PDF report can be generated server-side from the same JSON.

---

*Advisory tool. A "legitimate" result is not a guarantee of safety — always verify
sensitive sites manually.*
