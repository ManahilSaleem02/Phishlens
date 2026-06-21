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

## The key engineering decision (please read this)

Your notebook trained a RandomForest on **all 50 columns** of the PhiUSIIL dataset.
About 27 of those are **page-content features** — `LineOfCode`, `HasTitle`,
`NoOfJS`, `HasPasswordField`, `NoOfiFrame`, etc. Those can only be obtained by
actually **fetching and rendering the live HTML** of the URL. At inference time your
function either had to scrape the page (slow, unreliable, and a phishing page may be
dead by the time you check it) or fill them with zeros/defaults — which means the
model sees inputs it was never trained on. That's **train/serve skew**, and it's the
single most common reason a "99% accurate" notebook model behaves randomly in
production.

A few more landmines in the raw dataset:

- **`URLSimilarityIndex`** is almost perfectly correlated with the label — it's
  effectively a leak. Training on it gives a fantastic-looking score that doesn't
  generalize.
- **Path artifact:** ~0% of the *legitimate* URLs in PhiUSIIL have a path, vs ~27% of
  phishing ones. A naive model learns "has a path → phishing", which would flag
  `github.com/yourname` as malicious.
- **`www.` artifact:** nearly all legit URLs start with `www.`, so bare domains like
  `github.com` get unfairly penalized.

### What I did instead

1. **Retrained a URL-only model** using a *single shared feature extractor*
   (`utils/feature_extractor.py`) that runs identically during training and
   inference. Zero skew — what the model learns is exactly what it sees in prod.
2. **Judged the host/domain structure, not the path.** Legit deep links generalize,
   so `rspapp.ai/dashboard` and `github.com/ayanmurad987` are treated fairly.
3. **Normalized the leading `www.`** so bare and `www.` domains score the same.
4. **Dropped the leaky/artifact features** (`URLSimilarityIndex`, `URLLength`,
   `PathLength`).
5. **Pushed the full-URL lures into a rule layer** instead of the model: IP-as-host,
   `@` in the URL, punycode/`xn--`, missing HTTPS, risky TLDs (`.tk .cf .ml .gq`…),
   brand-bait keywords (`secure`, `login`, `verify`, `update`…), excessive
   subdomains, obfuscation. These are *explainable*, which is exactly what you
   wanted — the user sees a reason, not just a number.
6. **Brand-impersonation detection** for typosquats and homoglyphs: a domain like
   `paypa1.com` (digit `1` for `l`) or `amaz0n.com` is de-glyphed and edit-distance
   compared to the brand allowlist. A look-alike that *isn't* the real brand is a
   high-severity flag. Because the host-only ML model literally cannot perceive this
   (it's a short, clean, HTTPS `.com` host), these near-certain signals — along with
   IP-as-host and punycode — impose a **risk floor** so they can't be averaged away
   by the model.

The final prediction **blends** the ML risk with the rule layer
(`predictor.py`, `RULE_WEIGHT = 0.45`), and a small **trusted-domain allowlist**
(~24 megabrands) keeps things like `accounts.google.com` from tripping over their own
"login" keyword.

The model uses 17 host-focused features (HTTPS, TLD legitimacy probability, character
continuation rate, dot/subdomain counts, digit/letter/special-char ratios, a
character-model probability for the host, etc.). The two probability lookups
(`tld_prob`, `char_logprob`) are built **from the training split only** and saved
inside the joblib bundle.

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
