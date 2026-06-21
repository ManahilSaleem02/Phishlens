"""
Prediction service. Loads the trained bundle once and exposes predict(url),
which returns a fully structured, explainable result combining:
  * the ML model's probability (domain/host reputation), and
  * the rule-based reasoning layer (full-URL lures + trust signals).
"""

from __future__ import annotations

import os
import joblib

from utils import feature_extractor as fx
from utils import explainer as ex

_BUNDLE = None
_BUNDLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "model", "phishing_model.joblib")

# How much the transparent rule layer can move the ML risk. The model stays
# the primary decision-maker; rules refine and explain.
RULE_WEIGHT = 0.45


def load(bundle_path: str | None = None):
    global _BUNDLE
    if _BUNDLE is None:
        path = bundle_path or _BUNDLE_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Run: python -m model.train_model")
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def model_info() -> dict:
    b = load()
    return {
        "trained_at": b.get("trained_at"),
        "n_train": b.get("n_train"),
        "n_test": b.get("n_test"),
        "metrics": b.get("metrics"),
        "top_features": dict(list(b.get("feature_importances", {}).items())[:8]),
    }


def _label(risk_score: float) -> str:
    if risk_score >= 70:
        return "phishing"
    if risk_score >= 40:
        return "suspicious"
    return "legitimate"


def _reconcile_reasons(reasons, label, registered_domain, risk):
    """Keep the 'Why this verdict' list consistent with the actual risk score.

    Three cases are handled so the explanation never contradicts the gauge:
      * risky verdict  -> drop reassuring notes; ensure a concern-level reason.
      * borderline-legit (model a bit unsure, no rule fired) -> replace the
        'all clear' note with a calibrated 'only moderately confident' note.
      * clearly clean  -> leave the positive confirmations as-is."""
    if label != "legitimate":
        # the trusted-domain note pulls risk down, so it can't co-occur with a
        # risky verdict; dropping info notes here just removes reassurances.
        reasons = [r for r in reasons if r["severity"] != "info"]
        if not any(r["severity"] in ("high", "medium") for r in reasons):
            tld = registered_domain.rsplit(".", 1)[-1] if "." in registered_domain else ""
            ending = f" — its '.{tld}' ending" if tld else ""
            txt = ("No single obvious trick in the address, but the model rated the domain "
                   f"itself risky{ending} and overall name pattern resemble sites it "
                   "learned to flag as phishing.")
            reasons.insert(0, {"severity": "high" if label == "phishing" else "medium",
                               "text": txt})
        return reasons

    # legitimate, but the model wasn't fully confident and no rule explained it
    has_concern = any(r["severity"] in ("high", "medium", "low") for r in reasons)
    if risk >= 25 and not has_concern:
        reasons = [r for r in reasons if "no suspicious patterns" not in r["text"]]
        reasons.append({"severity": "low",
                        "text": "Nothing clearly wrong with the address, but the model is only "
                                "moderately confident here — the domain's overall pattern is a little "
                                "less typical than well-established sites, so treat it with mild caution."})
    return reasons


def predict(url: str) -> dict:
    """Return a structured, explainable prediction for a single URL."""
    if not url or not url.strip():
        raise ValueError("Empty URL")

    b = load()
    model = b["model"]
    legit_idx = list(model.classes_).index(b["legit_label"])
    importances = b.get("feature_importances", {})

    feats = fx.extract(url, b["tld_prob"], b["char_logprob"])
    vec = [fx.to_vector(feats)]
    p_legit = float(model.predict_proba(vec)[0][legit_idx])
    ml_risk = (1.0 - p_legit) * 100.0  # 0..100, higher = more phishing

    reasons, rule_delta, meta = ex.build_reasons(url, feats)

    # Blend: shift ML risk by the rule delta (scaled). Clamp to [0,100].
    final_risk = ml_risk + RULE_WEIGHT * rule_delta * 100.0
    final_risk = max(0.0, min(100.0, final_risk))

    # Near-certain signals (brand impersonation, IP host, punycode) the host-only
    # model cannot perceive impose a minimum risk so they aren't averaged away.
    floor = float(meta.get("risk_floor", 0.0))
    if floor > final_risk:
        final_risk = floor

    label = _label(final_risk)
    reasons = _reconcile_reasons(reasons, label, meta["registered_domain"], final_risk)

    # Attach the model's own top contributing features (global importance)
    # alongside this URL's actual values for transparency.
    feature_contributions = []
    for name, imp in list(importances.items())[:6]:
        feature_contributions.append({
            "feature": name,
            "importance": round(imp, 4),
            "value": round(float(feats[name]), 4),
        })

    return {
        "url": url.strip(),
        "prediction": label,
        "risk_score": round(final_risk, 1),
        "ml_risk_score": round(ml_risk, 1),
        "confidence": round(abs(final_risk - 50) / 50 * 100, 1),
        "reasons": reasons,
        "rule_adjustment": round(RULE_WEIGHT * rule_delta * 100.0, 1),
        "registered_domain": meta["registered_domain"],
        "trusted_domain": meta["trusted"],
        "top_feature_contributions": feature_contributions,
        "extracted_features": {k: round(float(v), 4) for k, v in feats.items()},
    }
