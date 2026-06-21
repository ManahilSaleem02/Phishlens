"""
Explainability + rule-based reasoning layer.

The ML model gives a probability. This layer turns the prediction into
human-readable WHY, by inspecting the raw URL with deterministic rules, and
applies small, transparent adjustments to the final risk (e.g. an IP-address
host always pushes risk up; a well-known trusted domain over HTTPS pulls it
down). Every adjustment is reported, so nothing is a black box.
"""

from __future__ import annotations

import os

from utils import feature_extractor as fx

# A tiny allow-list of high-traffic domains. Real deployments would load a
# much larger curated list (e.g. Tranco top-N). Megabrand sign-in pages are
# structurally similar to phishing, so an allow-list is the standard fix.
TRUSTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "github.com", "microsoft.com", "apple.com", "linkedin.com", "x.com",
    "twitter.com", "instagram.com", "netflix.com", "openai.com", "anthropic.com",
    "cloudflare.com", "paypal.com", "bankofamerica.com", "chase.com",
    "stackoverflow.com", "reddit.com", "yahoo.com", "live.com", "office.com",
}

# Widen coverage from an editable data file (utils/brands.txt). This is the easy
# place to extend protection — a production system would load the Tranco top-1M.
def _load_brand_file():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brands.txt")
    out = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    out.add(line)
    except OSError:
        pass
    return out


TRUSTED_DOMAINS |= _load_brand_file()

# Suspicious TLDs frequently abused for free/cheap phishing domains.
RISKY_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
              "country", "stream", "download", "loan", "zip", "mov"}

# Brand names (the SLD, i.e. part before the final TLD) we protect against
# look-alikes. Built from the trusted list; short/generic ones are excluded so
# we don't fire on every 4-letter domain.
BRAND_NAMES = {d.rsplit(".", 1)[0] for d in TRUSTED_DOMAINS}
BRAND_NAMES = {b for b in BRAND_NAMES if len(b) >= 5}

# Common visual / keyboard substitutions attackers use to imitate a brand.
# Map look-alike -> the letter it imitates, then compare to the real brand.
_HOMOGLYPHS = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "6": "b", "7": "t", "8": "b", "9": "g", "$": "s", "@": "a",
}
_HOMOGLYPH_PAIRS = (("rn", "m"), ("vv", "w"), ("cl", "d"))


def _deglyph(s: str) -> str:
    """Undo common homoglyph substitutions to expose brand impersonation."""
    s = s.lower()
    for pair, rep in _HOMOGLYPH_PAIRS:
        s = s.replace(pair, rep)
    return "".join(_HOMOGLYPHS.get(c, c) for c in s)


def _levenshtein(a: str, b: str) -> int:
    """Edit distance (pure stdlib) — small strings, so the simple DP is fine."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lookalike_brand(reg: str):
    """If `reg` (eTLD+1) imitates a protected brand without BEING it, return
    (brand, kind) where kind is 'homoglyph' or 'typo'. Else None."""
    sld = reg.rsplit(".", 1)[0].lower()
    if not sld or sld in BRAND_NAMES:          # exact brand SLD -> not a fake
        return None
    deglyphed = _deglyph(sld)
    # 1) character-swap impersonation: paypa1 -> paypal, amaz0n -> amazon
    if deglyphed in BRAND_NAMES and deglyphed != sld:
        return (deglyphed, "homoglyph")
    # 2) near-miss typo: gogle/googel ~ google, paypall ~ paypal
    for brand in BRAND_NAMES:
        if abs(len(sld) - len(brand)) <= 2 and _levenshtein(sld, brand) == 1:
            return (brand, "typo")
    return None


def _registered_domain(parts: dict) -> str:
    """Best-effort eTLD+1 (no public-suffix lib to stay dependency-light)."""
    labels = parts["labels"]
    if parts["is_ip"] or len(labels) < 2:
        return parts["host"]
    return ".".join(labels[-2:])


def build_reasons(url: str, feats: dict | None = None):
    """Return (reasons, risk_delta, meta). reasons is a list of dicts:
    {severity: high|medium|low|info, text: str}. risk_delta in [-1,1] is a
    transparent nudge applied to the ML risk. When `feats` (the extracted
    feature values for this URL) is given, softer model-aware signals are added
    so the explanation reflects what the model actually weighed, not just hard
    rules."""
    parts = fx.parse_parts(url)
    sig = fx.full_url_signals(url)
    reasons = []
    delta = 0.0
    floor = 0.0  # near-certain signals impose a MINIMUM risk the ML can't override

    reg = _registered_domain(parts)
    trusted = reg.lower() in TRUSTED_DOMAINS

    # ---- brand impersonation (typosquat / homoglyph) ----
    # Checked first because it's the strongest single phishing signal: a domain
    # that looks like "paypal" but isn't the real paypal.com.
    look = None if trusted else _lookalike_brand(reg)
    if look:
        brand, kind = look
        if kind == "homoglyph":
            reasons.append({"severity": "high",
                            "text": f"Domain imitates '{brand}' using look-alike characters "
                                    f"(e.g. digits swapped for letters) — a classic spoofing trick."})
            delta += 0.55
            floor = max(floor, 82.0)
        else:
            reasons.append({"severity": "high",
                            "text": f"Domain is one character away from the well-known brand "
                                    f"'{brand}' — likely a typosquat impersonation."})
            delta += 0.45
            floor = max(floor, 72.0)

    # ---- high-severity structural red flags ----
    if parts["is_ip"]:
        reasons.append({"severity": "high",
                        "text": "Uses a raw IP address as the host instead of a domain name — a common phishing tactic."})
        delta += 0.40
        floor = max(floor, 85.0)
    if sig["has_at_symbol"]:
        reasons.append({"severity": "high",
                        "text": "Contains an '@' symbol, which can hide the real destination after the userinfo part."})
        delta += 0.30
    if sig["is_punycode"]:
        reasons.append({"severity": "high",
                        "text": "Host uses punycode (xn--), often used to imitate a real brand with look-alike characters."})
        delta += 0.30
        floor = max(floor, 80.0)
    if parts["scheme"] != "https":
        reasons.append({"severity": "high",
                        "text": "No HTTPS encryption — legitimate sites handling any login almost always use HTTPS."})
        delta += 0.20

    # ---- medium-severity signals ----
    if parts["tld"] in RISKY_TLDS:
        reasons.append({"severity": "medium",
                        "text": f"Top-level domain '.{parts['tld']}' is frequently abused for disposable phishing sites."})
        delta += 0.20
    if parts["subdomains"] >= 3:
        reasons.append({"severity": "medium",
                        "text": f"Has {parts['subdomains']} subdomains — attackers stack subdomains to look like a trusted brand."})
        delta += 0.15
    if sig["suspicious_keywords"]:
        kws = ", ".join(sig["suspicious_keywords"][:5])
        reasons.append({"severity": "medium",
                        "text": f"Contains lure keywords often seen in phishing: {kws}."})
        delta += min(0.05 * len(sig["suspicious_keywords"]), 0.20)
    if sig["obfuscated_chars"] >= 3:
        reasons.append({"severity": "medium",
                        "text": f"Contains {sig['obfuscated_chars']} percent-encoded characters, which can obscure the true address."})
        delta += 0.10

    # ---- low / informational ----
    host_digits = sum(c.isdigit() for c in parts["host"])
    if host_digits >= 4:
        reasons.append({"severity": "low",
                        "text": f"The domain contains {host_digits} digits, unusual for established brands."})
        delta += 0.05
    if "-" in parts["host"] and parts["host"].count("-") >= 2:
        reasons.append({"severity": "low",
                        "text": "Multiple hyphens in the domain — sometimes used to splice brand names (e.g. 'paypal-secure-login')."})
        delta += 0.05

    # ---- soft, model-aware signals (below the hard-rule thresholds) ----
    # These surface what the ML model actually weighed, so a borderline URL
    # gets a real explanation instead of a generic "looks fine" note.
    if feats and not trusted:
        tld = parts["tld"]
        tldp = feats.get("TLDLegitimateProb")
        if tldp is not None and tldp < 0.40 and tld not in RISKY_TLDS:
            reasons.append({"severity": "low",
                            "text": f"The '.{tld}' ending is less common among legitimate sites in the training data."})
            delta += 0.05
        ccr = feats.get("CharContinuationRateHost")
        if ccr is not None and ccr < 0.30:
            reasons.append({"severity": "low",
                            "text": "The domain name's letters don't flow like a normal word — it looks more random than a typical brand name."})
            delta += 0.05
        nd = int(feats.get("NoOfDigitsInHost", 0) or 0)
        if 2 <= nd <= 3:
            reasons.append({"severity": "low",
                            "text": f"The name contains {nd} digits, a little unusual for an established brand."})
            delta += 0.03
        dl = feats.get("DomainLength")
        if dl is not None and dl > 25:
            reasons.append({"severity": "low",
                            "text": f"The domain name is quite long ({int(dl)} characters); phishing domains tend to run longer than brands."})
            delta += 0.03
        if parts["subdomains"] == 2:
            reasons.append({"severity": "low",
                            "text": "Has 2 subdomains — worth a glance, though plenty of real sites do this too."})

    # ---- trust signals (pull risk DOWN, transparently) ----
    if trusted:
        reasons.append({"severity": "info",
                        "text": f"'{reg}' is on the trusted-domain allow-list of well-known sites."})
        delta -= 0.60

    # ---- positive confirmations (only when nothing was flagged) ----
    concerns = [r for r in reasons if r["severity"] in ("high", "medium", "low")]
    if not concerns:
        if parts["scheme"] == "https" and not parts["is_ip"]:
            reasons.append({"severity": "info", "text": "Uses a secure HTTPS connection."})
        tldp = feats.get("TLDLegitimateProb") if feats else None
        if tldp is not None and tldp >= 0.60:
            reasons.append({"severity": "info",
                            "text": f"Common domain ending ('.{parts['tld']}'), typical of legitimate sites."})
        if not trusted:
            reasons.append({"severity": "info",
                            "text": "Normal length and characters — no suspicious patterns found in the address."})

    if not reasons:
        reasons.append({"severity": "info",
                        "text": "No obvious structural red flags detected in the URL."})

    return reasons, max(-1.0, min(1.0, delta)), {"trusted": trusted, "registered_domain": reg,
                                                 "risk_floor": 0.0 if trusted else floor}
