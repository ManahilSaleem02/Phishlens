"""
Shared URL feature extractor — single source of truth for both training and
serving (zero train/serve skew).

Design decisions driven by the PhiUSIIL dataset's properties:
  * Every feature is computable from the URL STRING alone. No live page fetch,
    no HTML parsing -> fast, safe (we never visit a malicious page), reproducible.
  * MODEL features are derived from the HOST / DOMAIN, not the path. PhiUSIIL's
    legitimate class contains only bare homepages (zero legit URLs have a path),
    so any path-content feature collapses to "has a path -> phishing", which is a
    sampling artifact, not real signal. Judging the host generalises correctly to
    legitimate deep links (github.com/x, amazon.com/gp/...).
  * Full-URL lure signals (suspicious keywords, '@', obfuscation, IP host,
    punycode) are still computed and handed to the rule-based explainer, which
    scans the WHOLE url. ML judges the domain; rules judge the rest.
"""

from __future__ import annotations

import math
import re
import ipaddress
from urllib.parse import urlparse, unquote

# Ordered features fed to the classifier. ORDER MATTERS.
FEATURE_ORDER = [
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "NoOfDotsInHost",
    "NoOfHyphensInHost",
    "IsHTTPS",
    "NoOfLettersInHost",
    "LetterRatioInHost",
    "NoOfDigitsInHost",
    "DigitRatioInHost",
    "NoOfSpecialCharsInHost",
    "SpecialCharRatioInHost",
    "HasObfuscationInHost",
    "CharContinuationRateHost",
    "TLDLegitimateProb",
    "HostCharProb",
]

SUSPICIOUS_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification",
    "secure", "account", "update", "confirm", "banking", "bank", "wallet",
    "password", "credential", "webscr", "ebayisapi", "paypal", "appleid",
    "apple", "office365", "microsoft", "amazon", "netflix", "support",
    "recover", "unlock", "billing", "invoice", "payment", "suspended",
    "alert", "limited", "authenticate", "security",
]

_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_HOST_SPECIALS = set("-_.~")


def _ensure_scheme(url: str) -> str:
    # A user-typed bare domain ("google.com") carries no scheme information.
    # Browsers default such input to HTTPS, so we do the same — defaulting to
    # http would unfairly penalize every scheme-less input (IsHTTPS is the
    # model's strongest feature and a "no HTTPS" rule would fire). All training
    # URLs carry explicit schemes, so this only affects user input.
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        return "https://" + url
    return url


def _is_ip(host: str) -> bool:
    host = host.split(":")[0]
    if _IP_RE.match(host):
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _char_continuation_rate(s: str) -> float:
    if not s:
        return 0.0

    def cls(c):
        return "a" if c.isalpha() else ("d" if c.isdigit() else "s")

    runs, cur = [], 1
    for i in range(1, len(s)):
        if cls(s[i]) == cls(s[i - 1]):
            cur += 1
        else:
            runs.append(cur); cur = 1
    runs.append(cur)
    return (sum(runs) / len(runs)) / len(s)


def _char_prob(s: str, char_logprob: dict | None) -> float:
    if not char_logprob or not s:
        return 0.5
    floor = char_logprob.get("__floor__", math.log(1e-6))
    total = sum(char_logprob.get(c, floor) for c in s.lower())
    return math.exp(total / len(s))


def parse_parts(url: str) -> dict:
    """Structural pieces used by features, rules and the explainer."""
    raw = url.strip()
    p = urlparse(_ensure_scheme(raw))
    userinfo = p.netloc.split("@")[0] if "@" in p.netloc else ""
    host_full = p.netloc.split("@")[-1].split(":")[0]
    # Normalise a leading "www." so example.com and www.example.com are treated
    # identically. PhiUSIIL legit URLs nearly all carry www, which otherwise
    # makes the model flag bare modern domains (github.com, openai.com).
    host = host_full[4:] if host_full.lower().startswith("www.") else host_full
    labels = host.split(".") if host else []
    tld = labels[-1] if len(labels) >= 2 else ""
    is_ip = _is_ip(host)
    subdomains = 0 if is_ip else max(0, len(labels) - 2)
    return {
        "raw": raw,
        "scheme": p.scheme.lower(),
        "host": host,
        "userinfo": userinfo,
        "path": p.path or "",
        "query": p.query or "",
        "tld": tld.lower(),
        "is_ip": is_ip,
        "labels": labels,
        "subdomains": subdomains,
        "has_at": "@" in raw,
        "is_punycode": "xn--" in host.lower(),
    }


def extract(url: str, tld_prob: dict | None = None,
            char_logprob: dict | None = None) -> dict:
    """Build the model feature dict (host-focused) for one URL."""
    parts = parse_parts(url)
    host = parts["host"] or ""
    hl = len(host) or 1

    letters = sum(c.isalpha() for c in host)
    digits = sum(c.isdigit() for c in host)
    specials = sum(1 for c in host if c in _HOST_SPECIALS)

    return {
        "DomainLength": len(host),
        "IsDomainIP": int(parts["is_ip"]),
        "TLDLength": len(parts["tld"]),
        "NoOfSubDomain": parts["subdomains"],
        "NoOfDotsInHost": host.count("."),
        "NoOfHyphensInHost": host.count("-"),
        "IsHTTPS": int(parts["scheme"] == "https"),
        "NoOfLettersInHost": letters,
        "LetterRatioInHost": letters / hl,
        "NoOfDigitsInHost": digits,
        "DigitRatioInHost": digits / hl,
        "NoOfSpecialCharsInHost": specials,
        "SpecialCharRatioInHost": specials / hl,
        "HasObfuscationInHost": int("%" in host or parts["is_punycode"]),
        "CharContinuationRateHost": _char_continuation_rate(host),
        "TLDLegitimateProb": (tld_prob or {}).get(
            parts["tld"], (tld_prob or {}).get("__default__", 0.5)),
        "HostCharProb": _char_prob(host, char_logprob),
    }


def full_url_signals(url: str) -> dict:
    """Whole-URL lure signals for the rule-based explainer (NOT model inputs)."""
    parts = parse_parts(url)
    raw = parts["raw"]
    decoded = unquote(raw).lower()
    keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in decoded]
    return {
        "url_length": len(raw),
        "path_length": len(parts["path"]),
        "obfuscated_chars": raw.count("%"),
        "has_at_symbol": parts["has_at"],
        "userinfo": parts["userinfo"],
        "is_punycode": parts["is_punycode"],
        "suspicious_keywords": keywords,
        "num_query_params": raw.count("=") if parts["query"] else 0,
    }


def to_vector(feature_dict: dict) -> list:
    return [feature_dict[name] for name in FEATURE_ORDER]
