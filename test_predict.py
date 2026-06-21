"""Quick CLI sanity check, no server needed:  python test_predict.py <url>"""
import sys
from utils import predictor

urls = sys.argv[1:] or [
    "https://github.com/ayanmurad987",
    "http://paypal-secure-login.verify-account.tk/update",
    "http://192.168.1.1/login.php?account=verify",
    "https://www.amazon.com/gp/product/B08",
]
for u in urls:
    r = predictor.predict(u)
    print(f"\n{r['risk_score']:5.1f}%  {r['prediction'].upper():11s}  {u}")
    print(f"        ml={r['ml_risk_score']}  rule_adj={r['rule_adjustment']}  trusted={r['trusted_domain']}")
    for rs in r["reasons"]:
        print(f"        [{rs['severity']}] {rs['text']}")
