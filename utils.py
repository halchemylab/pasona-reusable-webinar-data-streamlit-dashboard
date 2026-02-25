import re
from typing import Any, Dict, List, Optional


def to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+", str(v).replace(",", ""))
    return int(m.group()) if m else None


def to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"-?\d+(\.\d+)?", str(v).replace(",", "").replace("%", ""))
    return float(m.group()) if m else None


def clip_snippet(text: str, n: int = 90) -> str:
    t = re.sub(r"\S+@\S+", "[redacted-email]", str(text or "")).strip()
    return t if len(t) <= n else t[: n - 3] + "..."


def norm(s: str) -> str:
    return re.sub(r"[\s_/:\-\(\)\[\]　]+", "", str(s).lower())


def detect_col(cols: List[str], kws: List[str]) -> Optional[str]:
    best, score = None, 0
    nk = [norm(k) for k in kws]
    for c in cols:
        nc = norm(c)
        s = sum(len(k) for k in nk if k in nc)
        if s > score:
            best, score = c, s
    return best if score > 0 else None


def parse_yes_no(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    if any(x in s for x in ["yes", "はい", "希望", "true", "1"]):
        return True
    if any(x in s for x in ["no", "いいえ", "不要", "false", "0"]):
        return False
    return None
