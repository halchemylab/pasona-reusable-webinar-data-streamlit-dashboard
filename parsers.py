import json
import re
import unicodedata
from io import StringIO
from typing import Any, Dict, Optional

import pandas as pd

from models import ConsultLead, EmailMetrics, ExecSummaryText, LandingMetrics, QualSummary, RegistrantList, SurveyDerived, ThemeItem, ValueRatingStats
from openai_client import api_structured
from utils import clip_snippet, detect_col, parse_yes_no, to_float, to_int


def _norm_ws(s: str) -> str:
    x = unicodedata.normalize("NFKC", str(s or ""))
    x = x.replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"\s+", " ", x).strip()


def _split_email_blocks(text: str):
    t = (text or "").strip()
    if not t:
        return []
    starts = [m.start() for m in re.finditer(r"Click-Through Rate Report", t, flags=re.I)]
    if not starts:
        return [t]
    blocks = []
    for i, s in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else len(t)
        page_end = re.search(r"Page\s+1\s+of\s+1", t[s:next_start], flags=re.I)
        e = s + page_end.end() if page_end else next_start
        b = t[s:e].strip()
        if b:
            blocks.append(b)
    return blocks


def _is_metrics_block(block: str) -> bool:
    keys = ["Total Sent", "Total Delivered", "Unique HTML Opens", "Unique Click Through Rate", "Delivery Rate"]
    hits = sum(1 for k in keys if re.search(re.escape(k), block, flags=re.I))
    return hits >= 2


def _find_group(text: str, pattern: str, flags: int = re.I) -> Optional[str]:
    m = re.search(pattern, text, flags=flags)
    return m.group(1).strip() if m else None


def _find_first_float(text: str, patterns) -> Optional[float]:
    for p in patterns:
        v = to_float(_find_group(text, p))
        if v is not None:
            return v
    return None


def _extract_email_metrics(block: str, idx: int) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "name": None,
        "subject": _find_group(block, r"Subject\s+(.+?)\s+Tracker\s+Domain"),
        "started_at": _find_group(block, r"Started\s+At\s+(.+?)\s+Created\s+At"),
        "total_sent": to_int(_find_group(block, r"Total\s+Sent\s+([\d,]+)")),
        "total_delivered": to_int(_find_group(block, r"Total\s+Delivered\s+([\d,]+)")),
        "delivery_rate": _find_first_float(
            block,
            [
                r"Total\s+Delivered\s+[\d,]+\s+Total\s+Failed\s+[\d,]+\s+Delivery\s+Rate\s+([\d.]+)%?",
                r"Delivery\s+Rate\s+([\d.]+)%?",
            ],
        ),
        "unique_opens": to_int(_find_group(block, r"Unique\s+HTML\s+Opens\s+([\d,]+)")),
        "open_rate": _find_first_float(
            block,
            [
                r"Unique\s+HTML\s+Opens\s+[\d,]+\s+HTML\s+Open\s+Rate\s+([\d.]+)%?",
                r"HTML\s+Open\s+Rate\s+([\d.]+)%?\s+Total\s+Clicks",
                r"HTML\s+Open\s+Rate\s+([\d.]+)%?",
            ],
        ),
        "unique_clicks": to_int(_find_group(block, r"Unique\s+Clicks\s+([\d,]+)")),
        "unique_ctr": _find_first_float(
            block,
            [
                r"Unique\s+Clicks\s+[\d,]+\s+Unique\s+Click\s+Through\s+Rate\s+([\d.]+)%?",
                r"Unique\s+Click\s+Through\s+Rate\s+([\d.]+)%?",
            ],
        ),
        "click_to_open": _find_first_float(
            block,
            [
                r"Unique\s+Click\s+Through\s+Rate\s+[\d.]+%?\s+Click\s+to\s+Open\s+Ratio\s+([\d.]+)%?",
                r"Click\s+to\s+Open\s+Ratio\s+([\d.]+)%?",
            ],
        ),
        "opt_outs": to_int(_find_group(block, r"Total\s+Opt\s+Outs\s+([\d,]+)")),
    }
    if not d.get("name"):
        d["name"] = f"Email {idx}"
    return d


def _extract_social_platform(text: str, platform: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if not t:
        return {}

    def fg(pattern: str):
        m = re.search(pattern, t, flags=re.I)
        return m.group(1).strip() if m else None

    out: Dict[str, Any] = {"platform": platform}
    if platform == "linkedin":
        out.update(
            {
                "impressions": to_int(fg(r"Organic\s+discovery\s+([\d,]+)") or fg(r"Impressions\s+([\d,]+)") or fg(r"([\d,]+)\s+Impressions")),
                "members_reached": to_int(fg(r"Members\s+reached\s+([\d,]+)") or fg(r"([\d,]+)\s+Members\s+reached")),
                "engagements": to_int(fg(r"Organic\s+engagement\s+([\d,]+)") or fg(r"Engagements\s+([\d,]+)\b")),
                "engagement_rate": to_float(fg(r"Engagement\s+rate\s+([\d.]+)%?") or fg(r"([\d.]+)%\s+Engagement\s+rate")),
                "clicks": to_int(fg(r"Clicks\s+([\d,]+)")),
                "ctr": to_float(fg(r"Click-through\s+rate\s+([\d.]+)%?") or fg(r"([\d.]+)%\s+Click-through\s+rate")),
                "reactions": to_int(fg(r"Reactions\s+([\d,]+)")),
                "comments": to_int(fg(r"Comments\s+([\d,]+)")),
                "reposts": to_int(fg(r"Reposts\s+([\d,]+)")),
                "page_viewers_from_post": to_int(fg(r"Page\s+viewers\s+from\s+this\s+post\s+([\d,]+)")),
                "followers_gained": to_int(fg(r"Followers\s+gained\s+from\s+this\s+post\s+([\d,]+)")),
            }
        )
    else:
        out.update(
            {
                "views": to_int(fg(r"Views\s+(\d[\d,]*)")),
                "viewers": to_int(fg(r"Viewers\s+(\d[\d,]*)")),
                "engagements": to_int(fg(r"(\d[\d,]*)\s+Engagement")),
                "link_clicks": to_int(fg(r"Link\s+clicks\s+(\d[\d,]*)")),
                "followers_views": to_int(fg(r"Followers\s+(\d[\d,]*)")),
                "non_followers_views": to_int(fg(r"Non-followers\s+(\d[\d,]*)")),
                "net_follows": to_int(fg(r"Net\s+follows\s+(\d[\d,]*)")),
            }
        )
    return {k: v for k, v in out.items() if v is not None or k == "platform"}


def parse_social(linkedin_text: str, facebook_text: str):
    li = _extract_social_platform(linkedin_text, "linkedin")
    fb = _extract_social_platform(facebook_text, "facebook")
    d = {"linkedin": li, "facebook": fb}
    ok = (len(li.keys()) > 1) or (len(fb.keys()) > 1)
    dbg = "" if ok else "No social metrics detected. Paste LinkedIn/Facebook post analytics text."
    return d, dbg, ok


def parse_emails(text: str, api_key: str, model: str, temp: float):
    blocks = _split_email_blocks(text)
    if not blocks:
        blocks = [text.strip()]
    out, dbg, ok = [], [], True
    for i, b in enumerate(blocks, start=1):
        if not _is_metrics_block(b):
            dbg.append(f"Block {i}: skipped non-metrics content block.")
            continue
        d = _extract_email_metrics(b, i)
        core = [
            d.get("total_sent"),
            d.get("total_delivered"),
            d.get("open_rate"),
            d.get("unique_clicks"),
            d.get("unique_ctr"),
        ]
        if sum(x is not None for x in core) >= 3:
            out.append(d)
            continue
        p, raw, err = api_structured(api_key, model, temp, EmailMetrics, "Extract email metrics. Missing -> null.", b)
        if p is None:
            ok = False
            dbg.append(f"Block {i}: {err}\nRaw:\n{raw}")
            continue
        out.append(p.model_dump())
    return pd.DataFrame(out), dbg, ok and len(out) > 0


def parse_landing(text: str, api_key: str, model: str, temp: float):
    t = (text or "").strip()
    if t:
        d = {
            "views": to_int(_find_group(t, r"Views\s+([\d,]+)")),
            "active_users": to_int(_find_group(t, r"Active\s+users\s+([\d,]+)")),
            "views_per_user": to_float(_find_group(t, r"Views\s+per\s+active\s+user\s+([\d.]+)")),
            "avg_engagement_seconds": to_float(_find_group(t, r"Average\s+engagement\s+time\s+per\s+active\s+user\s+([\d.]+)\s*s?")),
            "event_count": to_int(_find_group(t, r"Event\s+count(?:\s+\(All\s+events\))?\s+([\d,]+)")),
            "jp_views": to_int(_find_group(t, r"JP\s+views\s+([\d,]+)")),
            "en_views": to_int(_find_group(t, r"EN\s+views\s+([\d,]+)")),
        }
        if sum(v is not None for v in d.values()) >= 3:
            return d, "", True
    p, raw, err = api_structured(api_key, model, temp, LandingMetrics, "Extract landing metrics. Missing -> null.", text)
    return (p.model_dump(), "", True) if p else ({}, f"{err}\nRaw:\n{raw}", False)


def _extract_regs_rulebased(text: str):
    rows = []
    for ln in (text or "").splitlines():
        line = ln.strip()
        if not line:
            continue
        if ":" not in line:
            continue
        parts = [p.strip() for p in re.split(r"\s*\|\s*", line) if p.strip()]
        rec: Dict[str, Any] = {}
        for p in parts:
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            key = re.sub(r"\s+", " ", k.strip().lower())
            val = v.strip()
            if "name" in key and "company" not in key:
                rec["name"] = val or None
            elif "company" in key:
                rec["company"] = val or None
            elif "score" in key:
                rec["score"] = to_float(val)
            elif "last submitted" in key:
                rec["last_submitted"] = val or None
            elif "last activity" in key:
                rec["last_activity"] = val or None
        if rec:
            rows.append(rec)
    if rows:
        return rows

    sf_rows = _extract_regs_salesforce_list(text or "")
    if sf_rows:
        return sf_rows

    chunks = [c.strip() for c in re.split(r"(?i)(?=\bname\s*:)", text or "") if c.strip()]
    for chunk in chunks:
        rec: Dict[str, Any] = {}
        rec["name"] = _find_group(chunk, r"(?i)\bname\s*:\s*(.+?)(?=\s+\b(company|score|last submitted|last activity)\b\s*:|$)")
        rec["company"] = _find_group(chunk, r"(?i)\bcompany\s*:\s*(.+?)(?=\s+\b(name|score|last submitted|last activity)\b\s*:|$)")
        rec["score"] = to_float(_find_group(chunk, r"(?i)\bscore\s*:\s*([0-9]+(?:\.[0-9]+)?)"))
        rec["last_submitted"] = _find_group(chunk, r"(?i)\blast submitted\s*:\s*(.+?)(?=\s+\b(name|company|score|last activity)\b\s*:|$)")
        rec["last_activity"] = _find_group(chunk, r"(?i)\blast activity\s*:\s*(.+?)(?=\s+\b(name|company|score|last submitted)\b\s*:|$)")
        rec = {k: v for k, v in rec.items() if v not in [None, ""]}
        if rec:
            rows.append(rec)
    if rows:
        return rows

    raw = (text or "").strip()
    if "," in raw and ("name" in raw.lower() or "company" in raw.lower()):
        try:
            df = pd.read_csv(StringIO(raw))
            cols = list(df.columns)
            c_name = detect_col(cols, ["name"])
            c_company = detect_col(cols, ["company"])
            c_score = detect_col(cols, ["score"])
            c_last_submitted = detect_col(cols, ["last submitted"])
            c_last_activity = detect_col(cols, ["last activity"])
            for _, r in df.iterrows():
                rec = {
                    "name": str(r.get(c_name, "")).strip() or None if c_name else None,
                    "company": str(r.get(c_company, "")).strip() or None if c_company else None,
                    "score": to_float(r.get(c_score)) if c_score else None,
                    "last_submitted": str(r.get(c_last_submitted, "")).strip() or None if c_last_submitted else None,
                    "last_activity": str(r.get(c_last_activity, "")).strip() or None if c_last_activity else None,
                }
                rec = {k: v for k, v in rec.items() if v not in [None, ""]}
                if rec:
                    rows.append(rec)
        except Exception:
            pass
    return rows


def _extract_regs_salesforce_list(text: str):
    # Handles raw Salesforce/Pardot list dumps where each prospect is spread across lines:
    # Name(+View in CRM), Company, Score, Joined, Opted Out, Created.
    raw_lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = []
    for ln in raw_lines:
        if not ln:
            continue
        l = _norm_ws(ln)
        low = l.lower()
        if low in {
            "skip navigation",
            "salesforce pardot",
            "search salesforce",
            "search",
            "marketing",
            "prospects",
            "reports",
            "admin",
            "home",
            "segmentation",
            "lists",
            "prospects details",
            "list emails",
            "usage",
            "prospects",
            "actions",
            "tags tools",
            "tagstools",
            "filter:",
            "created",
            "all time",
            "name",
            "company",
            "score",
            "grade",
            "joined",
            "opted out of list",
            "with 0 selected:",
        }:
            continue
        if re.search(r"^page\s+\d+\s+of\s+\d+", low):
            continue
        if re.search(r"^showing\s+\d+\s+of\s+\d+", low):
            continue
        if re.search(r"^date range", low):
            continue
        if re.search(r"^view:", low):
            continue
        if re.search(r"^total prospects|^mailable prospects|^mailable$", low):
            continue
        if low in {"next»", "split copy", "edit"}:
            continue
        if re.fullmatch(r"view in crm", low):
            continue
        lines.append(l)

    dt_re = re.compile(r"^[A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]\.?M\.?$", re.I)

    def clean_name(v: str) -> str:
        s = re.sub(r"\bview in crm\b", "", v, flags=re.I)
        s = re.sub(r"\boperational emails only\b", "", s, flags=re.I)
        return re.sub(r"\s+", " ", s).strip(" '\"")

    def likely_company(v: str) -> bool:
        s = v.lower()
        keys = ["inc", "llc", "ltd", "corporation", "corp", "company", "co.", "university", "bank", "pharmaceutical", "america", "usa", "services", "consulting", "translation"]
        return any(k in s for k in keys)

    def likely_person(v: str) -> bool:
        s = re.sub(r"[^A-Za-z\.\-\s'()]", "", v).strip()
        if not s:
            return False
        if "," in s:
            return False
        parts = [p for p in s.split() if p]
        return 1 <= len(parts) <= 5

    out = []
    used = set()
    for i in range(len(lines) - 3):
        if i in used:
            continue
        score_line, joined_line, opt_line, created_line = lines[i], lines[i + 1], lines[i + 2], lines[i + 3]
        if not re.fullmatch(r"\d{1,4}", score_line):
            continue
        if not dt_re.match(joined_line):
            continue
        if opt_line.lower() not in {"yes", "no"}:
            continue
        if not dt_re.match(created_line):
            continue

        prev1 = _norm_ws(lines[i - 1]) if i - 1 >= 0 else ""
        prev2 = _norm_ws(lines[i - 2]) if i - 2 >= 0 else ""
        prev1_clean = clean_name(prev1)
        prev2_clean = clean_name(prev2)

        name = None
        company = None
        if prev2 and prev1 and (likely_company(prev1_clean) or likely_person(prev2_clean)):
            name = prev2_clean
            company = prev1_clean
        else:
            name = prev1_clean if prev1_clean else prev2_clean
            company = None
            if prev2 and prev2_clean and likely_company(prev2_clean):
                company = prev2_clean

        if name:
            out.append(
                {
                    "name": name,
                    "company": company or None,
                    "score": to_float(score_line),
                    "last_submitted": joined_line,
                    "last_activity": created_line,
                }
            )
            used.update({i, i + 1, i + 2, i + 3})
    return out


def _extract_regs_salesforce_regex(text: str):
    t = unicodedata.normalize("NFKC", (text or "")).replace("\u00a0", " ").replace("\u202f", " ").replace("\r\n", "\n")
    dt = r"[A-Za-z]{3,9}\.?\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}\s*[AP]\.?M\.?"
    pat = re.compile(
        rf"(?P<name>[^\n]+?)\s*(?:View in CRM)?\s*\n+\s*(?P<company>[^\n]+?)\s*\n+\s*(?P<score>\d{{1,4}})\s*\n+\s*(?P<joined>{dt})\s*\n+\s*(?:Yes|No)\s*\n+\s*(?P<created>{dt})",
        re.I,
    )
    out = []
    for m in pat.finditer(t):
        name = re.sub(r"\bOperational Emails Only\b", "", _norm_ws(m.group("name")), flags=re.I).strip(" '\"")
        company = _norm_ws(m.group("company")).strip(" '\"")
        if not name:
            continue
        if any(x in name.lower() for x in ["tagstools", "actions", "all prospects", "filter", "view:"]):
            continue
        out.append(
            {
                "name": name,
                "company": company or None,
                "score": to_float(m.group("score")),
                "last_submitted": m.group("joined"),
                "last_activity": m.group("created"),
            }
        )
    return out


def _list_type_from_name(list_name: str) -> str:
    return "attendee" if re.search(r"attendee", str(list_name or ""), flags=re.I) else "registrant"


def _extract_list_blocks(text: str):
    lines = (text or "").splitlines()
    starts = []
    for i, raw in enumerate(lines):
        ln = _norm_ws(raw)
        if re.search(r"^webinar_", ln, flags=re.I):
            starts.append((i, ln))
    if not starts:
        return []
    blocks = []
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        block_text = "\n".join(lines[start:end]).strip()
        if block_text:
            blocks.append((name, block_text))
    return blocks


def _extract_regs_salesforce_regex_with_context(text: str, list_name: Optional[str] = None, list_type: Optional[str] = None):
    out = _extract_regs_salesforce_regex(text)
    for r in out:
        if list_name:
            r["list_name"] = list_name
        if list_type:
            r["list_type"] = list_type
    return out


def _extract_regs_list_summaries(text: str):
    lines = [_norm_ws(ln) for ln in (text or "").splitlines()]
    lines = [ln for ln in lines if ln]

    def find_number_before_label(start: int, end: int, label_pat: str):
        for j in range(start, min(end, len(lines))):
            if re.search(label_pat, lines[j], flags=re.I):
                if j - 1 >= start:
                    n = to_int(lines[j - 1])
                    if n is not None:
                        return n
                n2 = to_int(lines[j])
                if n2 is not None:
                    return n2
        return None

    def find_percent_mailable(start: int, end: int):
        for j in range(start, min(end, len(lines))):
            if "%" in lines[j]:
                p0 = to_float(lines[j])
                if p0 is not None:
                    return p0
        for j in range(start, min(end, len(lines))):
            if re.search(r"mailable", lines[j], flags=re.I):
                if j - 1 >= start:
                    p = to_float(lines[j - 1])
                    if p is not None and "%" in lines[j - 1]:
                        return p
                p2 = to_float(lines[j])
                if p2 is not None and "%" in lines[j]:
                    return p2
        return None

    out = []
    for i, ln in enumerate(lines):
        if not re.search(r"^webinar_", ln, flags=re.I):
            continue
        w_start = i
        w_end = min(i + 30, len(lines))
        total = find_number_before_label(w_start, w_end, r"total\s+prospects")
        mailable = find_number_before_label(w_start, w_end, r"mailable\s+prospects")
        rate = find_percent_mailable(w_start, w_end)
        if total is None and mailable is None and rate is None:
            continue
        out.append(
            {
                "list_name": ln,
                "list_type": _list_type_from_name(ln),
                "total_prospects": total,
                "mailable_prospects": mailable,
                "mailable_rate": rate,
            }
        )

    if not out:
        return []
    df = pd.DataFrame(out).drop_duplicates(subset=["list_name"], keep="first")
    return df.to_dict(orient="records")


def parse_regs(text: str, api_key: str, model: str, temp: float):
    rows = []
    blocks = _extract_list_blocks(text)
    if blocks:
        for list_name, block_text in blocks:
            rows += _extract_regs_salesforce_regex_with_context(block_text, list_name=list_name, list_type=_list_type_from_name(list_name))
        if not rows:
            rows += _extract_regs_rulebased(text)
    else:
        rows += _extract_regs_salesforce_regex(text)
        rows += _extract_regs_rulebased(text)
    if rows:
        df = pd.DataFrame(rows)
        for col in ["name", "company", "score", "last_submitted", "last_activity", "list_name", "list_type"]:
            if col not in df.columns:
                df[col] = None
        for c in ["name", "company", "last_submitted", "last_activity"]:
            df[c] = df[c].apply(lambda v: _norm_ws(v) if isinstance(v, str) else v)
        df["list_type"] = df["list_type"].apply(lambda v: _norm_ws(v).lower() if isinstance(v, str) else v)
        if "list_name" in df.columns:
            df.loc[df["list_type"].isna() & df["list_name"].notna(), "list_type"] = df.loc[df["list_type"].isna() & df["list_name"].notna(), "list_name"].apply(_list_type_from_name)
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df = df.dropna(subset=["name"])
        df = df[~df["name"].str.lower().str.contains(r"tagstools|actions|all prospects|filter|view:", na=False)]
        df = df.drop_duplicates(subset=["name", "company", "last_submitted", "list_name"], keep="first")
        cols = ["name", "company", "score", "last_submitted", "last_activity"]
        if df["list_name"].notna().any():
            cols.append("list_name")
        if df["list_type"].notna().any():
            cols.append("list_type")
        return df[cols], "", True
    list_rows = _extract_regs_list_summaries(text)
    if list_rows:
        df = pd.DataFrame(list_rows)
        cols = ["list_name", "total_prospects", "mailable_prospects", "mailable_rate"]
        if "list_type" in df.columns:
            cols.insert(1, "list_type")
        return df[cols], "", True
    return pd.DataFrame(), "Could not detect registrant rows from this raw export. Try including the table section with Name/Company/Score/Joined/Created.", False


def parse_survey_text(text: str, api_key: str, model: str, temp: float):
    p, raw, err = api_structured(api_key, model, temp, SurveyDerived, "Infer survey derived metrics. Missing -> null/empty.", text)
    if not p:
        return {}, f"{err}\nRaw:\n{raw}", False
    d = p.model_dump()
    for t in d.get("top_themes", []):
        t["example_quotes"] = [clip_snippet(x) for x in t.get("example_quotes", [])]
    return d, "", True


def parse_survey_csv(df: pd.DataFrame, api_key: str, model: str, temp: float):
    cols = list(df.columns)
    c = {
        "first": detect_col(cols, ["first name", "名"]),
        "last": detect_col(cols, ["last name", "姓"]),
        "company": detect_col(cols, ["company name", "貴社名", "会社名"]),
        "title": detect_col(cols, ["title", "役職名"]),
        "email": detect_col(cols, ["email", "メール"]),
        "q6": detect_col(cols, ["primary job function", "職種", "q6"]),
        "q7": detect_col(cols, ["job level", "役職レベル", "q7"]),
        "q8": detect_col(cols, ["industry", "業界", "q8"]),
        "q9": detect_col(cols, ["how valuable", "有益", "q9"]),
        "q10": detect_col(cols, ["valuable topic", "有益", "q10"]),
        "q11": detect_col(cols, ["challenges", "課題", "q11"]),
        "q12": detect_col(cols, ["hear more", "もう少し", "q12"]),
        "q13": detect_col(cols, ["one-on-one consultation", "個別相談", "q13"]),
        "q14": detect_col(cols, ["comments", "ご意見", "q14"]),
    }

    def vc(col):
        if not col:
            return {}
        s = df[col].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA}).dropna()
        return s.value_counts().to_dict()

    q6, q7, q8 = vc(c["q6"]), vc(c["q7"]), vc(c["q8"])
    if q8:
        it = sorted(q8.items(), key=lambda x: x[1], reverse=True)
        q8 = dict(it[:10] + ([("Other", sum(v for _, v in it[10:]))] if len(it) > 10 else []))

    stats = ValueRatingStats(avg=None, distribution_counts={})
    if c["q9"]:
        s = df[c["q9"]].apply(to_float).dropna()
        if len(s) > 0:
            stats.avg = float(s.mean())
            stats.distribution_counts = s.round(0).astype(int).value_counts().sort_index().astype(int).to_dict()

    yes, no, leads = 0, 0, []
    if c["q13"]:
        yn = df[c["q13"]].apply(parse_yes_no)
        yes, no = int((yn == True).sum()), int((yn == False).sum())  # noqa: E712
        for _, r in df[yn == True].iterrows():  # noqa: E712
            full = " ".join([str(r.get(c["first"], "")).strip(), str(r.get(c["last"], "")).strip()]).strip() or None
            leads.append(
                ConsultLead(
                    full_name=full,
                    company=str(r.get(c["company"], "")).strip() or None,
                    title=str(r.get(c["title"], "")).strip() or None,
                    email=str(r.get(c["email"], "")).strip() or None,
                )
            )

    payload = {
        "q10": [clip_snippet(x, 220) for x in (df[c["q10"]].dropna().astype(str).tolist() if c["q10"] else [])][:250],
        "q11": [clip_snippet(x, 220) for x in (df[c["q11"]].dropna().astype(str).tolist() if c["q11"] else [])][:250],
        "q12": [clip_snippet(x, 220) for x in (df[c["q12"]].dropna().astype(str).tolist() if c["q12"] else [])][:250],
        "q14": [clip_snippet(x, 220) for x in (df[c["q14"]].dropna().astype(str).tolist() if c["q14"] else [])][:250],
    }
    p, raw, err = api_structured(
        api_key,
        model,
        temp,
        QualSummary,
        "Summarize bullets and top themes. Keep snippets short.",
        json.dumps(payload, ensure_ascii=False),
    )
    if not p:
        return {}, {}, f"{err}\nRaw:\n{raw}", False

    d = SurveyDerived(
        n_responses=len(df),
        job_function_counts=q6,
        job_level_counts=q7,
        industry_counts=q8,
        value_rating_stats=stats,
        consult_yes_count=yes,
        consult_no_count=no,
        consult_yes_leads=leads,
        free_text_summaries={"Q10": p.q10, "Q11": p.q11, "Q12": p.q12, "Q14": p.q14},
        top_themes=[ThemeItem(theme=t.theme, count=t.count, example_quotes=[clip_snippet(x) for x in t.example_quotes]) for t in p.top_themes],
    ).model_dump()

    return d, {}, "", True


def exec_summary(
    api_key: str,
    model: str,
    temp: float,
    emails_df: pd.DataFrame,
    landing: Dict[str, Any],
    social: Dict[str, Any],
    regs_df: pd.DataFrame,
    survey: Dict[str, Any],
):
    payload: Dict[str, Any] = {}
    if not emails_df.empty:
        sent = int(pd.to_numeric(emails_df.get("total_sent"), errors="coerce").fillna(0).sum())
        delivered = int(pd.to_numeric(emails_df.get("total_delivered"), errors="coerce").fillna(0).sum())
        opens = int(pd.to_numeric(emails_df.get("unique_opens"), errors="coerce").fillna(0).sum())
        clicks = int(pd.to_numeric(emails_df.get("unique_clicks"), errors="coerce").fillna(0).sum())
        payload["email_kpis"] = {
            "email_count": int(len(emails_df)),
            "sent": sent if sent > 0 else delivered,
            "delivered": delivered,
            "open": opens,
            "click": clicks,
            "avg_open_rate": to_float(pd.to_numeric(emails_df.get("open_rate"), errors="coerce").mean()),
            "avg_unique_ctr": to_float(pd.to_numeric(emails_df.get("unique_ctr"), errors="coerce").mean()),
        }
    if landing:
        payload["landing_kpis"] = {
            "views": to_int(landing.get("views")),
            "active_users": to_int(landing.get("active_users")),
            "views_per_user": to_float(landing.get("views_per_user")),
            "avg_engagement_seconds": to_float(landing.get("avg_engagement_seconds")),
            "jp_views": to_int(landing.get("jp_views")),
            "en_views": to_int(landing.get("en_views")),
        }
    if social:
        li = social.get("linkedin") or {}
        fb = social.get("facebook") or {}
        payload["social_kpis"] = {
            "linkedin_impressions": to_int(li.get("impressions")),
            "linkedin_engagements": to_int(li.get("engagements")),
            "linkedin_clicks": to_int(li.get("clicks")),
            "facebook_views": to_int(fb.get("views")),
            "facebook_engagements": to_int(fb.get("engagements")),
            "facebook_link_clicks": to_int(fb.get("link_clicks")),
        }
    if not regs_df.empty:
        reg_count = int(len(regs_df))
        if "name" in regs_df.columns:
            reg_count = int(regs_df["name"].fillna("").astype(str).str.strip().ne("").sum())
        company_count = int(regs_df["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna().nunique()) if "company" in regs_df.columns else 0
        attendance = None
        if "score" in regs_df.columns:
            scores = pd.to_numeric(regs_df["score"], errors="coerce")
            if len(scores.dropna()) > 0:
                attendance = int((scores.fillna(0) > 0).sum())
        payload["registrant_kpis"] = {
            "registration": reg_count,
            "attendance": attendance,
            "unique_companies": company_count,
        }
    if survey:
        top_themes = []
        for t in (survey.get("top_themes") or [])[:5]:
            top_themes.append({"theme": str(t.get("theme", "")), "count": to_int(t.get("count"))})
        payload["survey_kpis"] = {
            "responses": to_int(survey.get("n_responses")),
            "avg_value_rating": to_float((survey.get("value_rating_stats") or {}).get("avg")),
            "consultation_leads": to_int(survey.get("consult_yes_count")),
            "consultation_no": to_int(survey.get("consult_no_count")),
            "top_themes": top_themes,
        }
    if not payload:
        return "", "No parsed data available yet."
    p, raw, err = api_structured(
        api_key,
        model,
        temp,
        ExecSummaryText,
        "Write concise 1-2 paragraph management-ready summary from KPI aggregates only. Mention funnel progression when available. Omit missing sections.",
        json.dumps(payload, ensure_ascii=False),
    )
    return (p.summary.strip(), "") if p else ("", f"{err}\nRaw:\n{raw}")
