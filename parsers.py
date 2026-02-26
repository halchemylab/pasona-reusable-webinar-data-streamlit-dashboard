import json
import re
from typing import Any, Dict, Optional

import pandas as pd

from models import ConsultLead, EmailMetrics, ExecSummaryText, LandingMetrics, QualSummary, RegistrantList, SurveyDerived, ThemeItem, ValueRatingStats
from openai_client import api_structured
from utils import clip_snippet, detect_col, parse_yes_no, to_float, to_int


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
    return rows


def parse_regs(text: str, api_key: str, model: str, temp: float):
    rows = _extract_regs_rulebased(text)
    if len(rows) > 0:
        return pd.DataFrame(rows), "", True
    # Fallback to LLM parse for non-standard dumps; trim very large payloads for responsiveness.
    payload = (text or "")[:12000]
    p, raw, err = api_structured(api_key, model, temp, RegistrantList, "Extract registrants array.", payload)
    return (pd.DataFrame([x.model_dump() for x in p.registrants]), "", True) if p else (pd.DataFrame(), f"{err}\nRaw:\n{raw}", False)


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
        payload["emails"] = emails_df.fillna("").to_dict(orient="records")
    if landing:
        payload["landing"] = landing
    if social:
        payload["social_organic"] = social
    if not regs_df.empty:
        payload["registrants"] = regs_df.fillna("").to_dict(orient="records")
    if survey:
        payload["survey"] = survey
    if not payload:
        return "", "No parsed data available yet."
    p, raw, err = api_structured(
        api_key,
        model,
        temp,
        ExecSummaryText,
        "Write concise 1-2 paragraph management-ready summary. Omit missing sections.",
        json.dumps(payload, ensure_ascii=False),
    )
    return (p.summary.strip(), "") if p else ("", f"{err}\nRaw:\n{raw}")
