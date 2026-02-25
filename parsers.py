import json
import re
from typing import Any, Dict

import pandas as pd

from models import ConsultLead, EmailMetrics, ExecSummaryText, LandingMetrics, QualSummary, RegistrantList, SurveyDerived, ThemeItem, ValueRatingStats
from openai_client import api_structured
from utils import clip_snippet, detect_col, parse_yes_no, to_float


def parse_emails(text: str, api_key: str, model: str, temp: float):
    blocks = [b.strip() for b in re.split(r"(?=Click-Through Rate Report)", text.strip(), flags=re.I) if b.strip()]
    if not blocks:
        blocks = [text.strip()]
    out, dbg, ok = [], [], True
    for i, b in enumerate(blocks, start=1):
        p, raw, err = api_structured(api_key, model, temp, EmailMetrics, "Extract email metrics. Missing -> null.", b)
        if p is None:
            ok = False
            dbg.append(f"Block {i}: {err}\nRaw:\n{raw}")
            continue
        d = p.model_dump()
        if not d.get("name"):
            d["name"] = f"Email {i}"
        out.append(d)
    return pd.DataFrame(out), dbg, ok and len(out) > 0


def parse_landing(text: str, api_key: str, model: str, temp: float):
    p, raw, err = api_structured(api_key, model, temp, LandingMetrics, "Extract landing metrics. Missing -> null.", text)
    return (p.model_dump(), "", True) if p else ({}, f"{err}\nRaw:\n{raw}", False)


def parse_regs(text: str, api_key: str, model: str, temp: float):
    p, raw, err = api_structured(api_key, model, temp, RegistrantList, "Extract registrants array.", text)
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


def exec_summary(api_key: str, model: str, temp: float, emails_df: pd.DataFrame, landing: Dict[str, Any], regs_df: pd.DataFrame, survey: Dict[str, Any]):
    payload: Dict[str, Any] = {}
    if not emails_df.empty:
        payload["emails"] = emails_df.fillna("").to_dict(orient="records")
    if landing:
        payload["landing"] = landing
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
