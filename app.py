import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError, field_validator

USAGE_FILE = Path("usage.json")


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


class EmailMetrics(BaseModel):
    name: Optional[str] = None
    subject: Optional[str] = None
    started_at: Optional[str] = None
    total_sent: Optional[int] = None
    total_delivered: Optional[int] = None
    delivery_rate: Optional[float] = None
    unique_opens: Optional[int] = None
    open_rate: Optional[float] = None
    unique_clicks: Optional[int] = None
    unique_ctr: Optional[float] = None
    click_to_open: Optional[float] = None
    opt_outs: Optional[int] = None

    @field_validator("total_sent", "total_delivered", "unique_opens", "unique_clicks", "opt_outs", mode="before")
    @classmethod
    def parse_i(cls, v: Any) -> Optional[int]:
        return to_int(v)

    @field_validator("delivery_rate", "open_rate", "unique_ctr", "click_to_open", mode="before")
    @classmethod
    def parse_f(cls, v: Any) -> Optional[float]:
        return to_float(v)


class LandingMetrics(BaseModel):
    page_path: Optional[str] = None
    views: Optional[int] = None
    active_users: Optional[int] = None
    views_per_user: Optional[float] = None
    avg_engagement_seconds: Optional[float] = None
    event_count: Optional[int] = None
    jp_views: Optional[int] = None
    en_views: Optional[int] = None


class Registrant(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    score: Optional[float] = None
    last_submitted: Optional[str] = None
    last_activity: Optional[str] = None


class RegistrantList(BaseModel):
    registrants: List[Registrant] = Field(default_factory=list)


class ThemeItem(BaseModel):
    theme: str
    count: int
    example_quotes: List[str] = Field(default_factory=list)


class ValueRatingStats(BaseModel):
    avg: Optional[float] = None
    distribution_counts: Dict[str, int] = Field(default_factory=dict)


class ConsultLead(BaseModel):
    full_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None


class SurveyResponseRow(BaseModel):
    row_index: int
    responses: Dict[str, Optional[str]] = Field(default_factory=dict)


class SurveyDerived(BaseModel):
    n_responses: Optional[int] = None
    job_function_counts: Dict[str, int] = Field(default_factory=dict)
    job_level_counts: Dict[str, int] = Field(default_factory=dict)
    industry_counts: Dict[str, int] = Field(default_factory=dict)
    value_rating_stats: ValueRatingStats = Field(default_factory=ValueRatingStats)
    consult_yes_count: Optional[int] = None
    consult_no_count: Optional[int] = None
    consult_yes_leads: List[ConsultLead] = Field(default_factory=list)
    free_text_summaries: Dict[str, List[str]] = Field(default_factory=dict)
    top_themes: List[ThemeItem] = Field(default_factory=list)


class QualSummary(BaseModel):
    q10: List[str] = Field(default_factory=list)
    q11: List[str] = Field(default_factory=list)
    q12: List[str] = Field(default_factory=list)
    q14: List[str] = Field(default_factory=list)
    top_themes: List[ThemeItem] = Field(default_factory=list)


class ExecSummaryText(BaseModel):
    summary: str


def init_state() -> None:
    d = {
        "parsed_emails_df": pd.DataFrame(),
        "landing_metrics_dict": {},
        "registrants_df": pd.DataFrame(),
        "survey_derived": {},
        "survey_tables": {},
        "exec_summary_text": "",
        "times_used": 0,
        "usage_persist_ok": True,
        "usage_loaded": False,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_usage() -> None:
    try:
        if USAGE_FILE.exists():
            st.session_state["times_used"] = int(json.loads(USAGE_FILE.read_text(encoding="utf-8")).get("times_used", 0))
        st.session_state["usage_persist_ok"] = True
    except Exception:
        st.session_state["usage_persist_ok"] = False


def save_usage() -> None:
    try:
        USAGE_FILE.write_text(json.dumps({"times_used": int(st.session_state["times_used"])}, indent=2), encoding="utf-8")
        st.session_state["usage_persist_ok"] = True
    except Exception:
        st.session_state["usage_persist_ok"] = False


def usage_success() -> None:
    st.session_state["times_used"] = int(st.session_state.get("times_used", 0)) + 1
    if st.session_state.get("usage_persist_ok", True):
        save_usage()


def get_key(override: str) -> str:
    return (override or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


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


def _api_error_message(err: Exception, attempt: int, retries: int) -> str:
    prefix = f"(attempt {attempt + 1}/{retries + 1})"
    if isinstance(err, APITimeoutError):
        return f"{prefix} API timeout: request exceeded timeout window."
    if isinstance(err, RateLimitError):
        return f"{prefix} Rate limited by API (429)."
    if isinstance(err, APIConnectionError):
        return f"{prefix} API connection failed. Check network access."
    if isinstance(err, APIStatusError):
        return f"{prefix} API status error {getattr(err, 'status_code', 'unknown')}."
    if isinstance(err, ValidationError):
        return f"{prefix} Schema validation error: {err}"
    if isinstance(err, json.JSONDecodeError):
        return f"{prefix} Invalid JSON returned by model: {err}"
    return f"{prefix} Unexpected error: {err}"


def api_structured(api_key: str, model: str, temp: float, schema_model: type[BaseModel], sys: str, usr: str):
    client = OpenAI(api_key=api_key, timeout=45.0)
    schema = schema_model.model_json_schema()
    raw = None
    retries = 2
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.responses.create(
                model=model,
                temperature=temp,
                input=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                text={"format": {"type": "json_schema", "name": schema_model.__name__, "schema": schema, "strict": True}},
            )
            raw = getattr(r, "output_text", None) or ""
            if not raw:
                for o in getattr(r, "output", []) or []:
                    for c in getattr(o, "content", []) or []:
                        if hasattr(c, "text"):
                            raw += c.text
            parsed = schema_model.model_validate(json.loads(raw))
            return parsed, raw, None
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as e:
            last_err = _api_error_message(e, attempt, retries)
            status_code = getattr(e, "status_code", None)
            retryable_status = status_code is None or int(status_code) >= 500 or int(status_code) == 429
            if attempt < retries and retryable_status:
                time.sleep(1.2 * (attempt + 1))
                continue
            break
        except (json.JSONDecodeError, ValidationError) as e:
            return None, raw, _api_error_message(e, attempt, retries)
        except Exception as e:
            return None, raw, _api_error_message(e, attempt, retries)
    return None, raw, last_err or "Unknown API failure"


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


def parse_yes_no(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    if any(x in s for x in ["yes", "はい", "希望", "true", "1"]):
        return True
    if any(x in s for x in ["no", "いいえ", "不要", "false", "0"]):
        return False
    return None


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

    def vc(col: Optional[str]) -> Dict[str, int]:
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
    p, raw, err = api_structured(api_key, model, temp, QualSummary, "Summarize bullets and top themes. Keep snippets short.", json.dumps(payload, ensure_ascii=False))
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
    p, raw, err = api_structured(api_key, model, temp, ExecSummaryText, "Write concise 1-2 paragraph management-ready summary. Omit missing sections.", json.dumps(payload, ensure_ascii=False))
    return (p.summary.strip(), "") if p else ("", f"{err}\nRaw:\n{raw}")


st.set_page_config(page_title="Marketing Analytics Dashboard MVP", layout="wide")
init_state()
if not st.session_state["usage_loaded"]:
    load_usage()
    st.session_state["usage_loaded"] = True

st.title("Marketing Analytics Dashboard MVP")
with st.sidebar:
    t = int(st.session_state["times_used"])
    h, m = divmod(t * 30, 60)
    st.metric("Times Used", f"{t}")
    st.metric("Time Saved", f"{h}h {m}m")
    st.metric("Money Saved", f"${t * 15:,.0f}")
    st.divider()
    st.subheader("LLM Settings")
    key_override = st.text_input("OpenAI API Key (override)", type="password", placeholder="sk-...")
    api_key = get_key(key_override)
    st.caption(f"Status: {'API key set ✅' if api_key else 'missing ❌'}")
    model = st.selectbox("Model", ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"], index=0)
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    if st.button("Clear Data", use_container_width=True):
        st.session_state["parsed_emails_df"] = pd.DataFrame()
        st.session_state["landing_metrics_dict"] = {}
        st.session_state["registrants_df"] = pd.DataFrame()
        st.session_state["survey_derived"] = {}
        st.session_state["survey_tables"] = {}
        st.session_state["exec_summary_text"] = ""
        st.success("Parsed dashboard data cleared.")

tabs = st.tabs(["Emails", "Landing Page", "Registrants + Attendees", "Survey (MS Forms)", "Executive Summary"])
t1, t2, t3, t4, t5 = tabs

with t1:
    txt = st.text_area("Paste email report text", height=220, placeholder="Paste Pardot Click-Through Rate Report text here...")
    with st.expander("Example paste format"):
        st.code("Click-Through Rate Report\nName: Email A\nTotal Delivered: 500\nOpen Rate: 35.2%\nUnique CTR: 7.1%")
    if st.button("Parse Emails"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        elif not txt.strip():
            st.warning("Please paste email report text.")
        else:
            df, dbg, ok = parse_emails(txt, api_key, model, temp)
            if ok:
                st.session_state["parsed_emails_df"] = df
                usage_success()
                st.success("Email reports parsed successfully.")
            else:
                st.error("Email parsing failed.")
                for d in dbg:
                    with st.expander("Model output/debug"):
                        st.text(d)
    df = st.session_state["parsed_emails_df"]
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Open Rate", f"{pd.to_numeric(df.get('open_rate'), errors='coerce').mean():.2f}%")
        c2.metric("Avg Unique CTR", f"{pd.to_numeric(df.get('unique_ctr'), errors='coerce').mean():.2f}%")
        c3.metric("Total Unique Opens", f"{int(pd.to_numeric(df.get('unique_opens'), errors='coerce').fillna(0).sum()):,}")
        c4.metric("Total Unique Clicks", f"{int(pd.to_numeric(df.get('unique_clicks'), errors='coerce').fillna(0).sum()):,}")
        labels = df.get("name", pd.Series([f"Email {i+1}" for i in range(len(df))])).fillna("")
        labels = [x if str(x).strip() else f"Email {i+1}" for i, x in enumerate(labels)]
        a, b = st.columns(2)
        with a:
            fig, ax = plt.subplots()
            ax.bar(labels, pd.to_numeric(df.get("open_rate"), errors="coerce").fillna(0))
            ax.set_title("Open Rate by Email")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        with b:
            fig, ax = plt.subplots()
            ax.bar(labels, pd.to_numeric(df.get("unique_ctr"), errors="coerce").fillna(0))
            ax.set_title("Unique CTR by Email")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)
        f1, f2, f3 = st.columns(3)
        f1.metric("Delivered", f"{int(pd.to_numeric(df.get('total_delivered'), errors='coerce').fillna(0).sum()):,}")
        f2.metric("Opens", f"{int(pd.to_numeric(df.get('unique_opens'), errors='coerce').fillna(0).sum()):,}")
        f3.metric("Unique Clicks", f"{int(pd.to_numeric(df.get('unique_clicks'), errors='coerce').fillna(0).sum()):,}")

with t2:
    txt = st.text_area("Paste landing page analytics text", height=220, placeholder="Paste GA4 landing page metrics here...")
    with st.expander("Example paste format"):
        st.code("Page path: /x\nViews: 1000\nActive users: 700\nViews per user: 1.4\nAvg engagement seconds: 52\nJP views: 600\nEN views: 400")
    if st.button("Parse Landing Page"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        elif not txt.strip():
            st.warning("Please paste landing page analytics text.")
        else:
            d, dbg, ok = parse_landing(txt, api_key, model, temp)
            if ok:
                st.session_state["landing_metrics_dict"] = d
                usage_success()
                st.success("Landing page metrics parsed successfully.")
            else:
                st.error("Landing parse failed.")
                with st.expander("Model output/debug"):
                    st.text(dbg)
    d = st.session_state["landing_metrics_dict"]
    if d:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Views", f"{int(to_int(d.get('views')) or 0):,}")
        c2.metric("Active Users", f"{int(to_int(d.get('active_users')) or 0):,}")
        c3.metric("Views/User", f"{(to_float(d.get('views_per_user')) or 0):.2f}")
        c4.metric("Avg Engagement (sec)", f"{(to_float(d.get('avg_engagement_seconds')) or 0):.2f}")
        jp, en = to_int(d.get("jp_views")) or 0, to_int(d.get("en_views")) or 0
        if jp + en > 0:
            fig, ax = plt.subplots()
            ax.pie([jp, en], labels=["JP", "EN"], autopct="%1.1f%%")
            ax.set_title("JP vs EN Views")
            st.pyplot(fig)

with t3:
    txt = st.text_area("Paste registrant/attendee text", height=220, placeholder="Paste registrants/attendees export text here...")
    with st.expander("Example paste format"):
        st.code("Name: A | Company: X | Score: 10 | Last Submitted: 2025-11-01\nName: B | Company: Y | Score: 8 | Last Submitted: 2025-11-02")
    if st.button("Parse Registrants + Attendees"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        elif not txt.strip():
            st.warning("Please paste registrant/attendee text.")
        else:
            df, dbg, ok = parse_regs(txt, api_key, model, temp)
            if ok:
                st.session_state["registrants_df"] = df
                usage_success()
                st.success("Registrant data parsed successfully.")
            else:
                st.error("Registrant parse failed.")
                with st.expander("Model output/debug"):
                    st.text(dbg)
    df = st.session_state["registrants_df"]
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        if "last_submitted" in df.columns:
            x = df.copy()
            x["d"] = pd.to_datetime(x["last_submitted"], errors="coerce").dt.date
            by = x.dropna(subset=["d"]).groupby("d").size().reset_index(name="count").sort_values("d")
            if not by.empty:
                fig, ax = plt.subplots()
                ax.plot(by["d"].astype(str), by["count"], marker="o")
                ax.set_title("Registrations Over Time")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
        if "company" in df.columns:
            top = df["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna().value_counts().head(10)
            if not top.empty:
                fig, ax = plt.subplots()
                ax.bar(top.index, top.values)
                ax.set_title("Top Companies by Registrant Count")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)

with t4:
    up = st.file_uploader("Upload MS Forms CSV", type=["csv"])
    txt = st.text_area("Or paste survey text", height=180, placeholder="Paste survey text here if CSV is not available...")
    with st.expander("Example paste format"):
        st.code("Total responses: 42\nQ9 Avg=4.3/5\nQ13 Consultation Yes=9 No=33\nQ10: ...\nQ11: ...")
    if st.button("Parse Survey"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        elif up is not None:
            try:
                try:
                    sdf = pd.read_csv(up)
                except Exception:
                    up.seek(0)
                    sdf = pd.read_csv(up, encoding="utf-8-sig")
                d, tables, dbg, ok = parse_survey_csv(sdf, api_key, model, temp)
                if ok:
                    st.session_state["survey_derived"] = d
                    st.session_state["survey_tables"] = tables
                    usage_success()
                    st.success("Survey CSV parsed successfully.")
                else:
                    st.error("Survey parse failed.")
                    with st.expander("Model output/debug"):
                        st.text(dbg)
            except Exception as e:
                st.error(f"CSV read failed: {e}")
        elif txt.strip():
            d, dbg, ok = parse_survey_text(txt, api_key, model, temp)
            if ok:
                st.session_state["survey_derived"] = d
                st.session_state["survey_tables"] = {}
                usage_success()
                st.success("Survey text parsed successfully.")
            else:
                st.error("Survey parse failed.")
                with st.expander("Model output/debug"):
                    st.text(dbg)
        else:
            st.warning("Upload CSV or paste survey text.")
    d = st.session_state["survey_derived"]
    if d:
        n = int(to_int(d.get("n_responses")) or 0)
        avg = to_float((d.get("value_rating_stats") or {}).get("avg"))
        y = int(to_int(d.get("consult_yes_count")) or 0)
        n0 = int(to_int(d.get("consult_no_count")) or 0)
        rate = (100 * y / (y + n0)) if (y + n0) > 0 else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Responses", f"{n:,}")
        c2.metric("Avg Value Rating (Q9)", f"{avg:.2f}" if avg is not None else "N/A")
        c3.metric("Consultation Yes Rate", f"{rate:.1f}%" if rate is not None else "N/A")
        for title, key, xname in [
            ("Value Rating Distribution (Q9)", "distribution_counts", "rating"),
            ("Job Function Distribution (Q6)", "job_function_counts", "job_function"),
            ("Job Level Distribution (Q7)", "job_level_counts", "job_level"),
            ("Industry Distribution (Q8) Top 10 + Other", "industry_counts", "industry"),
        ]:
            src = (d.get("value_rating_stats") or {}).get(key, {}) if key == "distribution_counts" else d.get(key, {})
            if src:
                z = pd.DataFrame({xname: list(src.keys()), "count": list(src.values())})
                fig, ax = plt.subplots()
                ax.bar(z[xname].astype(str), z["count"])
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig)
        st.markdown("### Qualitative Insights")
        fs = d.get("free_text_summaries", {})
        for q in ["Q10", "Q11", "Q12", "Q14"]:
            with st.expander(f"{q} Summary"):
                for b in fs.get(q, []):
                    st.write(f"- {b}")
                if not fs.get(q):
                    st.write("No summary available.")
        themes = d.get("top_themes", [])
        if themes:
            tdf = pd.DataFrame(themes)
            if "example_quotes" in tdf.columns:
                tdf["example_quotes"] = tdf["example_quotes"].apply(lambda x: " | ".join([clip_snippet(v) for v in (x or [])]) if isinstance(x, list) else x)
            st.markdown("### Top Themes")
            st.dataframe(tdf, use_container_width=True)
        st.markdown("### Actionable Consultation Leads")
        leads = d.get("consult_yes_leads", [])
        if leads:
            ldf = pd.DataFrame(leads)
            st.dataframe(ldf, use_container_width=True)
            st.download_button("Download Consultation Leads CSV", ldf.to_csv(index=False).encode("utf-8-sig"), "consultation_leads.csv", "text/csv")
        else:
            st.info("No consultation-yes leads found.")

with t5:
    st.caption("Copy hint: click in the text area and copy.")
    if st.button("Generate Executive Summary"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        else:
            s, dbg = exec_summary(
                api_key,
                model,
                temp,
                st.session_state["parsed_emails_df"],
                st.session_state["landing_metrics_dict"],
                st.session_state["registrants_df"],
                st.session_state["survey_derived"],
            )
            if s:
                st.session_state["exec_summary_text"] = s
                st.success("Executive summary generated.")
            else:
                st.error("Executive summary generation failed.")
                with st.expander("Model output/debug"):
                    st.text(dbg)
    st.text_area("Executive Summary (copy-ready)", value=st.session_state.get("exec_summary_text", ""), height=220)
