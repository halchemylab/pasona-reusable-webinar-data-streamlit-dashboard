import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from parsers import exec_summary, parse_emails, parse_landing, parse_regs, parse_survey_csv, parse_survey_text
from usage_store import init_state, load_usage, usage_success
from utils import clip_snippet, to_float, to_int


def get_key(override: str) -> str:
    return (override or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


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
