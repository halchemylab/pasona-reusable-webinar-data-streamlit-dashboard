import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from snapshot_store import append_snapshot_row, build_snapshot_row, has_snapshot_data
from ui.tabs import render_emails_tab, render_exec_summary_tab, render_landing_tab, render_regs_tab, render_social_tab, render_survey_tab
from ui.theme import apply_dashboard_style
from usage_store import init_state, load_usage, webinar_saved_success

load_dotenv()


def get_key(override: str) -> str:
    return (override or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


st.set_page_config(page_title="Webinar Marketing Analytics Dashboard", layout="wide")
apply_dashboard_style()
init_state()
if not st.session_state["usage_loaded"]:
    load_usage()
    st.session_state["usage_loaded"] = True

st.title("Webinar Marketing Analytics Dashboard")
with st.sidebar:
    t = int(st.session_state["webinars_saved"])
    with st.container(border=True):
        st.metric("Webinars Saved", f"{t}")
    with st.container(border=True):
        st.metric("Time Saved", f"{t}h")
    with st.container(border=True):
        st.metric("Money Saved", f"${t * 45:,.0f}")
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
        st.session_state["social_metrics_dict"] = {}
        st.session_state["registrants_df"] = pd.DataFrame()
        st.session_state["survey_derived"] = {}
        st.session_state["survey_tables"] = {}
        st.session_state["exec_summary_text"] = ""
        st.success("Parsed dashboard data cleared.")
    st.divider()
    st.subheader("Save Webinar Snapshot")
    webinar_name = st.text_input("Webinar Name", placeholder="2026-02-25 AI Webinar")
    if st.button("Save Webinar Snapshot", use_container_width=True):
        emails_df = st.session_state["parsed_emails_df"]
        landing = st.session_state["landing_metrics_dict"]
        social = st.session_state["social_metrics_dict"]
        regs_df = st.session_state["registrants_df"]
        survey = st.session_state["survey_derived"]
        summary = st.session_state.get("exec_summary_text", "")
        if not has_snapshot_data(emails_df, landing, social, regs_df, survey, summary):
            st.warning("Nothing to save yet. Parse at least one section first.")
        else:
            row = build_snapshot_row(webinar_name, emails_df, landing, social, regs_df, survey, summary)
            path = append_snapshot_row(row)
            webinar_saved_success()
            st.success(f"Saved snapshot to {path}")

tabs = st.tabs(["Emails", "Landing Page", "Social Media (Organic)", "Registrants + Attendees", "Survey (MS Forms)", "Executive Summary"])
t1, t2, t3, t4, t5, t6 = tabs

with t1:
    render_emails_tab(api_key, model, temp)
with t2:
    render_landing_tab(api_key, model, temp)
with t3:
    render_social_tab()
with t4:
    render_regs_tab(api_key, model, temp)
with t5:
    render_survey_tab(api_key, model, temp)
with t6:
    render_exec_summary_tab(api_key, model, temp)
