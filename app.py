import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from snapshot_store import (
    append_snapshot_row,
    build_snapshot_row,
    has_snapshot_data,
    load_snapshot_history,
    load_snapshot_into_state,
)
from ui.tabs import render_emails_tab, render_exec_summary_tab, render_landing_tab, render_regs_tab, render_social_tab, render_survey_tab
from ui.theme import apply_dashboard_style
from usage_store import init_state, load_usage, webinar_saved_success

load_dotenv()


def get_key(override: str) -> str:
    return (override or "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


def _automation_roi_from_history(history: pd.DataFrame) -> tuple[int, float, float]:
    if history.empty:
        return 0, 0.0, 0.0

    h = history.copy()
    webinars = int(h["webinar_id"].astype(str).nunique()) if "webinar_id" in h.columns else int(len(h))
    # Keep ROI aligned to webinar count for a simple, predictable baseline.
    hours_saved = float(webinars)
    money_saved = hours_saved * 45.0
    return webinars, hours_saved, money_saved


st.set_page_config(page_title="Webinar Marketing Analytics Dashboard", layout="wide")
apply_dashboard_style()
init_state()
if not st.session_state["usage_loaded"]:
    load_usage()
    st.session_state["usage_loaded"] = True
st.title("Webinar Marketing Analytics Dashboard")
ctrl_a, ctrl_b, _ = st.columns([1, 1, 6])
with ctrl_a:
    if st.button("Clear", use_container_width=True, type="primary"):
        st.session_state["parsed_emails_df"] = pd.DataFrame()
        st.session_state["landing_metrics_dict"] = {}
        st.session_state["social_metrics_dict"] = {}
        st.session_state["registrants_df"] = pd.DataFrame()
        st.session_state["survey_derived"] = {}
        st.session_state["survey_tables"] = {}
        st.session_state["exec_summary_text"] = ""
        st.success("All parsed data cleared. Ready for a new webinar.")
with ctrl_b:
    st.toggle("Hide Inputs", key="hide_all_inputs")

with st.sidebar:
    roi_hist = load_snapshot_history()
    webinars_saved, hours_saved, money_saved = _automation_roi_from_history(roi_hist)
    with st.container(border=True):
        st.markdown("**Automation ROI**")
        st.metric("Webinars Saved", f"{webinars_saved}")
        st.metric("Time Saved", f"{hours_saved:,.1f}h")
        st.metric("Money Saved", f"${money_saved:,.0f}")
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

    st.divider()
    st.subheader("Load Webinar Snapshot")
    history = load_snapshot_history()
    if history.empty:
        st.caption("No saved snapshots yet.")
    elif "webinar_id" not in history.columns:
        st.caption("Snapshot history format is missing webinar IDs.")
    else:
        pick = history[["webinar_id", "webinar_name", "saved_at_utc"]].copy()
        pick["webinar_name"] = pick["webinar_name"].fillna("").astype(str).str.strip()
        pick["saved_at_utc"] = pd.to_datetime(pick["saved_at_utc"], errors="coerce")
        pick["label"] = pick.apply(
            lambda r: (
                f"{r['saved_at_utc'].strftime('%Y-%m-%d %H:%M UTC') if pd.notna(r['saved_at_utc']) else 'Unknown date'}"
                f" | {r['webinar_name'] if r['webinar_name'] else '(Untitled webinar)'}"
            ),
            axis=1,
        )
        options = dict(zip(pick["label"], pick["webinar_id"]))
        selected_label = st.selectbox("Choose saved webinar", list(options.keys()), key="snapshot_to_load")
        if st.button("Load Selected Webinar", use_container_width=True):
            state = load_snapshot_into_state(options[selected_label])
            if not state:
                st.error("Could not load that snapshot.")
            else:
                for k, v in state.items():
                    st.session_state[k] = v
                st.success("Snapshot loaded into dashboard.")

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
