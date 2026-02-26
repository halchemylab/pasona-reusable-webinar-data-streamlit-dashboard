import json
from pathlib import Path

import pandas as pd
import streamlit as st

USAGE_FILE = Path("usage.json")


def init_state() -> None:
    d = {
        "parsed_emails_df": pd.DataFrame(),
        "landing_metrics_dict": {},
        "social_metrics_dict": {},
        "registrants_df": pd.DataFrame(),
        "survey_derived": {},
        "survey_tables": {},
        "exec_summary_text": "",
        "webinars_saved": 0,
        "usage_persist_ok": True,
        "usage_loaded": False,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_usage() -> None:
    try:
        if USAGE_FILE.exists():
            raw = json.loads(USAGE_FILE.read_text(encoding="utf-8"))
            st.session_state["webinars_saved"] = int(raw.get("webinars_saved", raw.get("times_used", 0)))
        st.session_state["usage_persist_ok"] = True
    except Exception:
        st.session_state["usage_persist_ok"] = False


def save_usage() -> None:
    try:
        USAGE_FILE.write_text(json.dumps({"webinars_saved": int(st.session_state["webinars_saved"])}, indent=2), encoding="utf-8")
        st.session_state["usage_persist_ok"] = True
    except Exception:
        st.session_state["usage_persist_ok"] = False


def webinar_saved_success() -> None:
    st.session_state["webinars_saved"] = int(st.session_state.get("webinars_saved", 0)) + 1
    if st.session_state.get("usage_persist_ok", True):
        save_usage()
