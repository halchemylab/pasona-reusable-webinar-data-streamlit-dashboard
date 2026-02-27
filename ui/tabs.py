import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from parsers import (
    _extract_regs_rulebased,
    _extract_regs_salesforce_list,
    _extract_regs_salesforce_regex,
    exec_summary,
    parse_emails,
    parse_landing,
    parse_regs,
    parse_social,
    parse_survey_csv,
    parse_survey_text,
)
from utils import clip_snippet, to_float, to_int


def _load_history() -> pd.DataFrame:
    p = Path("data/webinar_history.csv")
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def render_emails_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Email Performance")
    if not st.session_state.get("hide_all_inputs", False):
        txt = st.text_area(
            "Paste email report text",
            height=220,
            placeholder="Paste Pardot Click-Through Rate Report text here...",
            key="email_input_text",
        )
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
                    st.success("Email reports parsed successfully.")
                else:
                    st.error("Email parsing failed.")
                    for d in dbg:
                        with st.expander("Model output/debug"):
                            st.text(d)
    df = st.session_state["parsed_emails_df"]
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        dfx = df.copy()
        if "name" not in dfx.columns:
            dfx["name"] = [f"Email {i+1}" for i in range(len(dfx))]
        dfx["name"] = dfx["name"].fillna("").astype(str)
        dfx["name"] = [v if v.strip() else f"Email {i+1}" for i, v in enumerate(dfx["name"])]
        for c in ["open_rate", "unique_ctr", "total_delivered", "unique_opens", "unique_clicks", "total_sent"]:
            dfx[c] = pd.to_numeric(dfx.get(c), errors="coerce")

        avg_open = dfx["open_rate"].mean()
        avg_ctr = dfx["unique_ctr"].mean()
        opens = int(dfx["unique_opens"].fillna(0).sum())
        clicks = int(dfx["unique_clicks"].fillna(0).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Open Rate", f"{avg_open:.2f}%")
        c2.metric("Avg Unique CTR", f"{avg_ctr:.2f}%")
        c3.metric("Total Unique Opens", f"{opens:,}")
        c4.metric("Total Unique Clicks", f"{clicks:,}")

        rate_df = (
            dfx[["name", "open_rate", "unique_ctr"]]
            .melt(id_vars=["name"], var_name="metric", value_name="value")
            .dropna(subset=["value"])
        )
        if not rate_df.empty:
            rate_chart = (
                alt.Chart(rate_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("name:N", title="Email"),
                    y=alt.Y("value:Q", title="Rate (%)"),
                    color=alt.Color(
                        "metric:N",
                        scale=alt.Scale(domain=["open_rate", "unique_ctr"], range=["#2563eb", "#10b981"]),
                        legend=alt.Legend(title="Metric"),
                    ),
                    tooltip=["name:N", "metric:N", alt.Tooltip("value:Q", format=".2f")],
                )
                .properties(height=340, title="Open Rate vs Unique CTR by Email")
            )
            st.altair_chart(rate_chart, use_container_width=True)

        sent = int(dfx["total_sent"].fillna(0).sum()) if "total_sent" in dfx else 0
        delivered = int(dfx["total_delivered"].fillna(0).sum())
        funnel = pd.DataFrame(
            {
                "stage": ["Sent", "Delivered", "Unique Opens", "Unique Clicks"],
                "value": [sent if sent > 0 else delivered, delivered, opens, clicks],
            }
        )
        fchart = (
            alt.Chart(funnel)
            .mark_bar(size=36, cornerRadiusEnd=5)
            .encode(
                y=alt.Y("stage:N", sort=["Sent", "Delivered", "Unique Opens", "Unique Clicks"], title=""),
                x=alt.X("value:Q", title="Volume"),
                color=alt.Color("stage:N", scale=alt.Scale(range=["#334155", "#2563eb", "#0ea5e9", "#10b981"]), legend=None),
                tooltip=["stage:N", "value:Q"],
            )
            .properties(height=230, title="Delivery Funnel")
        )
        st.altair_chart(fchart, use_container_width=True)

        st.markdown("#### Data Science View: Open Rate vs Unique CTR Correlation")
        scatter = dfx.dropna(subset=["open_rate", "unique_ctr"])[["name", "open_rate", "unique_ctr"]]
        if len(scatter) >= 2:
            points = (
                alt.Chart(scatter)
                .mark_circle(size=100, color="#2563eb", opacity=0.85)
                .encode(
                    x=alt.X("open_rate:Q", title="Open Rate (%)"),
                    y=alt.Y("unique_ctr:Q", title="Unique CTR (%)"),
                    tooltip=["name:N", alt.Tooltip("open_rate:Q", format=".2f"), alt.Tooltip("unique_ctr:Q", format=".2f")],
                )
            )
            trend = points.transform_regression("open_rate", "unique_ctr").mark_line(color="#ef4444", size=3)
            st.altair_chart((points + trend).properties(height=320), use_container_width=True)
        else:
            st.info("Need at least 2 email rows for correlation trendline.")

def render_landing_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Landing Page Performance")
    if not st.session_state.get("hide_all_inputs", False):
        txt = st.text_area(
            "Paste landing page analytics text",
            height=220,
            placeholder="Paste GA4 landing page metrics here...",
            key="landing_input_text",
        )
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
                    st.success("Landing page metrics parsed successfully.")
                else:
                    st.error("Landing parse failed.")
                    with st.expander("Model output/debug"):
                        st.text(dbg)
    d = st.session_state["landing_metrics_dict"]
    if d:
        views = int(to_int(d.get("views")) or 0)
        active = int(to_int(d.get("active_users")) or 0)
        vpu = float(to_float(d.get("views_per_user")) or 0)
        engage = float(to_float(d.get("avg_engagement_seconds")) or 0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Views", f"{views:,}")
        c2.metric("Active Users", f"{active:,}")
        c3.metric("Views/User", f"{vpu:.2f}")
        c4.metric("Avg Engagement (sec)", f"{engage:.2f}")

        left, right = st.columns([1.4, 1])
        with left:
            target_vpu = 1.50
            benchmark = pd.DataFrame({"label": ["Target", "Current"], "value": [target_vpu, vpu]})
            bench_chart = (
                alt.Chart(benchmark)
                .mark_bar(cornerRadiusEnd=6)
                .encode(
                    y=alt.Y("label:N", sort=["Target", "Current"], title=""),
                    x=alt.X("value:Q", title="Views per User"),
                    color=alt.Color("label:N", scale=alt.Scale(domain=["Target", "Current"], range=["#cbd5e1", "#2563eb"]), legend=None),
                    tooltip=["label:N", alt.Tooltip("value:Q", format=".2f")],
                )
                .properties(height=190, title="Views/User Benchmark")
            )
            st.altair_chart(bench_chart, use_container_width=True)

        jp, en = to_int(d.get("jp_views")) or 0, to_int(d.get("en_views")) or 0
        with right:
            if jp + en > 0:
                pie = pd.DataFrame({"lang": ["JP", "EN"], "views": [jp, en]})
                donut = (
                    alt.Chart(pie)
                    .mark_arc(innerRadius=58, outerRadius=95)
                    .encode(
                        theta=alt.Theta("views:Q"),
                        color=alt.Color("lang:N", scale=alt.Scale(domain=["JP", "EN"], range=["#10b981", "#f59e0b"])),
                        tooltip=["lang:N", "views:Q"],
                    )
                    .properties(height=220, title="JP vs EN View Mix")
                )
                st.altair_chart(donut, use_container_width=True)
            else:
                st.info("Add JP/EN split in landing input to render mix chart.")

        st.markdown("#### Data Science View: Engagement Z-Score Trend")
        hist = _load_history()
        zdf = pd.DataFrame()
        if not hist.empty and {"saved_at_utc", "landing_avg_engagement_seconds"}.issubset(hist.columns):
            zdf = hist[["saved_at_utc", "landing_avg_engagement_seconds"]].copy()
            zdf["saved_at_utc"] = pd.to_datetime(zdf["saved_at_utc"], errors="coerce")
            zdf["landing_avg_engagement_seconds"] = pd.to_numeric(zdf["landing_avg_engagement_seconds"], errors="coerce")
            zdf = zdf.dropna().sort_values("saved_at_utc")
        current = pd.DataFrame({"saved_at_utc": [pd.Timestamp.utcnow()], "landing_avg_engagement_seconds": [engage]})
        zdf = pd.concat([zdf, current], ignore_index=True)
        zdf = zdf.dropna().sort_values("saved_at_utc").reset_index(drop=True)
        if len(zdf) >= 3 and float(zdf["landing_avg_engagement_seconds"].std(ddof=0) or 0) > 0:
            mu = float(zdf["landing_avg_engagement_seconds"].mean())
            sigma = float(zdf["landing_avg_engagement_seconds"].std(ddof=0))
            zdf["z"] = (zdf["landing_avg_engagement_seconds"] - mu) / sigma
            zdf["kind"] = "History"
            zdf.loc[zdf.index.max(), "kind"] = "Current"
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94a3b8", strokeDash=[6, 4]).encode(y="y:Q")
            line = (
                alt.Chart(zdf)
                .mark_line(color="#2563eb", point=True)
                .encode(
                    x=alt.X("saved_at_utc:T", title="Snapshot Date"),
                    y=alt.Y("z:Q", title="Engagement Z-Score"),
                    tooltip=[alt.Tooltip("saved_at_utc:T"), alt.Tooltip("landing_avg_engagement_seconds:Q", format=".2f"), alt.Tooltip("z:Q", format=".2f")],
                )
            )
            highlight = alt.Chart(zdf[zdf["kind"] == "Current"]).mark_point(size=180, color="#ef4444").encode(x="saved_at_utc:T", y="z:Q")
            st.altair_chart((zero + line + highlight).properties(height=330), use_container_width=True)
        else:
            st.info("Save at least 3 webinars to unlock anomaly trend detection.")


def render_social_tab() -> None:
    st.markdown("### Social Media (Organic)")
    if not st.session_state.get("hide_all_inputs", False):
        li_txt = st.text_area("Paste LinkedIn post analytics text", height=180, placeholder="Paste LinkedIn organic post analytics text here...", key="social_linkedin_input_text")
        fb_txt = st.text_area("Paste Facebook post insights text", height=180, placeholder="Paste Facebook organic post insights text here...", key="social_facebook_input_text")
        if st.button("Parse Social Media (Organic)"):
            if not (li_txt.strip() or fb_txt.strip()):
                st.warning("Please paste LinkedIn and/or Facebook analytics text.")
            else:
                d, dbg, ok = parse_social(li_txt, fb_txt)
                if ok:
                    st.session_state["social_metrics_dict"] = d
                    st.success("Social media metrics parsed successfully.")
                else:
                    st.error("Social media parse failed.")
                    with st.expander("Parser debug"):
                        st.text(dbg)
    s = st.session_state.get("social_metrics_dict", {})
    li = s.get("linkedin", {}) or {}
    fb = s.get("facebook", {}) or {}
    if li or fb:
        li_impr = int(to_int(li.get("impressions")) or 0)
        li_eng = int(to_int(li.get("engagements")) or 0)
        li_clicks = int(to_int(li.get("clicks")) or 0)
        li_er = float(to_float(li.get("engagement_rate")) or 0)
        li_ctr = float(to_float(li.get("ctr")) or 0)
        fb_views = int(to_int(fb.get("views")) or 0)
        fb_eng = int(to_int(fb.get("engagements")) or 0)
        fb_clicks = int(to_int(fb.get("link_clicks")) or 0)
        fb_er = (100.0 * fb_eng / fb_views) if fb_views > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("LinkedIn Impressions", f"{li_impr:,}")
        c2.metric("LinkedIn Engagements", f"{li_eng:,}")
        c3.metric("Facebook Views", f"{fb_views:,}")
        c4.metric("Facebook Link Clicks", f"{fb_clicks:,}")
        p1, p2, p3 = st.columns(3)
        p1.metric("LinkedIn Engagement Rate", f"{li_er:.2f}%")
        p2.metric("LinkedIn CTR", f"{li_ctr:.2f}%")
        p3.metric("Facebook Engagement Rate", f"{fb_er:.2f}%")

        compare = pd.DataFrame({"metric": ["Reach", "Engagements", "Clicks"], "LinkedIn": [li_impr, li_eng, li_clicks], "Facebook": [fb_views, fb_eng, fb_clicks]}).melt(id_vars=["metric"], var_name="platform", value_name="value")
        compare_chart = alt.Chart(compare).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("metric:N", title="Metric"), y=alt.Y("value:Q", title="Count"),
            color=alt.Color("platform:N", scale=alt.Scale(domain=["LinkedIn", "Facebook"], range=["#2563eb", "#10b981"])),
            xOffset="platform:N", tooltip=["metric:N", "platform:N", "value:Q"],
        ).properties(height=320, title="Platform Performance Comparison")
        st.altair_chart(compare_chart, use_container_width=True)
        comp_df = pd.DataFrame({
            "platform": ["LinkedIn", "LinkedIn", "LinkedIn", "LinkedIn", "Facebook", "Facebook"],
            "component": ["Clicks", "Reactions", "Comments", "Reposts", "Link Clicks", "Other Engagements"],
            "value": [li_clicks, int(to_int(li.get("reactions")) or 0), int(to_int(li.get("comments")) or 0), int(to_int(li.get("reposts")) or 0), fb_clicks, max(fb_eng - fb_clicks, 0)],
        })
        comp_df = comp_df[comp_df["value"] > 0]
        if not comp_df.empty:
            comp_chart = alt.Chart(comp_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("platform:N", title=""), y=alt.Y("value:Q", title="Count"),
                color=alt.Color("component:N", scale=alt.Scale(range=["#2563eb", "#f59e0b", "#ef4444", "#8b5cf6", "#10b981", "#94a3b8"])),
                tooltip=["platform:N", "component:N", "value:Q"],
            ).properties(height=290, title="Engagement Composition by Platform")
            st.altair_chart(comp_chart, use_container_width=True)

        st.markdown("#### Data Science View: Organic Efficiency Frontier")
        frontier = pd.DataFrame({"platform": ["LinkedIn", "Facebook"], "reach": [li_impr, fb_views], "engagement_rate": [li_er, fb_er], "clicks": [li_clicks, fb_clicks]})
        frontier = frontier[(frontier["reach"] > 0) | (frontier["engagement_rate"] > 0) | (frontier["clicks"] > 0)]
        if not frontier.empty:
            front_chart = alt.Chart(frontier).mark_circle(opacity=0.85, stroke="#0f172a", strokeWidth=0.5).encode(
                x=alt.X("reach:Q", title="Reach (Impressions/Views)"),
                y=alt.Y("engagement_rate:Q", title="Engagement Rate (%)"),
                size=alt.Size("clicks:Q", title="Clicks", scale=alt.Scale(range=[200, 1200])),
                color=alt.Color("platform:N", scale=alt.Scale(domain=["LinkedIn", "Facebook"], range=["#2563eb", "#10b981"])),
                tooltip=["platform:N", "reach:Q", alt.Tooltip("engagement_rate:Q", format=".2f"), "clicks:Q"],
            ).properties(height=330)
            st.altair_chart(front_chart, use_container_width=True)

def render_regs_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Registrants + Attendees")
    if not st.session_state.get("hide_all_inputs", False):
        reg_txt = st.text_area(
            "Paste registrants list text",
            height=220,
            placeholder="Paste Salesforce/Pardot registrants list export here...",
            key="regs_input_text_registrants",
        )
        att_txt = st.text_area(
            "Paste attendees list text",
            height=220,
            placeholder="Paste Salesforce/Pardot attendees list export here...",
            key="regs_input_text_attendees",
        )
        with st.expander("Example paste format"):
            st.code("Name: A | Company: X | Score: 10 | Last Submitted: 2025-11-01\nName: B | Company: Y | Score: 8 | Last Submitted: 2025-11-02")
        if st.button("Parse Registrants + Attendees"):
            if not (reg_txt.strip() or att_txt.strip()):
                st.warning("Please paste registrants and/or attendees text.")
            else:
                frames = []
                debugs = []
                for label, raw_text, forced_type in [
                    ("registrants", reg_txt, "registrant"),
                    ("attendees", att_txt, "attendee"),
                ]:
                    if not raw_text.strip():
                        continue
                    df_part, dbg, ok = parse_regs(raw_text, api_key, model, temp)
                    if not ok:
                        debugs.append(f"[{label}] {dbg}")
                        continue
                    has_people = ("name" in df_part.columns) and df_part["name"].notna().astype(str).str.strip().ne("").any()
                    if not has_people:
                        fallback_rows = []
                        fallback_rows += _extract_regs_salesforce_list(raw_text)
                        fallback_rows += _extract_regs_salesforce_regex(raw_text)
                        if not fallback_rows:
                            fallback_rows += _extract_regs_rulebased(raw_text)
                        if fallback_rows:
                            df_part = pd.DataFrame(fallback_rows)
                            has_people = ("name" in df_part.columns) and df_part["name"].notna().astype(str).str.strip().ne("").any()
                            if has_people:
                                debugs.append(f"[{label}] recovered row-level people via direct extractor fallback.")
                    if df_part.empty:
                        debugs.append(f"[{label}] parsed 0 rows.")
                        continue
                    part = df_part.copy()
                    part["list_type"] = forced_type
                    if "list_name" not in part.columns:
                        part["list_name"] = f"{label}_manual"
                    frames.append(part)

                if frames:
                    st.session_state["registrants_df"] = pd.concat(frames, ignore_index=True)
                    st.success("Registrant data parsed successfully.")
                    if debugs:
                        with st.expander("Parser debug"):
                            st.text("\n\n".join(debugs))
                else:
                    st.error("Registrant parse failed.")
                    if debugs:
                        with st.expander("Parser debug"):
                            st.text("\n\n".join(debugs))
    df = st.session_state["registrants_df"]
    if df.empty:
        return

    x = df.copy()
    for col in ["name", "company", "last_submitted", "list_type", "list_name"]:
        if col not in x.columns:
            x[col] = None
    x["name_clean"] = x["name"].where(x["name"].notna(), "").astype(str).str.strip()
    x.loc[x["name_clean"].str.lower().isin(["none", "nan"]), "name_clean"] = ""
    x["company_clean"] = x["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    x["score_num"] = pd.to_numeric(x.get("score"), errors="coerce")
    x["list_type"] = x["list_type"].astype(str).str.strip().str.lower()
    missing_type = x["list_type"].isin(["", "none", "nan"]) & x["list_name"].notna()
    x.loc[missing_type, "list_type"] = x.loc[missing_type, "list_name"].astype(str).str.contains("attendee", case=False, na=False).map({True: "attendee", False: "registrant"})

    date_candidates = [c for c in ["last_submitted", "last_activity", "registered_at", "registered_time", "joined", "created"] if c in x.columns]
    date_col = date_candidates[0] if date_candidates else None
    if date_candidates:
        parsed_dates = pd.DataFrame({c: pd.to_datetime(x[c], errors="coerce") for c in date_candidates})
        # Use first available timestamp per row to keep timing view robust on noisy exports.
        x["registered_dt"] = parsed_dates.bfill(axis=1).iloc[:, 0]
    else:
        x["registered_dt"] = pd.NaT
    person_rows = x[x["name_clean"].ne("")].copy()

    raw_people_frames = []
    for raw_key, forced_type in [("regs_input_text_registrants", "registrant"), ("regs_input_text_attendees", "attendee")]:
        raw_text = st.session_state.get(raw_key, "")
        if not str(raw_text).strip():
            continue
        rows = []
        rows += _extract_regs_salesforce_list(raw_text)
        rows += _extract_regs_salesforce_regex(raw_text)
        if not rows:
            rows += _extract_regs_rulebased(raw_text)
        if not rows:
            continue
        rdf = pd.DataFrame(rows)
        if rdf.empty:
            continue
        rdf["list_type"] = forced_type
        for col in ["name", "company", "last_submitted", "last_activity", "score"]:
            if col not in rdf.columns:
                rdf[col] = None
        rdf["name_clean"] = rdf["name"].where(rdf["name"].notna(), "").astype(str).str.strip()
        rdf = rdf[rdf["name_clean"].ne("")].copy()
        if rdf.empty:
            continue
        rdf["company_clean"] = rdf["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        rdf["score_num"] = pd.to_numeric(rdf["score"], errors="coerce")
        rdf["registered_dt"] = pd.to_datetime(rdf["last_submitted"], errors="coerce")
        miss_dt = rdf["registered_dt"].isna()
        if miss_dt.any():
            rdf.loc[miss_dt, "registered_dt"] = pd.to_datetime(rdf.loc[miss_dt, "last_activity"], errors="coerce")
        raw_people_frames.append(rdf)

    if raw_people_frames:
        person_rows = pd.concat(raw_people_frames, ignore_index=True)

    reg_rows = x[x["list_type"] == "registrant"] if "list_type" in x.columns else x
    att_rows = x[x["list_type"] == "attendee"] if "list_type" in x.columns else pd.DataFrame()
    reg_names = set(reg_rows.loc[reg_rows["name_clean"].ne(""), "name_clean"].tolist())
    att_names = set(att_rows.loc[att_rows["name_clean"].ne(""), "name_clean"].tolist())

    if reg_names:
        reg_count = len(reg_names)
    else:
        reg_count = int(x["name_clean"].ne("").sum())
    if att_names:
        att_count = len(att_names)
    else:
        att_count = int((x["score_num"].fillna(0) > 0).sum())
    if {"total_prospects", "list_type"}.issubset(x.columns):
        totals = x.copy()
        totals["total_prospects"] = pd.to_numeric(totals["total_prospects"], errors="coerce").fillna(0)
        reg_total = int(totals.loc[totals["list_type"] == "registrant", "total_prospects"].sum())
        att_total = int(totals.loc[totals["list_type"] == "attendee", "total_prospects"].sum())
        if reg_count == 0 and reg_total > 0:
            reg_count = reg_total
        if att_count == 0 and att_total > 0:
            att_count = att_total
    attendance_rate = (100.0 * att_count / reg_count) if reg_count > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Registrants", f"{reg_count:,}")
    c2.metric("Attendees", f"{att_count:,}")
    c3.metric("Registrant -> Attendee", f"{attendance_rate:.1f}%")

    funnel = pd.DataFrame({"stage": ["Registrants", "Attendees"], "value": [reg_count, att_count]})
    fchart = (
        alt.Chart(funnel)
        .mark_bar(size=42, cornerRadiusEnd=6, color="#2563eb")
        .encode(
            y=alt.Y("stage:N", sort=["Registrants", "Attendees"], title=""),
            x=alt.X("value:Q", title="People"),
            tooltip=["stage:N", "value:Q"],
        )
        .properties(height=180, title="Registrant to Attendee Funnel")
    )
    st.altair_chart(fchart, use_container_width=True)

    reg_people = person_rows[person_rows["list_type"] == "registrant"] if "list_type" in person_rows.columns else person_rows
    top_source = reg_people if not reg_people.empty else person_rows
    reg_time_col = date_col if date_col else "last_submitted"
    top_people = (
        top_source[["name", "company_clean", reg_time_col, "score_num"]]
        .rename(columns={"company_clean": "company", reg_time_col: "registered_time", "score_num": "score"})
        .dropna(subset=["name"])
        .sort_values(["score"], ascending=False, na_position="last")
        .drop_duplicates(subset=["name", "company"], keep="first")
        .head(10)
    )
    st.markdown("#### Top Scoring People")
    if top_people.empty:
        st.info("No row-level people records were parsed.")
    else:
        st.dataframe(top_people, use_container_width=True)

    timed = reg_people if not reg_people.empty else person_rows
    timed = timed.dropna(subset=["registered_dt"]).copy()
    st.markdown("#### Registration Timing")
    if timed.empty:
        if person_rows.empty:
            st.info("Only list-level summary metrics were parsed from this paste. Top scoring people and registration timing need row-level Name/Score/Joined rows.")
        else:
            st.info("No registration timestamps found (expected `last_submitted` or similar date field).")
    else:
        by_day = timed.groupby(timed["registered_dt"].dt.to_period("W-MON").dt.start_time).size().reset_index(name="registrations")
        by_day.columns = ["week_start", "registrations"]
        reg_chart = (
            alt.Chart(by_day)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#10b981")
            .encode(
                x=alt.X("week_start:T", title="Week Start"),
                y=alt.Y("registrations:Q", title="People Registered"),
                tooltip=[alt.Tooltip("week_start:T"), "registrations:Q"],
            )
            .properties(height=300, title="How Many People Registered by Week")
        )
        st.altair_chart(reg_chart, use_container_width=True)

def render_survey_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Survey Insights")
    if not st.session_state.get("hide_all_inputs", False):
        up = st.file_uploader("Upload MS Forms CSV", type=["csv"], key="survey_uploader")
        txt = st.text_area("Or paste survey text", height=180, placeholder="Paste survey text here if CSV is not available...", key="survey_input_text")
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

        row1_l, row1_r = st.columns([1.3, 1])
        with row1_l:
            dist = (d.get("value_rating_stats") or {}).get("distribution_counts", {})
            if dist:
                q9 = pd.DataFrame({"rating": [str(k) for k in dist.keys()], "count": list(dist.values())})
                q9["rating_num"] = pd.to_numeric(q9["rating"], errors="coerce")
                q9 = q9.sort_values("rating_num")
                q9_chart = alt.Chart(q9).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2563eb").encode(
                    x=alt.X("rating:N", title="Rating"), y=alt.Y("count:Q", title="Responses"), tooltip=["rating:N", "count:Q"],
                ).properties(height=280, title="Q9 Value Rating Distribution")
                st.altair_chart(q9_chart, use_container_width=True)
            else:
                st.info("Q9 rating distribution not available.")
        with row1_r:
            yn = pd.DataFrame({"answer": ["Yes", "No"], "count": [y, n0]})
            if yn["count"].sum() > 0:
                yn_chart = alt.Chart(yn).mark_arc(innerRadius=52, outerRadius=90).encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("answer:N", scale=alt.Scale(domain=["Yes", "No"], range=["#10b981", "#ef4444"])),
                    tooltip=["answer:N", "count:Q"],
                ).properties(height=280, title="Consultation Intent Mix")
                st.altair_chart(yn_chart, use_container_width=True)
            else:
                st.info("Consultation yes/no values not available.")

        for title, key, xname, color in [
            ("Job Function Distribution (Q6)", "job_function_counts", "job_function", "#10b981"),
            ("Job Level Distribution (Q7)", "job_level_counts", "job_level", "#f59e0b"),
            ("Industry Distribution (Q8)", "industry", "industry", "#6366f1"),
        ]:
            src = d.get(key, {}) if key != "industry" else d.get("industry_counts", {})
            if src:
                z = pd.DataFrame({xname: list(src.keys()), "count": list(src.values())})
                z[xname] = z[xname].astype(str)
                z = z.sort_values("count", ascending=False).head(12)
                chart = alt.Chart(z).mark_bar(cornerRadiusEnd=5, color=color).encode(
                    y=alt.Y(f"{xname}:N", sort="-x", title=""), x=alt.X("count:Q", title="Responses"), tooltip=[f"{xname}:N", "count:Q"],
                ).properties(height=260, title=title)
                st.altair_chart(chart, use_container_width=True)

        st.markdown("#### Data Science View: Industry Response Concentration")
        ind = d.get("industry_counts", {})
        if ind and sum(ind.values()) > 0:
            conc = pd.DataFrame({"industry": list(ind.keys()), "count": list(ind.values())}).sort_values("count", ascending=False).reset_index(drop=True)
            conc["rank"] = conc.index + 1
            conc["share"] = conc["count"] / conc["count"].sum()
            conc["cum_share"] = conc["share"].cumsum()
            conc_chart = alt.Chart(conc).mark_line(point=True, color="#2563eb").encode(
                x=alt.X("rank:Q", title="Industry Rank (largest to smallest)"),
                y=alt.Y("cum_share:Q", axis=alt.Axis(format="%"), title="Cumulative Share"),
                tooltip=["industry:N", "count:Q", alt.Tooltip("cum_share:Q", format=".1%")],
            ).properties(height=300)
            st.altair_chart(conc_chart, use_container_width=True)
        else:
            st.info("Need industry distribution data for concentration analysis.")

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


def render_exec_summary_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Executive Summary")
    st.caption("Copy hint: click in the text area and copy.")
    if st.button("Generate Executive Summary"):
        if not api_key:
            st.error("API key missing. Set OPENAI_API_KEY or provide key in sidebar.")
        else:
            s, dbg = exec_summary(api_key, model, temp, st.session_state["parsed_emails_df"], st.session_state["landing_metrics_dict"], st.session_state["social_metrics_dict"], st.session_state["registrants_df"], st.session_state["survey_derived"])
            if s:
                st.session_state["exec_summary_text"] = s
                st.success("Executive summary generated.")
            else:
                st.error("Executive summary generation failed.")
                with st.expander("Model output/debug"):
                    st.text(dbg)
    summary_text = st.session_state.get("exec_summary_text", "").strip()
    if summary_text:
        emails_df = st.session_state["parsed_emails_df"]
        landing = st.session_state["landing_metrics_dict"] or {}
        social = st.session_state["social_metrics_dict"] or {}
        regs_df = st.session_state["registrants_df"]
        survey = st.session_state["survey_derived"] or {}
        sent = int(pd.to_numeric(emails_df.get("total_sent"), errors="coerce").fillna(0).sum()) if not emails_df.empty else 0
        delivered = int(pd.to_numeric(emails_df.get("total_delivered"), errors="coerce").fillna(0).sum()) if not emails_df.empty else 0
        opens = int(pd.to_numeric(emails_df.get("unique_opens"), errors="coerce").fillna(0).sum()) if not emails_df.empty else 0
        clicks = int(pd.to_numeric(emails_df.get("unique_clicks"), errors="coerce").fillna(0).sum()) if not emails_df.empty else 0
        landing_visits = int(to_int(landing.get("views")) or 0)
        consult_yes = int(to_int(survey.get("consult_yes_count")) or 0)

        if sent <= 0 and delivered > 0:
            sent = delivered
        if opens <= 0 and delivered > 0:
            avg_open = pd.to_numeric(emails_df.get("open_rate"), errors="coerce").mean()
            if pd.notna(avg_open):
                opens = int(round(delivered * float(avg_open) / 100.0))
        if clicks <= 0 and delivered > 0:
            avg_ctr = pd.to_numeric(emails_df.get("unique_ctr"), errors="coerce").mean()
            if pd.notna(avg_ctr):
                clicks = int(round(delivered * float(avg_ctr) / 100.0))

        registrations = 0
        attendance = None
        if not regs_df.empty:
            if "name" in regs_df.columns:
                registrations = int(regs_df["name"].fillna("").astype(str).str.strip().ne("").sum())
                if "score" in regs_df.columns:
                    scores = pd.to_numeric(regs_df["score"], errors="coerce")
                    if len(scores.dropna()) > 0:
                        attendance = int((scores.fillna(0) > 0).sum())
            elif "total_prospects" in regs_df.columns:
                totals = regs_df.copy()
                totals["total_prospects"] = pd.to_numeric(totals["total_prospects"], errors="coerce").fillna(0)
                if "list_type" in totals.columns:
                    lt = totals["list_type"].astype(str).str.strip().str.lower()
                    reg_total = int(totals.loc[lt == "registrant", "total_prospects"].sum())
                    att_total = int(totals.loc[lt == "attendee", "total_prospects"].sum())
                    if reg_total > 0:
                        registrations = reg_total
                    else:
                        registrations = int(totals["total_prospects"].sum())
                    if att_total > 0:
                        attendance = att_total
                else:
                    registrations = int(totals["total_prospects"].sum())

        st.markdown("#### Full Funnel")
        st.caption("Email Sent -> Open -> Click -> Landing Visit -> Registration -> Attendance -> Consultation Lead")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Email Sent", f"{sent:,}")
        r1c2.metric("Open", f"{opens:,}")
        r1c3.metric("Click", f"{clicks:,}")
        r1c4.metric("Landing Visit", f"{landing_visits:,}")
        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.metric("Registration", f"{registrations:,}")
        r2c2.metric("Attendance", f"{attendance:,}" if attendance is not None else "N/A")
        r2c3.metric("Consultation Lead", f"{consult_yes:,}")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", summary_text) if s.strip()]
        highlights = sentences[:4]
        if highlights:
            st.markdown("#### Key Highlights")
            for h in highlights:
                st.write(f"- {h}")

        with st.container(border=True):
            st.markdown("#### Narrative")
            st.markdown(summary_text)
