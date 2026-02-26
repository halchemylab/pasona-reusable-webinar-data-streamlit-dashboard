import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from parsers import exec_summary, parse_emails, parse_landing, parse_regs, parse_social, parse_survey_csv, parse_survey_text
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

        st.markdown("### Parsed Social Metrics")
        st.dataframe(pd.DataFrame([li, fb]), use_container_width=True)


def render_regs_tab(api_key: str, model: str, temp: float) -> None:
    st.markdown("### Registrants + Attendees")
    if not st.session_state.get("hide_all_inputs", False):
        txt = st.text_area("Paste registrant/attendee text", height=220, placeholder="Paste registrants/attendees export text here...", key="regs_input_text")
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
                    st.success("Registrant data parsed successfully.")
                else:
                    st.error("Registrant parse failed.")
                    with st.expander("Model output/debug"):
                        st.text(dbg)
    df = st.session_state["registrants_df"]
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        if {"list_name", "total_prospects", "mailable_prospects", "mailable_rate"}.issubset(df.columns):
            x = df.copy()
            x["total_prospects"] = pd.to_numeric(x["total_prospects"], errors="coerce")
            x["mailable_prospects"] = pd.to_numeric(x["mailable_prospects"], errors="coerce")
            x["mailable_rate"] = pd.to_numeric(x["mailable_rate"], errors="coerce")

            n_lists = int(len(x))
            total = int(x["total_prospects"].fillna(0).sum())
            mailable = int(x["mailable_prospects"].fillna(0).sum())
            weighted_rate = (100.0 * mailable / total) if total > 0 else float(x["mailable_rate"].mean() or 0)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Lists Parsed", f"{n_lists:,}")
            c2.metric("Total Prospects", f"{total:,}")
            c3.metric("Mailable Prospects", f"{mailable:,}")
            c4.metric("Mailable Rate", f"{weighted_rate:.1f}%")

            bars = (
                x[["list_name", "total_prospects", "mailable_prospects"]]
                .melt(id_vars=["list_name"], var_name="metric", value_name="value")
                .dropna(subset=["value"])
            )
            if not bars.empty:
                chart = (
                    alt.Chart(bars)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("list_name:N", title="List"),
                        y=alt.Y("value:Q", title="Prospects"),
                        color=alt.Color("metric:N", scale=alt.Scale(domain=["total_prospects", "mailable_prospects"], range=["#2563eb", "#10b981"])),
                        xOffset="metric:N",
                        tooltip=["list_name:N", "metric:N", "value:Q"],
                    )
                    .properties(height=300, title="Prospect Volume by List")
                )
                st.altair_chart(chart, use_container_width=True)

            st.markdown("#### Data Science View: Mailable Efficiency Frontier")
            frontier = x.dropna(subset=["total_prospects"]).copy()
            if not frontier.empty:
                if frontier["mailable_rate"].isna().all():
                    frontier["mailable_rate"] = (100 * frontier["mailable_prospects"] / frontier["total_prospects"]).fillna(0)
                front = (
                    alt.Chart(frontier)
                    .mark_circle(size=220, color="#2563eb", opacity=0.85)
                    .encode(
                        x=alt.X("total_prospects:Q", title="Total Prospects"),
                        y=alt.Y("mailable_rate:Q", title="Mailable Rate (%)"),
                        tooltip=["list_name:N", "total_prospects:Q", "mailable_prospects:Q", alt.Tooltip("mailable_rate:Q", format=".1f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(front, use_container_width=True)
            return

        x = df.copy()
        if "last_submitted" in x.columns:
            x["submitted_dt"] = pd.to_datetime(x["last_submitted"], errors="coerce")
        else:
            x["submitted_dt"] = pd.NaT
        x["score_num"] = pd.to_numeric(x.get("score"), errors="coerce")
        if "company" in x.columns:
            x["company_clean"] = x["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
        else:
            x["company_clean"] = pd.NA

        total_regs = len(x)
        unique_companies = int(x["company_clean"].dropna().nunique())
        median_score = float(x["score_num"].median()) if len(x["score_num"].dropna()) > 0 else None
        dated = x.dropna(subset=["submitted_dt"])
        day_span = max((dated["submitted_dt"].max() - dated["submitted_dt"].min()).days + 1, 1) if not dated.empty else 1
        avg_daily = (len(dated) / day_span) if not dated.empty else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Registrants", f"{total_regs:,}")
        c2.metric("Unique Companies", f"{unique_companies:,}")
        c3.metric("Median Score", f"{median_score:.1f}" if median_score is not None else "N/A")
        c4.metric("Avg Registrations/Day", f"{avg_daily:.2f}")

        left, right = st.columns([1.3, 1])
        with left:
            if not dated.empty:
                by_day = dated.groupby(dated["submitted_dt"].dt.date).size().reset_index(name="count")
                by_day.columns = ["date", "count"]
                line = alt.Chart(by_day).mark_area(line={"color": "#2563eb"}, color="#93c5fd", opacity=0.45).encode(
                    x=alt.X("date:T", title="Date"), y=alt.Y("count:Q", title="Registrations"), tooltip=[alt.Tooltip("date:T"), "count:Q"],
                ).properties(height=300, title="Registrations Over Time")
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("Include `last_submitted` values to render trend chart.")
        with right:
            top = x["company_clean"].dropna().value_counts().head(10).reset_index()
            if not top.empty:
                top.columns = ["company", "count"]
                bars = alt.Chart(top).mark_bar(cornerRadiusEnd=5, color="#10b981").encode(
                    y=alt.Y("company:N", sort="-x", title=""), x=alt.X("count:Q", title="Registrants"), tooltip=["company:N", "count:Q"],
                ).properties(height=300, title="Top Companies")
                st.altair_chart(bars, use_container_width=True)
            else:
                st.info("No company field found for top-company chart.")

        st.markdown("#### Data Science View: Weekly Cohort Heatmap")
        if not dated.empty:
            heat = dated.copy()
            heat["week"] = heat["submitted_dt"].dt.to_period("W").astype(str)
            heat["weekday"] = heat["submitted_dt"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            heat_map = heat.groupby(["week", "weekday"]).size().reset_index(name="count")
            heat_chart = alt.Chart(heat_map).mark_rect().encode(
                x=alt.X("week:N", title="Week Cohort"), y=alt.Y("weekday:N", sort=order, title=""),
                color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Registrants"), tooltip=["week:N", "weekday:N", "count:Q"],
            ).properties(height=260)
            st.altair_chart(heat_chart, use_container_width=True)
        else:
            st.info("Need date data to build cohort heatmap.")

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
        survey = st.session_state["survey_derived"] or {}
        avg_open = pd.to_numeric(emails_df.get("open_rate"), errors="coerce").mean() if not emails_df.empty else None
        landing_views = int(to_int(landing.get("views")) or 0)
        li_imp = int(to_int((social.get("linkedin") or {}).get("impressions")) or 0)
        fb_views = int(to_int((social.get("facebook") or {}).get("views")) or 0)
        consult_yes = int(to_int(survey.get("consult_yes_count")) or 0)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Email Avg Open", f"{avg_open:.2f}%" if avg_open is not None else "N/A")
        k2.metric("Landing Views", f"{landing_views:,}")
        k3.metric("Organic Reach", f"{li_imp + fb_views:,}")
        k4.metric("Consultation Leads", f"{consult_yes:,}")

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", summary_text) if s.strip()]
        highlights = sentences[:4]
        if highlights:
            st.markdown("#### Key Highlights")
            for h in highlights:
                st.write(f"- {h}")

        with st.container(border=True):
            st.markdown("#### Narrative")
            st.markdown(summary_text)

    st.text_area("Executive Summary (copy-ready)", value=summary_text, height=220)
