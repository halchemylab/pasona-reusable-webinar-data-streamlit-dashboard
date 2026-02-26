import streamlit as st


def apply_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-soft: #f8fafc;
            --ink: #0f172a;
            --blue: #2563eb;
            --green: #10b981;
            --amber: #f59e0b;
            --red: #ef4444;
            --slate: #64748b;
        }
        .stApp {
            background: linear-gradient(160deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.75);
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 8px 12px;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--slate);
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
