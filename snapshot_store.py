import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

HISTORY_FILE = Path("data/webinar_history.csv")


def _json_default(v: Any):
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return str(v)


def _to_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=_json_default)


def _hash_payload(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=_json_default)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _top_companies(regs_df: pd.DataFrame) -> Dict[str, int]:
    if regs_df.empty or "company" not in regs_df.columns:
        return {}
    top = regs_df["company"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA}).dropna().value_counts().head(10)
    return {str(k): int(v) for k, v in top.to_dict().items()}


def has_snapshot_data(
    emails_df: pd.DataFrame, landing: Dict[str, Any], social: Dict[str, Any], regs_df: pd.DataFrame, survey: Dict[str, Any], exec_summary_text: str
) -> bool:
    return (not emails_df.empty) or bool(landing) or bool(social) or (not regs_df.empty) or bool(survey) or bool((exec_summary_text or "").strip())


def build_snapshot_row(
    webinar_name: str,
    emails_df: pd.DataFrame,
    landing: Dict[str, Any],
    social: Dict[str, Any],
    regs_df: pd.DataFrame,
    survey: Dict[str, Any],
    exec_summary_text: str,
) -> Dict[str, Any]:
    emails_records = emails_df.fillna("").to_dict(orient="records") if not emails_df.empty else []
    regs_records = regs_df.fillna("").to_dict(orient="records") if not regs_df.empty else []
    survey = survey or {}
    landing = landing or {}
    social = social or {}

    email_open = pd.to_numeric(emails_df.get("open_rate"), errors="coerce") if not emails_df.empty else pd.Series(dtype=float)
    email_ctr = pd.to_numeric(emails_df.get("unique_ctr"), errors="coerce") if not emails_df.empty else pd.Series(dtype=float)
    delivered = pd.to_numeric(emails_df.get("total_delivered"), errors="coerce").fillna(0).sum() if not emails_df.empty else 0
    unique_opens = pd.to_numeric(emails_df.get("unique_opens"), errors="coerce").fillna(0).sum() if not emails_df.empty else 0
    unique_clicks = pd.to_numeric(emails_df.get("unique_clicks"), errors="coerce").fillna(0).sum() if not emails_df.empty else 0

    payload_for_hash = {
        "webinar_name": webinar_name.strip(),
        "emails": emails_records,
        "landing": landing,
        "social_organic": social,
        "registrants": regs_records,
        "survey": survey,
        "exec_summary_text": exec_summary_text or "",
    }

    row = {
        "webinar_id": str(uuid.uuid4()),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "webinar_name": webinar_name.strip(),
        "source_hash": _hash_payload(payload_for_hash),
        "email_count": int(len(emails_records)),
        "avg_open_rate": float(email_open.mean()) if len(email_open.dropna()) > 0 else None,
        "avg_unique_ctr": float(email_ctr.mean()) if len(email_ctr.dropna()) > 0 else None,
        "total_delivered": int(delivered),
        "total_unique_opens": int(unique_opens),
        "total_unique_clicks": int(unique_clicks),
        "landing_views": landing.get("views"),
        "landing_active_users": landing.get("active_users"),
        "landing_views_per_user": landing.get("views_per_user"),
        "landing_avg_engagement_seconds": landing.get("avg_engagement_seconds"),
        "landing_jp_views": landing.get("jp_views"),
        "landing_en_views": landing.get("en_views"),
        "social_linkedin_impressions": (social.get("linkedin") or {}).get("impressions"),
        "social_linkedin_engagements": (social.get("linkedin") or {}).get("engagements"),
        "social_linkedin_clicks": (social.get("linkedin") or {}).get("clicks"),
        "social_facebook_views": (social.get("facebook") or {}).get("views"),
        "social_facebook_engagements": (social.get("facebook") or {}).get("engagements"),
        "social_facebook_link_clicks": (social.get("facebook") or {}).get("link_clicks"),
        "registrant_count": int(len(regs_records)),
        "top_companies_json": _to_json(_top_companies(regs_df)),
        "survey_n_responses": survey.get("n_responses"),
        "survey_avg_value_rating": (survey.get("value_rating_stats") or {}).get("avg"),
        "survey_consult_yes_count": survey.get("consult_yes_count"),
        "survey_consult_no_count": survey.get("consult_no_count"),
        "emails_json": _to_json(emails_records),
        "landing_json": _to_json(landing),
        "social_json": _to_json(social),
        "registrants_json": _to_json(regs_records),
        "survey_json": _to_json(survey),
        "consultation_leads_json": _to_json(survey.get("consult_yes_leads", [])),
        "themes_json": _to_json(survey.get("top_themes", [])),
        "exec_summary_text": (exec_summary_text or "").strip(),
    }
    return row


def append_snapshot_row(row: Dict[str, Any]) -> Path:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if HISTORY_FILE.exists():
        existing = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
        merged = pd.concat([existing, frame], ignore_index=True, sort=False)
        merged.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")
    else:
        frame.to_csv(HISTORY_FILE, mode="a", index=False, header=True, encoding="utf-8-sig")
    return HISTORY_FILE
