from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from utils import to_float, to_int


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
    def parse_i(cls, v):
        return to_int(v)

    @field_validator("delivery_rate", "open_rate", "unique_ctr", "click_to_open", mode="before")
    @classmethod
    def parse_f(cls, v):
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
