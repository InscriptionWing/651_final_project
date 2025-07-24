from pydantic import BaseModel, Field, ValidationError
from typing import List
from datetime import date, time

class IncidentRecord(BaseModel):
    date: date
    name: str
    start_time: time
    duration_minutes: int = Field(gt=0, lt=240)
    location: str
    activity: str
    participation: str
    actions_taken: List[str]
    contributing_factors: List[str]
    productivity_level: int = Field(ge=1, le=5)
    engagement_level: int = Field(ge=1, le=3)
    narrative: str
