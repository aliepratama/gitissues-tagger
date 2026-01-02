from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional

class PredictionInput(BaseModel):
    title: str = Field(..., description="The issue title")
    body: str = Field(..., description="The issue body")

    @validator('title')
    def title_must_not_be_empty(cls, v):
        if v is not None and not v.strip():
             # It's okay if title is empty string if that's allowed, but usually we want some content.
             # The training script filled NaNs with empty string.
             pass
        return v

class PredictionOutput(BaseModel):
    labels: List[str] = Field(..., description="List of predicted labels")
    confidence_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for each label")
