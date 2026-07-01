"""
schemas.py
Pydantic data models for ZeraMatumizi FastAPI backend.
Defines request and response schemas for all API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class IndividualRiskRequest(BaseModel):
    """Request schema for individual disorder risk prediction."""
    age: int = Field(..., ge=18, le=65, description="Age in years")
    gender: str = Field(..., pattern="^(male|female)$")
    education_level: str = Field(
        ..., pattern="^(none|primary|secondary|tertiary)$"
    )
    wealth_index: str = Field(
        ..., pattern="^(poorest|poor|middle|rich|richest)$"
    )
    alcohol_use: int = Field(..., ge=0, le=1)
    cannabis_use: int = Field(..., ge=0, le=1)
    khat_use: int = Field(..., ge=0, le=1)
    age_of_initiation: int = Field(..., ge=10, le=60)
    hiv_status: str = Field(
        ..., pattern="^(positive|negative|unknown)$"
    )
    employment_status: str = Field(
        ..., pattern="^(employed|unemployed|student)$"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "age": 24,
                "gender": "male",
                "education_level": "secondary",
                "wealth_index": "poor",
                "alcohol_use": 1,
                "cannabis_use": 0,
                "khat_use": 1,
                "age_of_initiation": 16,
                "hiv_status": "negative",
                "employment_status": "unemployed"
            }
        }


class IndividualRiskResponse(BaseModel):
    """Response schema for individual disorder risk prediction."""
    risk_score: float = Field(..., description="Disorder progression probability 0-1")
    risk_tier: str = Field(..., description="Low/Medium/High/Very High")
    top_risk_factors: list = Field(..., description="Top 3 SHAP feature contributions")
    recommendation: str = Field(..., description="Intervention recommendation")


class CountyRiskResponse(BaseModel):
    """Response schema for county risk estimates."""
    county: str
    mean_risk: float
    lower_ci: float
    upper_ci: float
    risk_tier: str
    disorder_rate: float
    unemployment_rate: float


class RAGQueryRequest(BaseModel):
    """Request schema for RAG pipeline query."""
    query: str = Field(..., min_length=5, max_length=500)
    language: str = Field(default="english", pattern="^(english|swahili)$")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "A student appears to be using bhang. What steps should I take?",
                "language": "english"
            }
        }


class RAGQueryResponse(BaseModel):
    """Response schema for RAG pipeline query."""
    query: str
    response: str
    sources: list
    language: str


class ResourceAllocationResponse(BaseModel):
    """Response schema for resource allocation results."""
    county: str
    need_score: float
    classical_lp_slots: int
    qaoa_slots: int
    counsellors: int
    outreach_teams: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    message: str