from pydantic import BaseModel, Field,HttpUrl
from typing import List, Literal
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)

class QueryResponse(BaseModel):
    intents: List[str]
    personas: List[str]
    reason: str
    confidence: float = Field(..., ge=0, le=1)
    buying_intent_level: str
    keyword_type : str

class EnhancedQueryResponse(BaseModel):
    query: str

class AnalyzeRequest(BaseModel):
    website_url: HttpUrl
    keywords: str


class ScoreBlock(BaseModel):
    score: int
    good: List[str]
    needs_improvement: List[str]
    improvement_needed: str


class AnalyzeResponse(BaseModel):
    seo: ScoreBlock
    geo: ScoreBlock
    aeo: ScoreBlock
