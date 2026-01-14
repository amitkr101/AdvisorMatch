

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request model for search endpoint"""
    query: str = Field(..., description="Research query string", min_length=1)
    top_k: int = Field(10, description="Number of professors to return", ge=1, le=50)
    include_publications: bool = Field(True, description="Include top publications in response")


class PublicationSummary(BaseModel):
    """Summary of a publication"""
    paper_id: str
    title: str
    year: Optional[int]
    similarity: float = Field(..., description="Cosine similarity to query")
    citations: Optional[int]
    venue: Optional[str]
    url: Optional[str]


class ProfessorResult(BaseModel):
    """Professor search result with ranking details"""
    professor_id: int
    name: str
    department: str
    college: str
    interests: Optional[str]
    url: Optional[str]
    image_url: Optional[str] = None
    
    # Ranking details
    final_score: float = Field(..., description="Final ranking score")
    avg_similarity: float = Field(..., description="Average similarity of top papers")
    recency_weight: float = Field(..., description="Recency weighting factor")
    activity_bonus: float = Field(..., description="Activity bonus")
    citation_impact: float = Field(..., description="Citation impact score")
    num_matching_papers: int = Field(..., description="Number of matching papers")
    
    # Optional publications
    top_publications: Optional[List[PublicationSummary]] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint"""
    query: str
    results: List[ProfessorResult]
    total_results: int
    search_time_ms: float


class ProfessorDetail(BaseModel):
    """Detailed professor information"""
    id: int
    name: str
    college: str
    department: str
    interests: Optional[str]
    url: Optional[str]
    image_url: Optional[str] = None
    openalex_author_id: Optional[str]
    total_publications: int
    recent_publications: int  # Last 2 years


class PublicationAuthor(BaseModel):
    """Author information for a publication"""
    professor_id: int
    name: str
    position: int
    is_primary: bool


class PublicationDetail(BaseModel):
    """Detailed publication information"""
    paper_id: str
    title: str
    abstract: Optional[str]
    year: Optional[int]
    citation_count: Optional[int]
    venue: Optional[str]
    url: Optional[str]
    authors: List[PublicationAuthor]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database_connected: bool
    index_loaded: bool
    model_loaded: bool


# --- AI Assistant Models ---

class UnderstandResponse(BaseModel):
    themes: List[str]
    trajectory: str
    summary: str

class AngleRequest(BaseModel):
    professor_id: int
    student_interest: str = Field(..., min_length=5)
    resume_text: Optional[str] = None

class ResearchAngle(BaseModel):
    title: str
    logic: str
    background_needed: str

class AngleResponse(BaseModel):
    angles: List[ResearchAngle]

class NextStepsRequest(BaseModel):
    professor_id: int
    selected_angle: str
    student_level: str = Field(..., pattern="^(Undergrad|Masters|PhD)$")

class NextStepsResponse(BaseModel):
    checklist: List[str]
    outreach_tips: str

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    professor_id: int
    history: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
