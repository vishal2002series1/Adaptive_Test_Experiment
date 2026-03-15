from typing import TypedDict, Annotated, List, Dict, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import operator

# ==========================================
# 1. PYDANTIC MODELS (For Database & LLM Validation)
# ==========================================

class QuestionMetadata(BaseModel):
    exam: str = Field(description="Target examination, e.g., UPSC, SSC CGL")
    subject: str = Field(description="Broad subject, e.g., Economy, Science & Tech")
    topic: str
    sub_topic: str
    cognitive_skill: str = Field(description="e.g., Factual Recall, Analytical Reasoning")
    difficulty_level: int = Field(ge=1, le=5, description="1 (Beginner) to 5 (Exam-Ready)")
    ttl_days: Optional[int] = Field(default=None, description="Time-to-live in days. None for static subjects.")
    generation_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Question(BaseModel):
    id: str
    text: str = Field(description="The question text. Uses LaTeX delimiters for math/science.")
    options: Dict[str, str] = Field(description="e.g., {'A': 'Statement 1', 'B': 'Statement 2'}")
    correct_answer: str
    explanation: str
    metadata: QuestionMetadata

class StudentProfile(BaseModel):
    student_id: str
    target_exam: str
    tests_taken: int = 0
    overall_readiness_score: float = 0.0
    # Tracks proficiency score (0.0 to 1.0) per topic to guide the Orchestrator
    topic_proficiencies: Dict[str, float] = {}
    # Maps Question ID -> Number of times seen (Used for the Retrieval Penalty)
    seen_question_counts: Dict[str, int] = {}

class TestConfig(BaseModel):
    target_subject: str = "All Syllabus"
    target_difficulty: int = 3
    num_questions: int = 50

# ==========================================
# 2. LANGGRAPH STATE (The Shared Memory)
# ==========================================

class AdaptiveTestState(TypedDict):
    # --- Context ---
    profile: StudentProfile
    config: TestConfig
    
    # --- Execution Tracking ---
    current_question_index: int
    generation_attempts: int
    
    # --- Workspace ---
    # Annotated with operator.add so agents append questions to the list rather than overwriting it
    # --- Workspace (Updated for Batching) ---
    selected_questions: Annotated[List[Question], operator.add]
    draft_batch: List[Question]
    rejected_batch: List[Dict[str, str]] # e.g., [{"id": "q1", "feedback": "Too easy"}]
    current_batch_target: int
    
    # --- Post-Test Evaluation ---
    student_answers: Dict[str, str] # Maps question ID to the student's chosen option
    test_analysis: str
    study_plan: str 