from typing import TypedDict, Annotated, List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import operator

# ==========================================
# 0. NEW: STRUCTURED ANALYTICS MODELS
# ==========================================

class ProficiencyRecord(BaseModel):
    """Replaces the flat dictionary to enable SQL-style analytics on DynamoDB"""
    subject: str
    topic: str
    sub_topic: str
    score: float = 0.0
    questions_attempted: int = 0
    last_tested: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# ==========================================
# 1. PYDANTIC MODELS (Database & Validation)
# ==========================================

class QuestionMetadata(BaseModel):
    exam: str = Field(description="Target examination, e.g., UPSC, SSC CGL")
    subject: str = Field(description="Broad subject, e.g., Economy, Science & Tech")
    topic: str
    sub_topic: str
    
    # NEW: The Escape Hatch Tracker
    taxonomy_source: str = Field(default="official", description="'official' if from S3 JSON, 'llm_generated' if invented via escape hatch")
    
    cognitive_skill: str = Field(description="e.g., Factual Recall, Analytical Reasoning")
    difficulty_level: int = Field(ge=1, le=5, description="1 (Beginner) to 5 (Exam-Ready)")
    ttl_days: Optional[int] = Field(default=None, description="Time-to-live in days. None for static subjects.")
    generation_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Question(BaseModel):
    id: str
    shared_context: Optional[str] = Field(default=None, description="The shared passage or dataset if this question belongs to a group. Otherwise null.")
    text: str = Field(description="The question text. MUST use $ for inline math (e.g. $V$) and $$ for block math.")
    options: Dict[str, str] = Field(description="e.g., {'A': '...', 'B': '...'}. MUST use $ for math.")
    correct_answer: str
    explanation: str = Field(description="Detailed explanation. MUST use $ for inline math and $$ for block math.")
    metadata: QuestionMetadata

class StudentProfile(BaseModel):
    student_id: str
    target_exam: str
    tests_taken: int = 0
    overall_readiness_score: float = 0.0
    
    explored_topics: List[str] = Field(default_factory=list) 
    
    # UPGRADED: Structured Analytics Array
    proficiencies: List[ProficiencyRecord] = Field(default_factory=list)
    
    seen_question_counts: Dict[str, int] = {}
    last_study_plan: Optional[str] = None

class TestConfig(BaseModel):
    target_subject: str = "All Syllabus"
    target_topic: str = "All Syllabus" 
    
    # 👉 THE FIX: Changed to Optional and defaults to None for Auto-Mode
    target_difficulty: Optional[int] = None 
    
    num_questions: int = 50
    adaptive_mode: bool = True
    override_topics: Optional[List[str]] = None          

# ==========================================
# 2. THE BLUEPRINT (New Planner Architecture)
# ==========================================

class BlueprintRequirement(BaseModel):
    # UPGRADED: Forcing the Planner to output the strict 3-tier hierarchy
    subject: str = Field(description="The broad subject area.")
    topic: str = Field(description="The specific topic.")
    sub_topic: str = Field(description="The granular sub-topic to test.")
    
    quantity: int = Field(description="Number of questions for this node.")
    target_difficulty: int = Field(description="Difficulty level 1-5 based on student history.")
    question_type: str = Field(default="Standard", description="'Standard', 'Reading Comprehension', 'Data Interpretation', etc.")
    requires_shared_context: bool = Field(default=False, description="True if these questions must share a single comprehensive passage, dataset, or case-study.")
    reasoning: str = Field(description="Why the planner chose this specific configuration.")

class TestBlueprint(BaseModel):
    overall_strategy: str = Field(description="A brief summary of the pedagogical strategy for this test.")
    requirements: List[BlueprintRequirement]

# ==========================================
# 3. GENERATION STATE (Test Building Workflow)
# ==========================================

class AdaptiveTestState(TypedDict):
    profile: StudentProfile
    config: TestConfig
    blueprint: Optional[TestBlueprint]
    
    current_question_index: int
    generation_attempts: int
    
    selected_questions: Annotated[List[Question], operator.add]
    draft_batch: List[Question]
    rejected_batch: List[Dict[str, str]] 
    current_batch_target: int
    
    exploitation_topics: List[str]
    exploration_topics: List[str] 
    
# ==========================================
# 4. EVALUATION STATE (Post-Test Workflow)
# ==========================================

class EvaluationState(TypedDict):
    profile: StudentProfile
    questions: List[Question]
    student_answers: Dict[str, str]  
    graded_results: List[Dict[str, Any]] 
    score_percentage: float
    study_plan: str