from typing import TypedDict, Annotated, List, Dict, Any
import operator

class UPSCState(TypedDict):
    # --- 1. Student Identity & History (For persistent memory) ---
    student_id: str
    is_returning_student: bool
    historical_readiness_score: float
    past_weaknesses: List[str]
    
    # --- 2. Current Test Parameters (What the user configures) ---
    subject: str                  # e.g., "Economy", "All Syllabus"
    recency: str                  # e.g., "Last 3 months", "Last 1 year"
    num_questions: int            # Total questions requested (n)
    
    # --- 3. Internal Agent Workspace (Drafting & Reviewing) ---
    current_question_index: int   # To track generation progress
    # Annotated with operator.add means each time an agent returns questions, 
    # they are appended to this list, not overwritten.
    drafted_questions: Annotated[List[Dict[str, Any]], operator.add]
    critic_feedback: str          # Feedback from Critic to Generator if a question fails QA
    generation_attempts: int      # To prevent infinite loops between Generator and Critic
    
    # --- 4. Evaluation & Outputs (After the student takes the test) ---
    student_answers: Dict[int, str] # e.g., {0: "C", 1: "A", 2: "D"}
    test_analysis: str            # Breakdown of confusing areas
    study_plan: str               # The feasible plan generated for the student