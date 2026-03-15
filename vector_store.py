import math
from datetime import datetime, timezone
from typing import List, Optional
from schema import Question, QuestionMetadata, StudentProfile

# ==========================================
# 1. MOCK DATABASE
# ==========================================
# In production, this will be your PostgreSQL (pgvector) or Pinecone DB.
MOCK_QUESTION_BANK: List[Question] = [
    Question(
        id="q_econ_001",
        text="Consider the following statements regarding the RBI's Standing Deposit Facility (SDF)...",
        options={"A": "1 only", "B": "2 only", "C": "Both 1 and 2", "D": "Neither 1 nor 2"},
        correct_answer="A",
        explanation="SDF allows RBI to absorb liquidity without providing collateral...",
        metadata=QuestionMetadata(
            exam="UPSC",
            subject="Economy",
            topic="Monetary Policy",
            sub_topic="Liquidity Management",
            cognitive_skill="Conceptual Clarity",
            difficulty_level=4,
            ttl_days=180, # High volatility, ages out in 6 months
            generation_date=datetime(2025, 12, 1, tzinfo=timezone.utc)
        )
    ),
    Question(
        id="q_hist_001",
        text="Who among the following was the founder of the broadly localized hyper-transit movements in colonial India?",
        options={"A": "Leader A", "B": "Leader B", "C": "Leader C", "D": "Leader D"},
        correct_answer="C",
        explanation="Historical context...",
        metadata=QuestionMetadata(
            exam="UPSC",
            subject="History",
            topic="Modern India",
            sub_topic="Pre-Independence",
            cognitive_skill="Factual Recall",
            difficulty_level=3,
            ttl_days=None, # Static subject, never decays
            generation_date=datetime(2024, 1, 1, tzinfo=timezone.utc)
        )
    )
]

# ==========================================
# 2. THE HYBRID SEARCH ENGINE
# ==========================================

def retrieve_best_question(
    target_exam: str,
    target_subject: str,
    target_topic: str,
    target_difficulty: int,
    student_profile: StudentProfile,
    exclude_ids: List[str],  # <--- NEW: List of IDs to ignore
    threshold: float = 0.6
) -> Optional[Question]:
    
    best_question = None
    highest_score = -float('inf')
    
    ALPHA = 1.0   
    BETA = 0.5    
    GAMMA = 0.4   
    
    current_time = datetime.now(timezone.utc)
    
    for q in MOCK_QUESTION_BANK:
        # <--- NEW: Skip if we already grabbed it in this batch or previous batches
        if q.id in exclude_ids:
            continue
            
        # 1. HARD FILTERS (Must Match)
        if q.metadata.exam != target_exam or q.metadata.subject != target_subject:
            continue
            
        # Allow a slight variance in difficulty (+/- 1 level)
        if abs(q.metadata.difficulty_level - target_difficulty) > 1:
            continue
            
        # ... (The rest of the scoring logic remains exactly the same) ...
        # 2. BASE MATCH SCORE
        semantic_score = 1.0 if q.metadata.topic == target_topic else 0.8
        
        # 3. RECENCY DECAY
        recency_score = 1.0
        if q.metadata.ttl_days:
            age_days = (current_time - q.metadata.generation_date).days
            if age_days > q.metadata.ttl_days:
                decay_rate = 0.05 
                recency_score = math.exp(-decay_rate * (age_days - q.metadata.ttl_days))
        
        # 4. EXPOSURE PENALTY
        times_seen = student_profile.seen_question_counts.get(q.id, 0)
        
        # 5. FINAL CALCULATION
        final_score = (ALPHA * semantic_score) + (BETA * recency_score) - (GAMMA * times_seen)
        
        if final_score > highest_score:
            highest_score = final_score
            best_question = q

    if highest_score >= threshold:
        return best_question
    
    return None