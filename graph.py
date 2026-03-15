import os
import json
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from schema import AdaptiveTestState, Question
from vector_store import retrieve_best_question

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# AGENT NODES
# ==========================================

def orchestrator_node(state: AdaptiveTestState) -> dict:
    """Calculates the deficit and sets the batch target."""
    current_count = len(state.get("selected_questions", []))
    target_count = state["config"].num_questions
    deficit = target_count - current_count
    
    print(f"\n📊 Orchestrator: {current_count}/{target_count} questions ready. Deficit: {deficit}")
    
    if deficit <= 0:
        return {"current_batch_target": 0}
        
    # Process in chunks of 5 to keep search grounding deep and accurate
    batch_size = min(deficit, 5) 
    
    return {
        "current_batch_target": batch_size,
        "generation_attempts": 0,
        "draft_batch": [],
        "rejected_batch": []
    }

def database_retriever_node(state: AdaptiveTestState) -> dict:
    """Attempts to fill the current batch target from the Vector DB."""
    target = state.get("current_batch_target", 0)
    if target == 0:
        return {}
        
    print(f"🔍 Retriever: Searching DB for up to {target} questions...")
    found_questions = []
    
    # Get IDs of questions already in the final test
    existing_ids = [existing.id for existing in state.get("selected_questions", [])]
    
    for _ in range(target):
        # Combine existing test IDs with the ones we JUST found in this loop
        current_exclude_ids = existing_ids + [fq.id for fq in found_questions]
        
        q = retrieve_best_question(
            target_exam=state["profile"].target_exam,
            target_subject=state["config"].target_subject,
            target_topic="Monetary Policy", 
            target_difficulty=state["config"].target_difficulty,
            student_profile=state["profile"],
            exclude_ids=current_exclude_ids  # <--- Pass the dynamic list here
        )
        
        if q:
            found_questions.append(q)
        else:
            # If the DB returns None, we've exhausted all good DB questions.
            # Break the loop early and let the LLM generate the rest!
            break 
            
    print(f"✅ Retriever: Found {len(found_questions)} suitable questions in DB.")
    
    new_target = target - len(found_questions)
    return {
        "selected_questions": found_questions, 
        "current_batch_target": new_target     
    }

def generator_node(state: AdaptiveTestState) -> dict:
    """Generates a batch of questions or fixes rejected ones."""
    target = state.get("current_batch_target", 0)
    rejected = state.get("rejected_batch", [])
    
    # If we have rejected questions, we only regenerate those. Otherwise, we generate new ones.
    num_to_generate = len(rejected) if rejected else target
    
    if num_to_generate <= 0:
         return {"draft_batch": []}
         
    print(f"✍️ Generator: Drafting batch of {num_to_generate} questions via Gemini...")
    
    topic = "Monetary Policy"
    
    # --- NEW: Concept Collision Logic ---
    # Gather the text of questions already in the test (from DB and previous batches)
    existing_questions = state.get("selected_questions", [])
    
    collision_context = ""
    if existing_questions:
        collision_context = "CRITICAL AVOIDANCE INSTRUCTIONS:\nThe test currently includes the following questions. You MUST NOT generate questions that test the exact same factual concepts, mechanisms, or angles as these. Explore different facets, historical context, or broader impacts within the topic.\n\n"
        for i, eq in enumerate(existing_questions, 1):
            collision_context += f"Existing Q{i}: {eq.text}\n"

    # Inject specific feedback if we are fixing broken questions
    feedback_context = ""
    if rejected:
        feedback_context = "CRITICAL: Fix the following rejected questions based on this feedback:\n"
        for r in rejected:
            feedback_context += f"- ID {r['id']}: {r['feedback']}\n"
            
    prompt = f"""
    Generate a JSON array containing EXACTLY {num_to_generate} highly accurate UPSC Prelims questions about {topic}.
    
    {collision_context}
    
    {feedback_context}
    
    REQUIREMENTS:
    - Ground facts using the latest internet data.
    - Output strictly as a JSON array of objects matching this schema:
    [
      {{
        "id": "q_gen_unique_id",
        "text": "Question text...",
        "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
        "correct_answer": "A",
        "explanation": "Detailed explanation...",
        "metadata": {{
            "exam": "UPSC", "subject": "Economy", "topic": "Monetary Policy", 
            "sub_topic": "Repo Rate", "cognitive_skill": "Analytical", 
            "difficulty_level": 4, "ttl_days": 180
        }}
      }}
    ]
    """
    
    response = client.models.generate_content(
        model='gemini-3.1-pro-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[{"google_search": {}}],
            temperature=0.2,
            response_mime_type="application/json", 
        )
    )
    
    try:
        raw_json_array = json.loads(response.text)
        new_batch = [Question(**q) for q in raw_json_array]
        return {
            "draft_batch": new_batch, 
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "rejected_batch": [] # Clear old rejections
        }
    except Exception as e:
        print(f"❌ Generator format error: {e}")
        # Force a retry on the whole batch if JSON parsing fails entirely
        return {"generation_attempts": state.get("generation_attempts", 0) + 1}

def critic_node(state: AdaptiveTestState) -> dict:
    """Evaluates the batch, separating approvals from rejections."""
    drafts = state.get("draft_batch", [])
    attempts = state.get("generation_attempts", 0)
    
    if not drafts:
        return {}
        
    print(f"🧐 Critic: Evaluating batch of {len(drafts)} questions...")
    
    approved = []
    rejected = []
    
    for draft in drafts:
        # Mock Evaluation Logic
        if "obvious" in draft.text.lower(): 
            rejected.append({"id": draft.id, "feedback": "Distractors are too obvious."})
        else:
            approved.append(draft)
            
    print(f"✅ Critic: {len(approved)} approved, {len(rejected)} rejected.")
    
    if attempts >= 3 and rejected:
        print("🛑 Critic: Max attempts reached. Dropping failed questions from this batch.")
        rejected = [] 
        
    return {
        "selected_questions": approved,
        "rejected_batch": rejected,
        "current_batch_target": len(rejected) 
    }

# ==========================================
# CONDITIONAL ROUTING EDGES
# ==========================================

def route_after_orchestrator(state: AdaptiveTestState) -> Literal["database_retriever", "__end__"]:
    if state.get("current_batch_target", 0) <= 0:
        return "__end__"
    return "database_retriever"

def route_after_retriever(state: AdaptiveTestState) -> Literal["orchestrator", "generator"]:
    if state.get("current_batch_target", 0) > 0:
        return "generator"
    return "orchestrator"

def route_after_critic(state: AdaptiveTestState) -> Literal["orchestrator", "generator"]:
    if state.get("rejected_batch"):
        return "generator"
    return "orchestrator"

# ==========================================
# BUILD THE LANGGRAPH
# ==========================================

workflow = StateGraph(AdaptiveTestState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("database_retriever", database_retriever_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges("orchestrator", route_after_orchestrator)
workflow.add_conditional_edges("database_retriever", route_after_retriever)
workflow.add_edge("generator", "critic")
workflow.add_conditional_edges("critic", route_after_critic)

app = workflow.compile()