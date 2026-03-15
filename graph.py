import os
import json
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from schema import AdaptiveTestState, Question
from vector_store import retrieve_best_question
from datetime import datetime
current_year = datetime.now().year

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# AGENT NODES
# ==========================================

def orchestrator_node(state: AdaptiveTestState) -> dict:
    """Calculates the deficit, analyzes history, and sets the batch target."""
    profile = state["profile"]
    config = state["config"]
    
    # --- NEW: Adaptive Logic vs Override Logic ---
    if config.adaptive_mode and profile.topic_proficiencies:
        # Find the topic with the lowest historical proficiency score
        weakest_topic = min(profile.topic_proficiencies, key=profile.topic_proficiencies.get)
        print(f"🧠 Orchestrator [ADAPTIVE]: Identified weakest area -> {weakest_topic}")
        config.target_topic = weakest_topic 
    else:
        print(f"🎯 Orchestrator [MANUAL/DEFAULT]: Targeting -> {config.target_topic}")

    current_count = len(state.get("selected_questions", []))
    target_count = config.num_questions
    deficit = target_count - current_count
    
    print(f"\n📊 Orchestrator: {current_count}/{target_count} questions ready. Deficit: {deficit}")
    
    if deficit <= 0:
        return {"current_batch_target": 0, "config": config}
        
    # Process in chunks of 5 to keep search grounding deep and accurate
    batch_size = min(deficit, 5) 
    
    return {
        "current_batch_target": batch_size,
        "generation_attempts": 0,
        "draft_batch": [],
        "rejected_batch": [],
        "config": config  # Pass the updated config forward
    }

def database_retriever_node(state: AdaptiveTestState) -> dict:
    """Attempts to fill the current batch target from the Vector DB."""
    target = state.get("current_batch_target", 0)
    if target == 0:
        return {}
        
    print(f"🔍 Retriever: Searching DB for up to {target} questions...")
    found_questions = []
    
    existing_ids = [existing.id for existing in state.get("selected_questions", [])]
    
    for _ in range(target):
        current_exclude_ids = existing_ids + [fq.id for fq in found_questions]
        
        q = retrieve_best_question(
            target_exam=state["profile"].target_exam,
            target_subject=state["config"].target_subject,
            target_topic=state["config"].target_topic,  # <-- NEW: Dynamic Topic
            target_difficulty=state["config"].target_difficulty,
            student_profile=state["profile"],
            exclude_ids=current_exclude_ids
        )
        
        if q:
            found_questions.append(q)
        else:
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
    
    num_to_generate = len(rejected) if rejected else target
    
    if num_to_generate <= 0:
         return {"draft_batch": []}
         
    topic = state["config"].target_topic  # <-- NEW: Dynamic Topic
    print(f"✍️ Generator: Drafting batch of {num_to_generate} questions on '{topic}' via Gemini...")
    
    existing_questions = state.get("selected_questions", [])
    
    collision_context = ""
    if existing_questions:
        collision_context = "CRITICAL AVOIDANCE INSTRUCTIONS:\nThe test currently includes the following questions. You MUST NOT generate questions that test the exact same factual concepts, mechanisms, or angles as these. Explore different facets, historical context, or broader impacts within the topic.\n\n"
        for i, eq in enumerate(existing_questions, 1):
            collision_context += f"Existing Q{i}: {eq.text}\n"

    feedback_context = ""
    if rejected:
        feedback_context = "CRITICAL: Fix the following rejected questions based on this feedback:\n"
        for r in rejected:
            feedback_context += f"- ID {r['id']}: {r['feedback']}\n"
            
    prompt = f"""
    Generate a JSON array containing EXACTLY {num_to_generate} highly accurate {state['profile'].target_exam} Prelims questions about {topic}.
    
    CURRENT YEAR: {current_year-1} to {current_year}
    
    {collision_context}
    
    {feedback_context}
    
    REQUIREMENTS:
    - CRITICAL: You are equipped with a Google Search tool. You MUST use it to prioritize current affairs, newly passed legislations, recent supreme court judgments, and scientific developments from {current_year - 1} and {current_year}.
    - Do not generate outdated questions based solely on your internal training data.
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
            "exam": "{state['profile'].target_exam}", "subject": "{state['config'].target_subject}", "topic": "{topic}", 
            "sub_topic": "Specific Sub-Topic Here", "cognitive_skill": "Analytical", 
            "difficulty_level": {state['config'].target_difficulty}, "ttl_days": 180
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
            "rejected_batch": [] 
        }
    except Exception as e:
        print(f"❌ Generator format error: {e}")
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