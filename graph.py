import os
import json
import math
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from schema import AdaptiveTestState, Question
from vector_store import retrieve_best_question, save_questions_to_db, semantic_snap_topic
from db import save_student_profile
from datetime import datetime

current_year = datetime.now().year

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# AGENT NODES
# ==========================================

def orchestrator_node(state: AdaptiveTestState) -> dict:
    profile = state["profile"]
    config = state["config"]
    
    exploitation_topic = state.get("exploitation_topic", "")
    exploration_topic = state.get("exploration_topic", "")
    
    # Run the adaptive discovery logic only on the very first pass
    if config.adaptive_mode and not exploitation_topic:
        
        # 1. EXPLOITATION (80%): Find the weakest known topic
        if profile.topic_proficiencies:
            exploitation_topic = min(profile.topic_proficiencies, key=profile.topic_proficiencies.get)
            print(f"🧠 Orchestrator [EXPLOIT]: Weakest known area -> {exploitation_topic}")
        else:
            # Cold start: They have no profile yet. Use the requested subject as a broad starting point.
            exploitation_topic = config.target_subject
            print(f"🧠 Orchestrator [COLD START]: No profile found. Starting with broad subject -> {exploitation_topic}")

        # 2. EXPLORATION (20%): Discover uncharted territory
        known_topics_str = ", ".join(profile.explored_topics) if profile.explored_topics else "None"
        
        exploration_prompt = f"""
        Analyze the {profile.target_exam} syllabus for the subject: {config.target_subject}.
        The student has already been tested on these topics: [{known_topics_str}].
        
        Identify exactly ONE major topic or sub-topic from the {config.target_subject} syllabus that the student has NOT been tested on yet.
        Respond ONLY with the name of the topic. Do not include any other text, reasoning, or punctuation.
        """
        
        try:
            print("🔭 Orchestrator: Asking Gemini to chart new territory...")
            response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=exploration_prompt,
                config=types.GenerateContentConfig(temperature=0.7) # Higher temp for diverse discovery
            )
            raw_new_topic = response.text.strip()
            
            # 3. SEMANTIC SNAPPING: Ensure it's not a duplicate
            exploration_topic = semantic_snap_topic(
                new_topic=raw_new_topic, 
                existing_topics=profile.explored_topics
            )
            
            # 4. PROFILE INJECTION: Add it to the profile with a baseline score if it's genuinely new
            if exploration_topic not in profile.explored_topics:
                profile.explored_topics.append(exploration_topic)
                profile.topic_proficiencies[exploration_topic] = 0.5 # Average Baseline
                print(f"✨ Orchestrator: Injected new topic '{exploration_topic}' into profile with baseline 0.5")
                save_student_profile(profile)

        except Exception as e:
            print(f"⚠️ Orchestrator Exploration failed: {e}. Defaulting entirely to exploitation.")
            exploration_topic = exploitation_topic # Fallback to 100% exploitation

    # Fallback for manual mode
    if not exploitation_topic:
        exploitation_topic = config.target_topic
        print(f"🎯 Orchestrator [MANUAL]: Targeting -> {exploitation_topic}")

    current_count = len(state.get("selected_questions", []))
    target_count = config.num_questions
    deficit = target_count - current_count
    
    print(f"\n📊 Orchestrator: {current_count}/{target_count} questions ready. Deficit: {deficit}")
    
    if deficit <= 0:
        return {"current_batch_target": 0, "config": config}
        
    batch_size = min(deficit, 5) 
    
    return {
        "current_batch_target": batch_size,
        "generation_attempts": 0,
        "draft_batch": [],
        "rejected_batch": [],
        "config": config,
        "exploitation_topic": exploitation_topic,
        "exploration_topic": exploration_topic,
        "profile": profile 
    }

def database_retriever_node(state: AdaptiveTestState) -> dict:
    target = state.get("current_batch_target", 0)
    if target <= 0:
        return {}
        
    print(f"🔍 Retriever: Searching DB for up to {target} questions...")
    found_questions = []
    
    existing_ids = [existing.id for existing in state.get("selected_questions", [])]
    seen_history = list(state["profile"].seen_question_counts.keys())
    
    exploitation_topic = state.get("exploitation_topic", state["config"].target_topic)
    exploration_topic = state.get("exploration_topic", "")
    
    # Calculate the 80/20 split for retrieval
    num_explore = 0
    num_exploit = target
    
    if exploration_topic and exploration_topic != exploitation_topic and target >= 2:
        num_explore = math.ceil(target * 0.20)
        num_exploit = target - num_explore
        
    # --- Search 1: Exploitation Topic (80%) ---
    print(f"   -> Looking for {num_exploit} questions on '{exploitation_topic}'")
    for _ in range(num_exploit):
        current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
        q = retrieve_best_question(
            target_exam=state["profile"].target_exam,
            target_subject=state["config"].target_subject,
            target_topic=exploitation_topic,  
            target_difficulty=state["config"].target_difficulty,
            student_profile=state["profile"],
            exclude_ids=current_exclude_ids
        )
        if q:
            found_questions.append(q)
        else:
            break # DB doesn't have enough, Generator will fill the gap
            
    # --- Search 2: Exploration Topic (20%) ---
    if num_explore > 0:
        print(f"   -> Looking for {num_explore} questions on '{exploration_topic}'")
        for _ in range(num_explore):
            current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
            q = retrieve_best_question(
                target_exam=state["profile"].target_exam,
                target_subject=state["config"].target_subject,
                target_topic=exploration_topic,  
                target_difficulty=state["config"].target_difficulty,
                student_profile=state["profile"],
                exclude_ids=current_exclude_ids
            )
            if q:
                found_questions.append(q)
            else:
                break # DB doesn't have enough, Generator will fill the gap
            
    print(f"✅ Retriever: Found {len(found_questions)} suitable questions in DB.")
    
    new_target = target - len(found_questions)
    return {
        "selected_questions": found_questions, 
        "current_batch_target": new_target     
    }

def generator_node(state: AdaptiveTestState) -> dict:
    target = state.get("current_batch_target", 0)
    rejected = state.get("rejected_batch", [])
    
    num_to_generate = len(rejected) if rejected else target
    
    if num_to_generate <= 0:
         return {"draft_batch": []}
         
    # --- The 80/20 Generation Logic ---
    exploitation_topic = state.get("exploitation_topic", state["config"].target_topic)
    exploration_topic = state.get("exploration_topic", "")
    
    # Calculate how many questions go to the weak area (80%) and the new area (20%)
    if exploration_topic and exploration_topic != exploitation_topic and num_to_generate >= 2:
        num_explore = math.ceil(num_to_generate * 0.20) # At least 1 question if target >= 2
        num_exploit = num_to_generate - num_explore
        composition_instruction = f"""
        You must generate exactly {num_to_generate} questions. 
        Allocate the topics strictly as follows:
        - {num_exploit} question(s) explicitly covering: {exploitation_topic}
        - {num_explore} question(s) explicitly covering: {exploration_topic}
        """
        print(f"✍️ Generator: Drafting {num_exploit} on '{exploitation_topic}', {num_explore} on '{exploration_topic}'...")
    else:
        # If target is 1, or exploration failed, dedicate 100% to the weak area
        composition_instruction = f"""
        You must generate exactly {num_to_generate} questions.
        All questions must explicitly cover: {exploitation_topic}
        """
        print(f"✍️ Generator: Drafting {num_to_generate} questions on '{exploitation_topic}'...")

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
    Generate a JSON array containing EXACTLY {num_to_generate} highly accurate {state['profile'].target_exam} Prelims questions.
    
    {composition_instruction}
    
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
            "exam": "{state['profile'].target_exam}", "subject": "{state['config'].target_subject}", 
            "topic": "The exact topic this question covers", 
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

def saver_node(state: AdaptiveTestState) -> dict:
    """Saves approved questions to the OpenSearch Vector Database."""
    approved_questions = state.get("selected_questions", [])
    drafts = state.get("draft_batch", [])
    draft_ids = [d.id for d in drafts]
    
    # Only save NEWly generated questions, not ones retrieved from DB
    new_approved = [q for q in approved_questions if q.id in draft_ids]
    
    if new_approved:
        print(f"💾 Saver: Pushing {len(new_approved)} new questions to OpenSearch Vector DB...")
        save_questions_to_db(new_approved)
        
    return {}

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

def route_after_critic(state: AdaptiveTestState) -> Literal["saver", "generator"]:
    if state.get("rejected_batch"):
        return "generator"
    return "saver"

# ==========================================
# BUILD THE LANGGRAPH
# ==========================================

workflow = StateGraph(AdaptiveTestState)

workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("database_retriever", database_retriever_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)
workflow.add_node("saver", saver_node) 

workflow.set_entry_point("orchestrator")

workflow.add_conditional_edges("orchestrator", route_after_orchestrator)
workflow.add_conditional_edges("database_retriever", route_after_retriever)
workflow.add_edge("generator", "critic")
workflow.add_conditional_edges("critic", route_after_critic) 
workflow.add_edge("saver", "orchestrator") 

app = workflow.compile()