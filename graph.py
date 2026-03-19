import os
import json
import math
import uuid
import re
import functools
from typing import Literal
from dotenv import load_dotenv
from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from schema import AdaptiveTestState, Question, TestBlueprint, BlueprintRequirement, QuestionMetadata
from vector_store import retrieve_best_question, save_questions_to_db, semantic_snap_topic
from db import save_student_profile
from datetime import datetime

current_year = datetime.now().year

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# 0. CONFIGURATION & UTILITIES
# ==========================================

@functools.lru_cache(maxsize=1)
def load_syllabus_map() -> dict:
    """
    Fetches the syllabus map. Cached in memory for warm AWS Lambdas.
    In production, swap this to boto3 to fetch from s3://my-edtech-configs/syllabus_maps.json
    """
    try:
        with open("syllabus_maps.json", "r", encoding="utf-8") as f:
            print("📦 Loading syllabus_maps.json into memory...")
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ syllabus_maps.json not found! System will rely purely on the LLM Escape Hatch.")
        return {}

def clean_json_response(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ==========================================
# 1. PLANNER NODE (Strategic Mastermind)
# ==========================================

def planner_node(state: AdaptiveTestState) -> dict:
    profile = state["profile"]
    config = state["config"]
    blueprint = state.get("blueprint")
    
    # 1. Load and Slice the JSON Map to save Tokens!
    full_syllabus_map = load_syllabus_map()
    exam_map = full_syllabus_map.get(profile.target_exam, {})
    
    if not blueprint:
        print(f"🧭 Planner: Assembling strict 3-tier blueprint for {profile.target_exam}...")
        
        # 2. Extract History from the NEW Structured Analytics Schema
        explored_subtopics = profile.explored_topics
        sorted_profs = sorted(profile.proficiencies, key=lambda x: x.score)
        weakest_subtopics = [p.sub_topic for p in sorted_profs[:3]] if sorted_profs else []
        
        planner_context = f"""
        OFFICIAL EXAM SYLLABUS MAP:
        {json.dumps(exam_map, indent=2) if exam_map else "No official map provided. Rely on your knowledge."}
        
        STUDENT HISTORY:
        - Weakest Sub-Topics (Needs Exploitation): {weakest_subtopics if weakest_subtopics else "None yet"}
        - Already Explored Sub-Topics: {explored_subtopics if explored_subtopics else "None yet"}
        """
        
        if config.adaptive_mode:
            planner_context += f"""
            ADAPTIVE RULES:
            - Target EXACTLY {config.num_questions} questions.
            - EXPLOITATION (~80%): Assign questions to the student's Weakest Sub-Topics.
            - EXPLORATION (~20%): Discover 1 or 2 brand new Sub-Topics from the Official Map that the student has NOT explored yet. 
            - ESCAPE HATCH: If the requested target ({config.target_subject}) is missing from the Official Map, you may invent a relevant Topic and Sub-Topic, but you MUST maintain the strict Subject->Topic->Sub-Topic structure.
            """
        else:
            planner_context += f"""
            MOCK EXAM / MANUAL RULES:
            - Target EXACTLY {config.num_questions} questions.
            - If the target is "All Syllabus", you MUST distribute the questions proportionally across at least 5 DIFFERENT Topics/Sub-Topics from the Official Map to create a realistic mock exam.
            - Target focus: Subject: {config.target_subject}, Topic: {config.target_topic}.
            """

        planner_prompt = f"""
        You are the Master Test Planner for the {profile.target_exam} exam.
        Your task is to create a Test Blueprint.
        
        {planner_context}
        
        Requested Base Difficulty: {config.target_difficulty} (Scale 1-5).
        
        Determine if a topic requires standard discrete questions, or if it requires grouped context (like Reading Comprehension, Data Interpretation, or Case Studies). 
        If it requires grouped context, set "requires_shared_context" to true.
        
        Output MUST be a valid JSON object matching this strict 3-tier structure:
        {{
            "overall_strategy": "Brief explanation",
            "requirements": [
                {{
                    "subject": "General Awareness",
                    "topic": "Economy",
                    "sub_topic": "Inflation",
                    "quantity": 3,
                    "target_difficulty": 4,
                    "question_type": "Standard",
                    "requires_shared_context": false,
                    "reasoning": "Targeting student weakness"
                }}
            ]
        }}
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=planner_prompt,
                config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json")
            )
            blueprint = TestBlueprint(**json.loads(clean_json_response(response.text)))
            
            # Math Validation Guardrail
            total_reqs = sum(req.quantity for req in blueprint.requirements)
            if total_reqs != config.num_questions:
                print(f"⚠️ Planner Math Correction: Forcing {total_reqs} to {config.num_questions}.")
                if blueprint.requirements:
                    blueprint.requirements[0].quantity += (config.num_questions - total_reqs)
                    
            print(f"📋 Blueprint Created: {blueprint.overall_strategy}")
        except Exception as e:
            print(f"❌ Planner failed: {e}. Falling back.")
            blueprint = TestBlueprint(
                overall_strategy="Fallback",
                requirements=[BlueprintRequirement(subject=config.target_subject, topic=config.target_topic, sub_topic="General", quantity=config.num_questions, target_difficulty=config.target_difficulty, reasoning="Fallback", question_type="Standard", requires_shared_context=False)]
            )

    current_count = len(state.get("selected_questions", []))
    deficit = config.num_questions - current_count
    print(f"\n📊 Planner: {current_count}/{config.num_questions} questions ready. Deficit: {deficit}")
    
    if deficit <= 0:
        return {"current_batch_target": 0, "blueprint": blueprint}
        
    return {
        "current_batch_target": min(deficit, 5), 
        "draft_batch": [],
        "rejected_batch": [],
        "blueprint": blueprint
    }

# ==========================================
# 2. RETRIEVER NODE (The Librarian)
# ==========================================

def database_retriever_node(state: AdaptiveTestState) -> dict:
    target = state.get("current_batch_target", 0)
    blueprint = state.get("blueprint")
    if target <= 0 or not blueprint:
        return {}
        
    print(f"🔍 Retriever: Searching DB based on Blueprint Requirements...")
    found_questions = []
    existing_ids = [q.id for q in state.get("selected_questions", [])]
    seen_history = list(state["profile"].seen_question_counts.keys())
    
    existing_counts = {}
    for q in state.get("selected_questions", []):
        existing_counts[q.metadata.sub_topic] = existing_counts.get(q.metadata.sub_topic, 0) + 1

    for req in blueprint.requirements:
        # BYPASS RETRIEVAL FOR SHARED CONTEXT (To prevent fracturing)
        if req.requires_shared_context:
            continue
            
        needed_for_topic = req.quantity - existing_counts.get(req.sub_topic, 0)
        for _ in range(needed_for_topic):
            if len(found_questions) >= target:
                break 
                
            current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
            q = retrieve_best_question(
                target_exam=state["profile"].target_exam,
                target_subject=req.subject,
                target_topic=req.topic,  
                target_difficulty=req.target_difficulty,
                student_profile=state["profile"],
                exclude_ids=current_exclude_ids
            )
            if q:
                found_questions.append(q)
            else:
                break 
            
    print(f"✅ Retriever: Found {len(found_questions)} suitable questions in DB.")
    return {
        "selected_questions": found_questions, 
        "current_batch_target": target - len(found_questions)     
    }

# ==========================================
# 3. GENERATOR NODE (The Specialist)
# ==========================================

def generator_node(state: AdaptiveTestState) -> dict:
    target = state.get("current_batch_target", 0)
    rejected = state.get("rejected_batch", [])
    blueprint = state.get("blueprint")
    attempts = state.get("generation_attempts", 0)
    
    if target <= 0 and not rejected:
         return {"draft_batch": []}
         
    if attempts >= 3:
        print("🛑 Generator Circuit Breaker Activated.")
        return {"draft_batch": [], "rejected_batch": [], "generation_attempts": attempts}

    batch_context = ""
    
    if rejected:
        batch_context += "REGENERATING REJECTED QUESTIONS:\n"
        for r in rejected:
            batch_context += f"- Sub-Topic: {r.get('sub_topic', 'Unknown')} | Feedback: {r['feedback']}\n"
        num_to_generate = len(rejected)
    else:
        existing_counts = {}
        for q in state.get("selected_questions", []):
            existing_counts[q.metadata.sub_topic] = existing_counts.get(q.metadata.sub_topic, 0) + 1
            
        batch_context += "CURRENT BATCH REQUIREMENTS:\n"
        slots_left = target
        actual_take_sum = 0
        
        for req in blueprint.requirements:
            needed = req.quantity - existing_counts.get(req.sub_topic, 0)
            if needed > 0 and slots_left > 0:
                take = needed if req.requires_shared_context else min(needed, slots_left)
                batch_context += f"- Subject: {req.subject} | Topic: {req.topic} | Sub-Topic: {req.sub_topic} | Difficulty: {req.target_difficulty} | Shared Context Required: {req.requires_shared_context} | Quantity: {take}\n"
                slots_left -= take
                actual_take_sum += take

        num_to_generate = actual_take_sum
        
    if num_to_generate <= 0:
        return {"draft_batch": []}

    prompt = f"""
    You are the Senior Content Creator for the {state['profile'].target_exam} exam.
    Generate EXACTLY {num_to_generate} questions.
    
    {batch_context}
    
    SHARED CONTEXT RULES:
    If a requirement specifies "Shared Context Required: True", you MUST generate ONE shared passage PER TOPIC and place it inside the "shared_context" field for EVERY related question.
    
    REQUIREMENTS:
    - Output strictly as a JSON array of objects matching this exact schema:
    [
      {{
        "shared_context": "The 500-word shared passage goes here if needed. Otherwise, null.",
        "text": "The specific question prompt goes here...",
        "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
        "correct_answer": "A",
        "explanation": "Detailed explanation...",
        "metadata": {{
            "subject": "Exact Subject from requirements", 
            "topic": "Exact Topic from requirements", 
            "sub_topic": "Exact Sub-Topic from requirements", 
            "cognitive_skill": "Analytical", 
            "difficulty_level": 3
        }}
      }}
    ]
    """
    
    print(f"✍️ Generator: Drafting {num_to_generate} questions (Attempt {attempts + 1})...")
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, response_mime_type="application/json")
        )
        
        raw_json_array = json.loads(clean_json_response(response.text))
        
        # --- THE PYTHON MAGIC: IDs, Normalization, and Escape Hatch Tracking ---
        full_syllabus_map = load_syllabus_map()
        exam_map = full_syllabus_map.get(state["profile"].target_exam, {})
        
        new_batch = []
        for q_dict in raw_json_array:
            meta = q_dict["metadata"]
            
            # 1. Deterministic Python IDs
            safe_exam = state["profile"].target_exam.replace(" ", "")[:4]
            safe_subj = meta.get("subject", "Subj").replace(" ", "")[:4]
            q_dict["id"] = f"q_{safe_exam}_{safe_subj}_{uuid.uuid4().hex[:8]}"
            
            # 2. String Normalization (Case-Sensitivity Fix)
            sub_topic = str(meta.get("sub_topic", "General")).strip().title()
            meta["sub_topic"] = sub_topic
            meta["exam"] = state["profile"].target_exam
            meta["ttl_days"] = 180 if "Current" in meta.get("subject", "") else None
            
            # 3. Escape Hatch Taxonomy Tracker
            is_official = False
            for subj, topics in exam_map.items():
                for top, sub_topics in topics.items():
                    if sub_topic.lower() in [s.lower() for s in sub_topics]:
                        is_official = True
                        break
            
            meta["taxonomy_source"] = "official" if is_official else "llm_generated"
            
            new_batch.append(Question(**q_dict))
            
        return {"draft_batch": new_batch, "generation_attempts": attempts + 1, "rejected_batch": []}
    except Exception as e:
        print(f"❌ Generator format error: {e}")
        return {"draft_batch": [], "generation_attempts": attempts + 1, "rejected_batch": []}

# ==========================================
# 4. CRITIC NODE (The AI Examiner)
# ==========================================

def critic_node(state: AdaptiveTestState) -> dict:
    drafts = state.get("draft_batch", [])
    attempts = state.get("generation_attempts", 0)
    blueprint = state.get("blueprint")
    
    if not drafts:
        return {}
        
    print(f"🧐 Critic: Evaluating {len(drafts)} drafts...")
    drafts_json = json.dumps([d.model_dump() for d in drafts], indent=2)
    
    critic_prompt = f"""
    Evaluate the following {len(drafts)} drafted questions for formatting, math syntax ($), and accuracy.
    {drafts_json}
    Output a strictly formatted JSON array: [{{"id": "...", "approved": true/false, "feedback": "..."}}]
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=critic_prompt,
            config=types.GenerateContentConfig(temperature=0.1, response_mime_type="application/json")
        )
        reviews = json.loads(clean_json_response(response.text))
        
        approved = []
        rejected = []
        
        for review in reviews:
            q_id = review.get("id")
            draft_q = next((d for d in drafts if d.id == q_id), None)
            if not draft_q: continue
                
            if review.get("approved"):
                approved.append(draft_q)
            else:
                rejected.append({
                    "id": q_id, 
                    "feedback": review.get("feedback"),
                    "sub_topic": draft_q.metadata.sub_topic,
                })

        return {"selected_questions": approved, "rejected_batch": rejected, "current_batch_target": len(rejected)}
        
    except Exception as e:
        print(f"❌ Critic error: {e}. Passing drafts through to prevent stall.")
        return {"selected_questions": drafts, "rejected_batch": [], "current_batch_target": 0}

# ==========================================
# 5. SAVER NODE (The Archivist)
# ==========================================

def saver_node(state: AdaptiveTestState) -> dict:
    approved_questions = state.get("selected_questions", [])
    drafts = state.get("draft_batch", [])
    draft_ids = [d.id for d in drafts]
    
    new_approved = [q for q in approved_questions if q.id in draft_ids]
    if new_approved:
        # Group Bypass Rule implementation warning
        print(f"💾 Saver: Processing {len(new_approved)} new questions for OpenSearch...")
        
        grouped_qs = [q for q in new_approved if q.shared_context]
        standalone_qs = [q for q in new_approved if not q.shared_context]
        
        if grouped_qs:
            print(f"⚠️ BYPASSING Deduplication for {len(grouped_qs)} grouped 'shared_context' questions to protect atomic blocks.")
            
        # Ensure your save_questions_to_db logic inside vector_store.py handles this properly!
        save_questions_to_db(new_approved)
        
    return {"generation_attempts": 0}

# ==========================================
# CONDITIONAL ROUTING EDGES
# ==========================================

def route_after_planner(state: AdaptiveTestState) -> Literal["database_retriever", "__end__"]:
    if state.get("current_batch_target", 0) <= 0:
        return "__end__"
    return "database_retriever"

def route_after_retriever(state: AdaptiveTestState) -> Literal["planner", "generator"]:
    if state.get("current_batch_target", 0) > 0:
        return "generator"
    return "planner"

def route_after_critic(state: AdaptiveTestState) -> Literal["saver", "generator"]:
    if state.get("rejected_batch"):
        return "generator"
    if state.get("current_batch_target", 0) > 0 and not state.get("draft_batch"):
        return "generator"
    return "saver"

# ==========================================
# BUILD THE LANGGRAPH
# ==========================================

workflow = StateGraph(AdaptiveTestState)

workflow.add_node("planner", planner_node)
workflow.add_node("database_retriever", database_retriever_node)
workflow.add_node("generator", generator_node)
workflow.add_node("critic", critic_node)
workflow.add_node("saver", saver_node) 

workflow.set_entry_point("planner")

workflow.add_conditional_edges("planner", route_after_planner)
workflow.add_conditional_edges("database_retriever", route_after_retriever)
workflow.add_edge("generator", "critic")
workflow.add_conditional_edges("critic", route_after_critic) 
workflow.add_edge("saver", "planner") 

app = workflow.compile()