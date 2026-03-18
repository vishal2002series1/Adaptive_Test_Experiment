import os
import json
import math
import uuid
import re
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

# --- UTILITY: JSON SCRUBBER ---
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
    
    if config.target_subject and config.target_subject.lower() not in ["entire syllabus", "all syllabus"]:
        print(f"🛡️ Gatekeeper: Validating if '{config.target_subject}' belongs to '{profile.target_exam}'...")
        validation_prompt = f"Does the subject '{config.target_subject}' legitimately belong to the official syllabus of the '{profile.target_exam}'? Reply ONLY with YES or NO."
        try:
            validation_response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=validation_prompt,
                config=types.GenerateContentConfig(temperature=0.0) 
            )
            if "no" in validation_response.text.strip().lower():
                raise ValueError(f"Subject '{config.target_subject}' is not a valid component of the '{profile.target_exam}'. Please correct your test configuration.")
            print("✅ Gatekeeper Approved.")
        except ValueError as ve:
            raise ve 
        except Exception as e:
            print(f"⚠️ Gatekeeper validation failed: {e}")

    if not blueprint:
        print("🧭 Planner: Assembling strict mathematical and pedagogical constraints...")
        last_plan = getattr(profile, 'last_study_plan', None)
        overrides = config.override_topics
        planner_context = ""
        
        if config.adaptive_mode:
            if overrides:
                planner_context += f"ADAPTIVE OVERRIDE: The user selected these specific topics: {overrides}.\n"
                if last_plan:
                    planner_context += f"AI'S PREVIOUS RECOMMENDATION:\n{last_plan}\n"
                    planner_context += "LOGIC: Target selected topics deeply for remediation if they were in the recommendation. Otherwise, treat as baseline exploration.\n"
                else:
                    planner_context += "LOGIC: Treat all selected topics as baseline exploration.\n"
            else:
                num_explore = math.ceil(config.num_questions * 0.20)
                num_exploit = config.num_questions - num_explore
                
                sorted_topics = sorted(profile.topic_proficiencies.items(), key=lambda item: item[1])
                weakest_topics = [t[0] for t in sorted_topics[:3]] if sorted_topics else [config.target_subject]
                
                known_topics_str = ", ".join(profile.explored_topics) if profile.explored_topics else "None"
                discovery_prompt = f"Identify 1 or 2 new topics for {profile.target_exam} ({config.target_subject}) not in [{known_topics_str}]. Comma-separated only."
                try:
                    res = client.models.generate_content(model='gemini-3.1-pro-preview', contents=discovery_prompt, config=types.GenerateContentConfig(temperature=0.7))
                    raw_new_topics = [t.strip() for t in res.text.split(",") if t.strip()]
                    valid_exploration_topics = []
                    for raw_topic in raw_new_topics:
                        snapped = semantic_snap_topic(raw_topic, profile.explored_topics)
                        if snapped not in profile.explored_topics:
                            profile.explored_topics.append(snapped)
                            profile.topic_proficiencies[snapped] = 0.5 
                            valid_exploration_topics.append(snapped)
                    if valid_exploration_topics:
                        save_student_profile(profile)
                except Exception as e:
                    valid_exploration_topics = []
                
                if not valid_exploration_topics:
                    num_exploit = config.num_questions
                    num_explore = 0
                
                planner_context += f"PURE ADAPTIVE: Allocate EXACTLY {num_exploit} questions to these weak topics: {weakest_topics}.\n"
                if valid_exploration_topics:
                    planner_context += f"EXPLORATION: Allocate EXACTLY {num_explore} questions to these newly discovered topics: {valid_exploration_topics}.\n"
                if last_plan:
                    planner_context += f"AI'S PREVIOUS RECOMMENDATION:\n{last_plan}\n"
        else:
            planner_context += f"MANUAL MODE: Disregard history. Allocate all {config.num_questions} questions strictly to Subject: {config.target_subject}, Topic: {config.target_topic}.\n"

        planner_prompt = f"""
        You are the Master Test Planner for the {profile.target_exam} exam.
        Your task is to create a Test Blueprint for exactly {config.num_questions} questions.
        
        {planner_context}
        
        Requested Base Difficulty: {config.target_difficulty} (Scale 1-5).
        
        Determine if a topic requires standard discrete questions, or if it requires grouped context (like Reading Comprehension, Data Interpretation, or Case Studies). 
        If it requires grouped context, set "requires_shared_context" to true.
        
        Output MUST be a valid JSON object matching this structure:
        {{
            "overall_strategy": "Brief explanation",
            "requirements": [
                {{
                    "topic": "Topic Name",
                    "quantity": 3,
                    "target_difficulty": 4,
                    "question_type": "Reading Comprehension",
                    "requires_shared_context": true,
                    "reasoning": "Why you chose this"
                }}
            ]
        }}
        Ensure the sum of all "quantity" values equals exactly {config.num_questions}.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=planner_prompt,
                config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json")
            )
            blueprint = TestBlueprint(**json.loads(clean_json_response(response.text)))
            
            total_reqs = sum(req.quantity for req in blueprint.requirements)
            if total_reqs < config.num_questions:
                diff = config.num_questions - total_reqs
                print(f"⚠️ Planner Math Correction: LLM hallucinated {total_reqs} questions. Forcing addition of {diff}.")
                if blueprint.requirements:
                    blueprint.requirements[0].quantity += diff
                else:
                    blueprint.requirements.append(BlueprintRequirement(topic=config.target_topic, quantity=diff, target_difficulty=config.target_difficulty, reasoning="Math correction", question_type="Standard", requires_shared_context=False))
                    
            print(f"📋 Blueprint Created: {blueprint.overall_strategy}")
        except Exception as e:
            print(f"❌ Planner failed: {e}. Falling back.")
            blueprint = TestBlueprint(
                overall_strategy="Fallback",
                requirements=[BlueprintRequirement(topic=config.target_topic, quantity=config.num_questions, target_difficulty=config.target_difficulty, reasoning="Fallback", question_type="Standard", requires_shared_context=False)]
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
        existing_counts[q.metadata.topic] = existing_counts.get(q.metadata.topic, 0) + 1

    for req in blueprint.requirements:
        if req.requires_shared_context:
            continue
            
        needed_for_topic = req.quantity - existing_counts.get(req.topic, 0)
        for _ in range(needed_for_topic):
            if len(found_questions) >= target:
                break 
                
            current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
            q = retrieve_best_question(
                target_exam=state["profile"].target_exam,
                target_subject=state["config"].target_subject,
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
        print("🛑 Generator Circuit Breaker Activated: Injecting Fallbacks.")
        num_fallback = len(rejected) if rejected else target
        fallback_batch = [
            Question(
                id=f"q_fallback_{uuid.uuid4().hex[:8]}",
                shared_context=None,
                text="[Emergency Fallback] The AI encountered a processing error. What is 2 + 2?",
                options={"A": "3", "B": "4", "C": "5", "D": "6"},
                correct_answer="B",
                explanation="This is a fallback placeholder to prevent system timeouts.",
                metadata=QuestionMetadata(exam=state["profile"].target_exam, subject="Fallback", topic="Fallback", sub_topic="Fallback", cognitive_skill="Recall", difficulty_level=1)
            ) for _ in range(num_fallback)
        ]
        return {"draft_batch": fallback_batch, "rejected_batch": [], "generation_attempts": attempts}

    batch_context = ""
    
    if rejected:
        batch_context += "REGENERATING REJECTED QUESTIONS:\n"
        for r in rejected:
            topic = r.get('topic', 'Unknown')
            diff = r.get('difficulty', '3')
            shared = r.get('requires_shared_context', 'False')
            batch_context += f"- Topic: {topic} | Difficulty: Level {diff} | Shared Context Required: {shared} | Feedback: {r['feedback']}\n"
        num_to_generate = len(rejected)
    else:
        existing_counts = {}
        for q in state.get("selected_questions", []):
            existing_counts[q.metadata.topic] = existing_counts.get(q.metadata.topic, 0) + 1
            
        batch_context += "CURRENT BATCH REQUIREMENTS:\n"
        slots_left = target
        actual_take_sum = 0
        
        for req in blueprint.requirements:
            needed = req.quantity - existing_counts.get(req.topic, 0)
            if needed > 0 and slots_left > 0:
                if req.requires_shared_context:
                    take = needed 
                else:
                    take = min(needed, slots_left)
                    
                batch_context += f"- Topic: {req.topic} | Type: {req.question_type} | Difficulty: Level {req.target_difficulty} | Shared Context Required: {req.requires_shared_context} | Quantity to generate now: {take}\n"
                slots_left -= take
                actual_take_sum += take

        num_to_generate = actual_take_sum
        
    if num_to_generate <= 0:
        return {"draft_batch": []}

    existing_questions = state.get("selected_questions", [])
    collision_context = "AVOID DUPLICATES: The test already contains questions on these specific angles:\n" if existing_questions else ""
    for i, eq in enumerate(existing_questions, 1):
        collision_context += f"Q{i}: {eq.text}\n"

    # --- UPDATED PROMPT: INCLUDES STRICT JSON TEMPLATE ---
    prompt = f"""
    You are the Senior Content Creator for the {state['profile'].target_exam} exam.
    Generate EXACTLY {num_to_generate} questions to fulfill the following active requirements.
    
    {batch_context}
    
    BLOOM'S TAXONOMY DIFFICULTY RUBRIC:
    - Level 1: Factual Recall (Definitions, direct formulas).
    - Level 3: Application (Multi-step calculation, synthesizing concepts).
    - Level 5: Evaluation & Synthesis (Complex inference, edge-case analysis, contains highly plausible 'trap' distractors).
    
    SHARED CONTEXT RULES (Generic Groups):
    If a requirement specifies "Shared Context Required: True", you MUST generate ONE shared, comprehensive passage, dataset, or case-study PER TOPIC. Place this massive passage strictly inside the "shared_context" field for EVERY related question in that topic block. Put ONLY the specific question prompt inside the "text" field.
    
    {collision_context}
    
    FORMATTING RULES (CRITICAL):
    1. DO NOT STRIP SPACES. Write in natural, grammatically correct English. Ensure proper spacing between words (e.g., "The boy went", NOT "Theboywent").
    2. Use raw LaTeX wrapped in $ for inline and $$ for block equations.
    
    REQUIREMENTS:
    - Output strictly as a JSON array of objects matching this exact schema:
    [
      {{
        "id": "q_gen_unique_id",
        "shared_context": "The 500-word shared passage goes here IF this topic requires a shared context. Otherwise, output null.",
        "text": "The specific question prompt goes here...",
        "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
        "correct_answer": "A",
        "explanation": "Detailed explanation...",
        "metadata": {{
            "exam": "{state['profile'].target_exam}", 
            "subject": "{state['config'].target_subject}", 
            "topic": "The exact topic this question covers", 
            "sub_topic": "Specific Sub-Topic Here", 
            "cognitive_skill": "Analytical", 
            "difficulty_level": 3, 
            "ttl_days": 180
        }}
      }}
    ]
    """
    
    print(f"✍️ Generator: Drafting {num_to_generate} questions (Attempt {attempts + 1})...")
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.4, response_mime_type="application/json")
        )
        
        clean_json = clean_json_response(response.text)
        raw_json_array = json.loads(clean_json)
        
        new_batch = []
        for q_dict in raw_json_array:
            base_id = q_dict.get("id", "q")
            unique_hash = uuid.uuid4().hex[:8]
            q_dict["id"] = f"{base_id}_{unique_hash}"
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
        
    print(f"🧐 Critic: Invoking AI to rigorously evaluate {len(drafts)} drafts...")
    
    drafts_json = json.dumps([d.model_dump() for d in drafts], indent=2)
    
    critic_prompt = f"""
    You are the elite Quality Assurance Reviewer for the {state['profile'].target_exam} exam.
    Review the following {len(drafts)} drafted questions.
    
    CRITICAL QUALITY CHECKS:
    1. FORMATTING & SPACING: Read the text carefully. Are there missing spaces between words resulting from JSON generation artifacts? (e.g., "Theboywent"). If a question is missing standard text spacing, reject it immediately.
    2. SHARED CONTEXT AUTHENTICITY: If questions share a common passage/dataset, verify that the shared context is identical across all related questions within the same topic.
    3. ACCURACY: Is the explanation mathematically and logically sound?
    
    QUESTIONS TO REVIEW:
    {drafts_json}
    
    Output a strictly formatted JSON array:
    [
      {{
        "id": "the_question_id",
        "approved": true or false,
        "feedback": "If false, explain exactly what the generator must fix. If true, write 'Approved'."
      }}
    ]
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
                is_shared = any(req.topic == draft_q.metadata.topic and req.requires_shared_context for req in blueprint.requirements)
                rejected.append({
                    "id": q_id, 
                    "feedback": review.get("feedback"),
                    "topic": draft_q.metadata.topic,
                    "difficulty": str(draft_q.metadata.difficulty_level),
                    "requires_shared_context": str(is_shared)
                })

        topics_to_fully_reject = set()
        for r in rejected:
            if r.get('requires_shared_context') == 'True':
                topics_to_fully_reject.add(r['topic'])
                
        if topics_to_fully_reject:
            new_approved = []
            for q in approved:
                if q.metadata.topic in topics_to_fully_reject:
                    rejected.append({
                        "id": q.id, 
                        "feedback": "Co-rejected because a related question sharing this context passage failed. Rewrite the ENTIRE block cohesively.",
                        "topic": q.metadata.topic,
                        "difficulty": str(q.metadata.difficulty_level),
                        "requires_shared_context": "True"
                    })
                else:
                    new_approved.append(q)
            approved = new_approved

        if attempts >= 3 and rejected:
            print("🛑 Critic Circuit Breaker: Replacing unfixable garbage drafts with Fallbacks.")
            for r in rejected:
                fallback = Question(
                    id=f"q_fallback_{uuid.uuid4().hex[:8]}",
                    shared_context=None,
                    text=f"[Emergency Fallback] The AI repeatedly failed formatting rules for '{r.get('topic', 'this topic')}'. What is 2 + 2?",
                    options={"A": "3", "B": "4", "C": "5", "D": "6"},
                    correct_answer="B",
                    explanation="Placeholder to prevent system stall due to generation artifacts.",
                    metadata=QuestionMetadata(exam=state["profile"].target_exam, subject=state["config"].target_subject, topic=r.get('topic', 'Fallback'), sub_topic="Fallback", cognitive_skill="Recall", difficulty_level=1)
                )
                approved.append(fallback)
            rejected = [] 
            
        print(f"✅ Critic: {len(approved)} approved, {len(rejected)} rejected.")
        return {"selected_questions": approved, "rejected_batch": rejected, "current_batch_target": len(rejected)}
        
    except Exception as e:
        print(f"❌ Critic parsing error: {e}. Circuit breaker activated.")
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
        print(f"💾 Saver: Pushing {len(new_approved)} new questions to OpenSearch Vector DB...")
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