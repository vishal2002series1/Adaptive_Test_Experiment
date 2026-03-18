import os
import json
import math
import uuid
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
    
    # --- AI VALIDATION GATEKEEPER ---
    # Only validate if the user provided a specific subject (not "Entire Syllabus" or "All Syllabus")
    if config.target_subject and config.target_subject.lower() not in ["entire syllabus", "all syllabus"]:
        print(f"🛡️ Gatekeeper: Validating if '{config.target_subject}' belongs to '{profile.target_exam}'...")
        validation_prompt = f"Does the subject '{config.target_subject}' legitimately belong to the official syllabus of the '{profile.target_exam}'? Reply ONLY with YES or NO. Do not explain."
        try:
            validation_response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=validation_prompt,
                config=types.GenerateContentConfig(temperature=0.0) # Zero creativity for strict factual check
            )
            if "no" in validation_response.text.strip().lower():
                print(f"❌ Gatekeeper Rejected: {config.target_subject} is not in {profile.target_exam}")
                raise ValueError(f"Subject '{config.target_subject}' is not a valid component of the '{profile.target_exam}'. Please correct your test configuration.")
            print("✅ Gatekeeper Approved.")
        except ValueError as ve:
            raise ve # Re-raise to abort the LangGraph execution immediately
        except Exception as e:
            print(f"⚠️ Gatekeeper validation failed to execute: {e}. Proceeding cautiously.")

    exploitation_topics = state.get("exploitation_topics", [])
    exploration_topics = state.get("exploration_topics", [])
    
    if config.adaptive_mode and not exploitation_topics:
        
        # 1. EXPLOITATION (80%): Find up to 3 weakest known topics
        if profile.topic_proficiencies:
            sorted_topics = sorted(profile.topic_proficiencies.items(), key=lambda item: item[1])
            exploitation_topics = [t[0] for t in sorted_topics[:3]]
            print(f"🧠 Orchestrator [EXPLOIT]: Weakest areas targeted -> {exploitation_topics}")
        else:
            exploitation_topics = [config.target_subject]
            print(f"🧠 Orchestrator [COLD START]: No profile found. Starting with broad subject -> {exploitation_topics[0]}")

        # 2. EXPLORATION (20%): Discover Multiple Uncharted Territories
        known_topics_str = ", ".join(profile.explored_topics) if profile.explored_topics else "None"
        
        # Adjust prompt context based on whether we are exploring the whole exam or a specific subject
        if config.target_subject.lower() in ["entire syllabus", "all syllabus"]:
            syllabus_context = f"Analyze the full {profile.target_exam} syllabus."
        else:
            syllabus_context = f"Analyze the {profile.target_exam} syllabus specifically for the subject: {config.target_subject}."
        
        exploration_prompt = f"""
        {syllabus_context}
        The student has already been tested on these topics: [{known_topics_str}].
        
        Identify 1 to 3 major topics or sub-topics that the student has NOT been tested on yet.
        If the syllabus is nearly exhausted, it is okay to return just 1 or 2 topics.
        Respond ONLY with a comma-separated list of the topic names. Do not include any other text, reasoning, or punctuation.
        """
        
        try:
            print("🔭 Orchestrator: Asking Gemini to chart new territories...")
            response = client.models.generate_content(
                model='gemini-3.1-pro-preview',
                contents=exploration_prompt,
                config=types.GenerateContentConfig(temperature=0.7) 
            )
            raw_new_topics = [t.strip() for t in response.text.split(",") if t.strip()]
            
            # 3. SEMANTIC SNAPPING (Looping through multiple topics)
            valid_exploration_topics = []
            for raw_topic in raw_new_topics:
                snapped_topic = semantic_snap_topic(
                    new_topic=raw_topic, 
                    existing_topics=profile.explored_topics
                )
                
                # 4. PROFILE INJECTION
                if snapped_topic not in profile.explored_topics:
                    profile.explored_topics.append(snapped_topic)
                    profile.topic_proficiencies[snapped_topic] = 0.5 
                    valid_exploration_topics.append(snapped_topic)
                    print(f"✨ Orchestrator: Injected new topic '{snapped_topic}' into profile with baseline 0.5")
            
            exploration_topics = valid_exploration_topics
            if exploration_topics:
                save_student_profile(profile)
            else:
                print("⚠️ Orchestrator: No genuinely new topics found (Syllabus exhausted?). Skipping exploration.")

        except Exception as e:
            print(f"⚠️ Orchestrator Exploration failed: {e}. Defaulting entirely to exploitation.")

    if not exploitation_topics:
        exploitation_topics = [config.target_topic]

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
        "exploitation_topics": exploitation_topics,
        "exploration_topics": exploration_topics,
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
    
    exploitation_topics = state.get("exploitation_topics", [state["config"].target_topic])
    exploration_topics = state.get("exploration_topics", [])
    
    num_explore = 0
    num_exploit = target
    
    if exploration_topics and target >= 2:
        num_explore = math.ceil(target * 0.20)
        num_exploit = target - num_explore
        
    # --- Search 1: Exploitation Topics (80% Distributed) ---
    exploit_allocations = {t: 0 for t in exploitation_topics}
    for i in range(num_exploit):
        topic = exploitation_topics[i % len(exploitation_topics)]
        exploit_allocations[topic] += 1

    for topic, count in exploit_allocations.items():
        if count > 0:
            print(f"   -> Looking for {count} questions on '{topic}'")
            for _ in range(count):
                current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
                q = retrieve_best_question(
                    target_exam=state["profile"].target_exam,
                    target_subject=state["config"].target_subject,
                    target_topic=topic,  
                    target_difficulty=state["config"].target_difficulty,
                    student_profile=state["profile"],
                    exclude_ids=current_exclude_ids
                )
                if q:
                    found_questions.append(q)
                else:
                    break 
            
    # --- Search 2: Exploration Topics (20% Distributed) ---
    if num_explore > 0 and exploration_topics:
        explore_allocations = {t: 0 for t in exploration_topics}
        for i in range(num_explore):
            topic = exploration_topics[i % len(exploration_topics)]
            explore_allocations[topic] += 1
            
        for topic, count in explore_allocations.items():
            if count > 0:
                print(f"   -> Looking for {count} questions on '{topic}'")
                for _ in range(count):
                    current_exclude_ids = existing_ids + seen_history + [fq.id for fq in found_questions]
                    q = retrieve_best_question(
                        target_exam=state["profile"].target_exam,
                        target_subject=state["config"].target_subject,
                        target_topic=topic,  
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
    target = state.get("current_batch_target", 0)
    rejected = state.get("rejected_batch", [])
    
    num_to_generate = len(rejected) if rejected else target
    
    if num_to_generate <= 0:
         return {"draft_batch": []}
         
    # --- The 80/20 Generation Logic ---
    exploitation_topics = state.get("exploitation_topics", [state["config"].target_topic])
    exploration_topics = state.get("exploration_topics", [])
    
    num_explore = 0
    num_exploit = num_to_generate
    
    if exploration_topics and num_to_generate >= 2:
        num_explore = math.ceil(num_to_generate * 0.20) 
        num_exploit = num_to_generate - num_explore

    batch_exploit_topics = []
    for i in range(num_exploit):
        batch_exploit_topics.append(exploitation_topics[i % len(exploitation_topics)])
        
    batch_explore_topics = []
    if num_explore > 0 and exploration_topics:
        existing_count = len(state.get("selected_questions", []))
        topic_index = existing_count % len(exploration_topics)
        for i in range(num_explore):
             batch_explore_topics.append(exploration_topics[(topic_index + i) % len(exploration_topics)])
    
    exploit_str = ", ".join(set(batch_exploit_topics))
    explore_str = ", ".join(set(batch_explore_topics)) if batch_explore_topics else ""
    
    if num_explore > 0:
        composition_instruction = f"""
        You must generate exactly {num_to_generate} questions. 
        Allocate the topics strictly as follows:
        - {num_exploit} question(s) explicitly covering topics from this list (distribute them evenly): [{exploit_str}]
        - {num_explore} question(s) explicitly covering topics from this new list (distribute evenly): [{explore_str}]
        """
        print(f"✍️ Generator: Drafting {num_exploit} across [{exploit_str}], and {num_explore} across [{explore_str}]...")
    else:
        composition_instruction = f"""
        You must generate exactly {num_to_generate} questions.
        All questions must explicitly cover topics from this list (distribute them evenly): [{exploit_str}]
        """
        print(f"✍️ Generator: Drafting {num_to_generate} questions across [{exploit_str}]...")

    existing_questions = state.get("selected_questions", [])
    
    collision_context = ""
    if existing_questions:
        collision_context = "CRITICAL AVOIDANCE INSTRUCTIONS:\nThe test currently includes the following questions. You MUST NOT generate questions that test the exact same factual concepts, mechanisms, or angles as these.\n\n"
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
    
    CURRENT AFFAIRS PROTOCOL:
    1. Assess the target subject: "{state['config'].target_subject}".
    2. DYNAMIC SUBJECTS (e.g., Economy, Environment, Current Events, Polity): You MUST use the Google Search tool to ground questions in real-world events, newly passed legislations, and recent judgments. Restrict your current affairs timeline strictly to the last 12 months ({current_year-1} to {current_year}).
    3. STATIC SUBJECTS (e.g., Physics, Mathematics, Core History, Static Geography): DO NOT force artificial current affairs (e.g., "In a 2026 study...") into the questions. Test the core theoretical concepts purely and directly without contemporary framing.
    
    {collision_context}
    
    {feedback_context}
    
    LATEX FORMATTING RULES (CRITICAL):
    1. You MUST use strictly raw LaTeX wrapped in $ for inline and $$ for block equations.
    2. NEVER use the word "\\backslash". Use the actual backslash character (e.g., \\frac, \\pi, \\sqrt).
    3. NEVER write out "^\\wedge". Use the standard caret symbol (e.g., x^2).
    4. Example Good: $\\frac{{\\pi^2}}{{4}}$
    5. Example Bad: $\\backslash frac{{\\backslash pi^{{\\wedge}}2}}{{4}}$
    
    REQUIREMENTS:
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
        new_batch = []
        for q_dict in raw_json_array:
            base_id = q_dict.get("id", "q")
            unique_hash = uuid.uuid4().hex[:8]
            q_dict["id"] = f"{base_id}_{unique_hash}"
            
            new_batch.append(Question(**q_dict))
            
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
    approved_questions = state.get("selected_questions", [])
    drafts = state.get("draft_batch", [])
    draft_ids = [d.id for d in drafts]
    
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