import os
import json
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

from schema import EvaluationState
from db import save_student_profile

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# HISTORY TRACKER (Temporary Storage)
# ==========================================
def save_test_history_locally(profile_id, exam, score, graded_results, study_plan):
    """Saves the test results to Lambda's /tmp directory to prevent read-only crashes."""
    tmp_dir = "/tmp/test_history"
    os.makedirs(tmp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{tmp_dir}/{profile_id}_{timestamp}.json"
    
    history_record = {
        "timestamp": timestamp,
        "exam": exam,
        "score_percentage": score,
        "graded_results": graded_results,
        "study_plan": study_plan
    }
    
    try:
        with open(filename, "w") as f:
            json.dump(history_record, f, indent=4)
        print(f"💾 Saved temporary test history to {filename}")
    except Exception as e:
        print(f"⚠️ Could not save history: {e}")

# ==========================================
# AGENT NODES
# ==========================================

def grader_node(state: EvaluationState) -> dict:
    """Evaluates student answers against the correct answers and compiles results."""
    print("📝 Grader: Grading student submissions...")
    
    questions = state.get("questions", [])
    student_answers = state.get("student_answers", {})
    
    graded_results = []
    correct_count = 0
    
    for q in questions:
        student_ans = student_answers.get(q.id, "Skipped")
        is_correct = (student_ans == q.correct_answer)
        
        if is_correct:
            correct_count += 1
            
        graded_results.append({
            "question_id": q.id,
            "topic": q.metadata.topic,
            "sub_topic": q.metadata.sub_topic,
            "difficulty": q.metadata.difficulty_level,
            "is_correct": is_correct,
            "student_answer": student_ans,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "text": q.text,
            "score_delta": "0.0" 
        })
        
    score_percentage = (correct_count / len(questions)) * 100 if questions else 0.0
    print(f"✅ Grader: Score calculated at {score_percentage:.2f}%")
    
    return {
        "graded_results": graded_results,
        "score_percentage": score_percentage
    }

def profiler_node(state: EvaluationState) -> dict:
    """Updates the student's proficiency metrics and calculates score deltas."""
    print("📈 Profiler: Updating student knowledge graph...")
    
    profile = state["profile"]
    graded_results = state.get("graded_results", [])
    
    profile.tests_taken += 1
    
    for result in graded_results:
        topic = result["topic"]
        difficulty = result["difficulty"]
        current_prof = profile.topic_proficiencies.get(topic, 0.5) 
        
        if result["is_correct"]:
            boost = 0.05 * difficulty 
            profile.topic_proficiencies[topic] = min(1.0, current_prof + boost)
            result["score_delta"] = f"+{boost:.3f}"
        else:
            penalty = 0.1 / difficulty 
            profile.topic_proficiencies[topic] = max(0.0, current_prof - penalty)
            result["score_delta"] = f"-{penalty:.3f}"
            
    if profile.topic_proficiencies:
        total_prof = sum(profile.topic_proficiencies.values())
        profile.overall_readiness_score = total_prof / len(profile.topic_proficiencies)
        
    save_student_profile(profile)
    
    return {
        "profile": profile,
        "graded_results": graded_results 
    }

def strategist_node(state: EvaluationState) -> dict:
    """Generates a structured plan and saves it to permanent memory."""
    print("🧠 Strategist: Generating Professional Academic Strategy...")
    
    graded_results = state.get("graded_results", [])
    score_percentage = state.get("score_percentage", 0.0)
    profile = state["profile"]
    
    mistakes = [r for r in graded_results if not r["is_correct"]]
    
    if not mistakes:
        study_plan = "### Diagnostic Summary\nExcellent performance. 100% accuracy achieved. Proceed to the next module to continue advancing your overall readiness."
        profile.last_study_plan = study_plan
        save_student_profile(profile)
        save_test_history_locally(profile.student_id, profile.target_exam, score_percentage, graded_results, study_plan)
        return {"study_plan": study_plan, "profile": profile}
        
    mistakes_context = ""
    for m in mistakes:
        mistakes_context += f"- Topic: {m['topic']} ({m['sub_topic']})\n"
        mistakes_context += f"  Fact missed: {m['explanation']}\n\n"
        
    history_context = f"Total Tests Taken: {profile.tests_taken}\n"
    history_context += f"Overall Exam Readiness: {profile.overall_readiness_score:.2f}/1.0\n"
    
    prompt = f"""
    You are a strict, professional academic evaluator and test strategist for the {profile.target_exam} exam.
    
    CURRENT TEST PERFORMANCE: Score: {score_percentage}%
    
    STUDENT'S MISTAKES:
    {mistakes_context}
    
    TASK: Generate a highly structured, professional academic remediation report. 
    
    CRITICAL CONSTRAINTS:
    1. TONE: Strictly academic, objective, and professional. 
    2. PROHIBITED: Do NOT use emojis. Do NOT use conversational filler. Do NOT generate URLs.
    3. FOCUS: Rely on standard academic concepts and standard reference materials.
    
    STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS (using Markdown headers):
    ### 1. Diagnostic Summary
    ### 2. Conceptual Remediation
    ### 3. Study Directives
    ### 4. Next Assessment Targets
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        study_plan = response.text
    except Exception as e:
        print(f"❌ Strategist error: {e}")
        study_plan = "Study plan generation temporarily unavailable."

    # --- PERMANENT MEMORY INJECTION ---
    print("💾 Strategist: Saving Study Plan to DynamoDB Memory...")
    profile.last_study_plan = study_plan
    save_student_profile(profile) # Commits to the new field we added in schema.py!

    save_test_history_locally(profile.student_id, profile.target_exam, score_percentage, graded_results, study_plan)

    return {"study_plan": study_plan, "profile": profile}

# ==========================================
# BUILD THE EVALUATOR LANGGRAPH
# ==========================================

evaluator_workflow = StateGraph(EvaluationState)

evaluator_workflow.add_node("grader", grader_node)
evaluator_workflow.add_node("profiler", profiler_node)
evaluator_workflow.add_node("strategist", strategist_node)

evaluator_workflow.set_entry_point("grader")
evaluator_workflow.add_edge("grader", "profiler")
evaluator_workflow.add_edge("profiler", "strategist")
evaluator_workflow.add_edge("strategist", END)

evaluator_app = evaluator_workflow.compile()