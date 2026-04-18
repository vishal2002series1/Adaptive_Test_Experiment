import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from schema import EvaluationState, ProficiencyRecord
from db import save_student_profile, save_test_history

# 👉 NEW: Import the centralized Bedrock client and Claude constants
from bedrock_client import bedrock_runtime

load_dotenv()

# Constants for Bedrock Claude
CLAUDE_MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-6")
ANTHROPIC_VERSION = "bedrock-2023-05-31"

def invoke_claude(prompt: str, system_prompt: str = "", temperature: float = 0.2) -> str:
    """Helper function to invoke Claude 3.5 Sonnet via Bedrock."""
    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system_prompt:
        body["system"] = system_prompt

    response = bedrock_runtime.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

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
            "subject": q.metadata.subject,      
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
    """Updates the structured DynamoDB array and calculates score deltas."""
    print("📈 Profiler: Updating structured NoSQL analytics array...")
    
    profile = state["profile"]
    graded_results = state.get("graded_results", [])
    
    profile.tests_taken += 1
    
    for result in graded_results:
        question_id = result["question_id"]
        subject = result["subject"]
        topic = result["topic"]
        sub_topic = result["sub_topic"]
        difficulty = result["difficulty"]
        
        # Mark this specific question ID as "seen" so the Retriever ignores it forever
        profile.seen_question_counts[question_id] = profile.seen_question_counts.get(question_id, 0) + 1
        
        # 1. Search for existing ProficiencyRecord in the array
        existing_record = next((p for p in profile.proficiencies if p.sub_topic == sub_topic), None)
        
        # 2. Calculate Math Delta
        if result["is_correct"]:
            score_delta = 0.05 * difficulty 
            result["score_delta"] = f"+{score_delta:.3f}"
        else:
            score_delta = -(0.1 / difficulty) 
            result["score_delta"] = f"{score_delta:.3f}"
            
        # 3. Update or Create
        if existing_record:
            existing_record.score = max(0.0, min(1.0, existing_record.score + score_delta))
            existing_record.questions_attempted += 1
            existing_record.last_tested = datetime.now(timezone.utc).isoformat()
        else:
            new_record = ProficiencyRecord(
                subject=subject,
                topic=topic,
                sub_topic=sub_topic,
                score=max(0.0, min(1.0, 0.5 + score_delta)), # Base 0.5 start
                questions_attempted=1
            )
            profile.proficiencies.append(new_record)
            
    # 4. Update overall score across all records
    if profile.proficiencies:
        total_prof = sum(p.score for p in profile.proficiencies)
        profile.overall_readiness_score = total_prof / len(profile.proficiencies)
        
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
        save_test_history(profile.student_id, profile.target_exam, score_percentage, graded_results, study_plan)
        return {"study_plan": study_plan, "profile": profile}
        
    mistakes_context = ""
    for m in mistakes:
        mistakes_context += f"- Taxonomy: {m['subject']} > {m['topic']} > {m['sub_topic']}\n"
        mistakes_context += f"  Fact missed: {m['explanation']}\n\n"
        
    history_context = f"Total Tests Taken: {profile.tests_taken}\n"
    history_context += f"Overall Exam Readiness: {profile.overall_readiness_score:.2f}/1.0\n"
    
    system_prompt = f"You are a strict, professional academic evaluator and test strategist for the {profile.target_exam} exam."

    prompt = f"""
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
        study_plan = invoke_claude(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
    except Exception as e:
        print(f"❌ Strategist error: {e}")
        study_plan = "Study plan generation temporarily unavailable."

    print("💾 Strategist: Saving Study Plan to DynamoDB Memory...")
    profile.last_study_plan = study_plan
    save_student_profile(profile) 

    save_test_history(profile.student_id, profile.target_exam, score_percentage, graded_results, study_plan)

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