import os
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
            "text": q.text
        })
        
    score_percentage = (correct_count / len(questions)) * 100 if questions else 0.0
    print(f"✅ Grader: Score calculated at {score_percentage:.2f}%")
    
    return {
        "graded_results": graded_results,
        "score_percentage": score_percentage
    }

def profiler_node(state: EvaluationState) -> dict:
    """Updates the student's proficiency metrics and saves them to DynamoDB."""
    print("📈 Profiler: Updating student knowledge graph...")
    
    profile = state["profile"]
    graded_results = state.get("graded_results", [])
    
    # 1. Update tests taken
    profile.tests_taken += 1
    
    # 2. Update topic proficiencies (Simple IRT approximation)
    for result in graded_results:
        topic = result["topic"]
        difficulty = result["difficulty"]
        current_prof = profile.topic_proficiencies.get(topic, 0.5) # Default to 0.5
        
        if result["is_correct"]:
            # Boost score, capped at 1.0
            boost = 0.05 * difficulty 
            profile.topic_proficiencies[topic] = min(1.0, current_prof + boost)
        else:
            # Penalize score, floored at 0.0
            penalty = 0.1 / difficulty 
            profile.topic_proficiencies[topic] = max(0.0, current_prof - penalty)
            
    # 3. Update overall readiness (Average of all topic proficiencies)
    if profile.topic_proficiencies:
        total_prof = sum(profile.topic_proficiencies.values())
        profile.overall_readiness_score = total_prof / len(profile.topic_proficiencies)
        
    # 4. Save the updated profile to DynamoDB
    save_student_profile(profile)
    
    return {"profile": profile}

def strategist_node(state: EvaluationState) -> dict:
    """Uses Gemini to generate a personalized study plan based on mistakes and history."""
    print("🧠 Strategist: Analyzing mistakes and history to generate a study plan...")
    
    graded_results = state.get("graded_results", [])
    score_percentage = state.get("score_percentage", 0.0)
    profile = state["profile"]
    
    # Isolate the mistakes to pass to the LLM
    mistakes = [r for r in graded_results if not r["is_correct"]]
    
    if not mistakes:
        return {"study_plan": "Excellent work! You achieved a perfect score. Continue reviewing your current syllabus materials to maintain this level of readiness."}
        
    mistakes_context = ""
    for m in mistakes:
        mistakes_context += f"- Topic: {m['topic']} ({m['sub_topic']})\n"
        mistakes_context += f"  Question: {m['text']}\n"
        mistakes_context += f"  Student chose: {m['student_answer']}, Correct was: {m['correct_answer']}\n"
        mistakes_context += f"  Fact: {m['explanation']}\n\n"
        
    # --- NEW: Format the student's historical journey ---
    history_context = f"Total Tests Taken: {profile.tests_taken}\n"
    history_context += f"Overall Exam Readiness: {profile.overall_readiness_score:.2f}/1.0\n"
    history_context += "Historical Topic Proficiencies (0.0 is weak, 1.0 is mastered):\n"
    for topic, score in profile.topic_proficiencies.items():
        history_context += f"- {topic}: {score:.2f}\n"
        
    prompt = f"""
    You are an expert {profile.target_exam} mentor. 
    
    STUDENT'S HISTORICAL JOURNEY:
    {history_context}
    
    CURRENT TEST PERFORMANCE:
    Score: {score_percentage}%
    Mistakes made on this specific test:
    {mistakes_context}
    
    Write a concise, encouraging, and highly specific 3-step study plan for this student. 
    Analyze their current mistakes in the context of their historical proficiencies. 
    If they are repeatedly failing a topic they are historically weak at, point it out. 
    Tell them exactly which concepts they confused and what they need to study to fix those conceptual gaps.
    Format the response using Markdown.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3, # Low temperature for focused advice
            )
        )
        return {"study_plan": response.text}
    except Exception as e:
        print(f"❌ Strategist error: {e}")
        return {"study_plan": "Study plan generation temporarily unavailable. Please review the answer explanations."}

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