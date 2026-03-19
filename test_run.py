import uuid
import json
from dotenv import load_dotenv # <-- Add here
load_dotenv()
from schema import StudentProfile, TestConfig, AdaptiveTestState, EvaluationState
from graph import app as generator_app
from evaluator_graph import evaluator_app


def run_local_test():
    print("\n" + "="*50)
    print("🚀 STARTING LOCAL ADAPTIVE TEST GENERATION")
    print("="*50)

    # 1. Mock the incoming request payload
    test_profile = StudentProfile(
        student_id="Local_Test_User_01",
        target_exam="SSC CGL" # Must match exactly what is in syllabus_maps.json
    )
    
    test_config = TestConfig(
        target_subject="Entire Syllabus",
        target_topic="All Syllabus",
        target_difficulty=3,
        num_questions=3, # Keep it small for a fast local test
        adaptive_mode=True
    )

    initial_state = AdaptiveTestState(
        profile=test_profile,
        config=test_config,
        blueprint=None,
        current_question_index=0,
        generation_attempts=0,
        selected_questions=[],
        draft_batch=[],
        rejected_batch=[],
        current_batch_target=0,
        exploitation_topics=[],
        exploration_topics=[]
    )

    # 2. RUN THE GENERATOR GRAPH
    try:
        final_gen_state = generator_app.invoke(initial_state)
    except Exception as e:
        print(f"\n❌ CRITICAL CRASH IN GENERATOR: {e}")
        return

    questions = final_gen_state.get("selected_questions", [])
    print(f"\n🎉 Generation Complete! Generated {len(questions)} questions.")
    
    if not questions:
        print("⚠️ No questions were generated. Check your terminal logs above for errors.")
        return

    # Print out the generated questions to verify taxonomy and IDs
    print("\n--- GENERATED QUESTIONS ---")
    for q in questions:
        print(f"ID: {q.id}")
        print(f"Taxonomy: {q.metadata.subject} > {q.metadata.topic} > {q.metadata.sub_topic} ({q.metadata.taxonomy_source})")
        print(f"Text: {q.text[:50]}...")
        print("-" * 30)

    # 3. MOCK STUDENT ANSWERS (Simulating Phase 2)
    print("\n" + "="*50)
    print("📝 SIMULATING STUDENT SUBMISSION")
    print("="*50)
    
    mock_answers = {}
    for i, q in enumerate(questions):
        # Let's force the student to get the first question right, and the rest wrong
        mock_answers[q.id] = q.correct_answer if i == 0 else "WrongAnswer"

    eval_initial_state = EvaluationState(
        profile=final_gen_state["profile"],
        questions=questions,
        student_answers=mock_answers,
        graded_results=[],
        score_percentage=0.0,
        study_plan=""
    )

    # 4. RUN THE EVALUATOR GRAPH
    try:
        final_eval_state = evaluator_app.invoke(eval_initial_state)
    except Exception as e:
        print(f"\n❌ CRITICAL CRASH IN EVALUATOR: {e}")
        return

    # 5. VERIFY THE NEW STRUCTURED ARRAY
    updated_profile = final_eval_state["profile"]
    print(f"\n✅ Evaluation Complete! Final Score: {final_eval_state['score_percentage']}%")
    print("\n--- UPDATED DYNAMODB STRUCTURE ---")
    print(json.dumps([p.model_dump() for p in updated_profile.proficiencies], indent=2))

if __name__ == "__main__":
    run_local_test()