from schema import StudentProfile, TestConfig
from graph import app

def main():
    print("🚀 Initializing Adaptive Test Generation System...")

    # 1. Setup a Mock Student Profile
    # In production, this data is pulled from DynamoDB or PostgreSQL
    student = StudentProfile(
        student_id="student_123",
        target_exam="UPSC",
        tests_taken=2,
        overall_readiness_score=0.65,
        topic_proficiencies={"Monetary Policy": 0.5},
        # We simulate that the student has seen the static DB question once already.
        # This will test our Hybrid Search penalty logic!
        seen_question_counts={"q_econ_001": 1} 
    )

    # 2. Setup the Test Configuration
    # We will request 3 questions for this local test so it runs quickly.
    config = TestConfig(
        target_subject="Economy",
        target_difficulty=4,
        num_questions=3
    )

    # 3. Initialize the LangGraph State
    initial_state = {
        "profile": student,
        "config": config,
        "current_question_index": 0,
        "generation_attempts": 0,
        "selected_questions": [],
        "draft_batch": [],
        "rejected_batch": [],
        "current_batch_target": config.num_questions
    }

    print(f"\n📚 Target: {config.num_questions} questions for {student.target_exam} ({config.target_subject})")
    print("-" * 50)

    # 4. Execute the LangGraph Workflow
    # .invoke() runs the state machine from the entry point to the END node
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        print(f"\n❌ Execution failed: {e}")
        return

    # 5. Output the Final Test Array
    print("\n" + "=" * 50)
    print("✅ TEST GENERATION COMPLETE")
    print("=" * 50)
    
    selected = final_state.get("selected_questions", [])
    
    for i, q in enumerate(selected, 1):
        # A quick visual check to see if the question came from the Vector DB or the LLM
        source = "Vector DB" if not str(q.id).startswith("q_gen") else "Gemini 3.1 Pro (Search Grounded)"
        
        print(f"\nQ{i} [Source: {source}]")
        print(f"Taxonomy: {q.metadata.subject} > {q.metadata.topic} > {q.metadata.sub_topic}")
        print(f"Difficulty: L{q.metadata.difficulty_level}")
        print(f"\n{q.text}")
        for key, val in q.options.items():
            print(f"  {key}) {val}")
        print(f"\nCorrect Answer: {q.correct_answer}")
        print(f"Explanation: {q.explanation}")
        print("-" * 50)

if __name__ == "__main__":
    main()