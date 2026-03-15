import json
from schema import StudentProfile, TestConfig
from graph import app as langgraph_app

def lambda_handler(event, context):
    """
    AWS Lambda entry point. 
    Expects a JSON payload with 'student_profile' and 'test_config'.
    """
    print("🚀 Received request from API Gateway")
    
    try:
        # 1. Parse the incoming HTTP request body
        body = json.loads(event.get('body', '{}'))
        
        # 2. Extract configuration (with fallbacks for safety)
        student_data = body.get('student_profile', {})
        config_data = body.get('test_config', {})
        
        # 3. Rehydrate our Pydantic models
        student = StudentProfile(
            student_id=student_data.get('student_id', 'anonymous'),
            target_exam=student_data.get('target_exam', 'UPSC'),
            tests_taken=student_data.get('tests_taken', 0),
            overall_readiness_score=student_data.get('overall_readiness_score', 0.0),
            topic_proficiencies=student_data.get('topic_proficiencies', {}),
            seen_question_counts=student_data.get('seen_question_counts', {})
        )
        
        config = TestConfig(
            target_subject=config_data.get('target_subject', 'Economy'),
            target_difficulty=config_data.get('target_difficulty', 3),
            num_questions=config_data.get('num_questions', 5)
        )
        
        # 4. Initialize the LangGraph State
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
        
        # 5. Execute the Graph
        print(f"🧠 Invoking LangGraph for {config.num_questions} questions...")
        final_state = langgraph_app.invoke(initial_state)
        
        # 6. Extract and serialize the generated test
        selected = final_state.get("selected_questions", [])
        output_test = [q.model_dump() for q in selected] # Pydantic built-in serialization
        
        # 7. Return the HTTP Response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Important for frontend CORS
            },
            'body': json.dumps({
                'message': 'Test generated successfully',
                'questions': output_test
            }, default=str) # default=str handles datetime serialization
        }

    except Exception as e:
        print(f"❌ Error in Lambda execution: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }