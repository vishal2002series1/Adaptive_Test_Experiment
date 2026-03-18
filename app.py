import json
from schema import StudentProfile, TestConfig, Question
from graph import app as generator_app
from evaluator_graph import evaluator_app 
from db import get_student_profile

def lambda_handler(event, context):
    """
    AWS Lambda entry point. 
    Expects a JSON payload with an 'action' key.
    """
    print("🚀 Received request from Lambda Function URL")
    
    try:
        # 1. Parse the incoming HTTP request body
        body = json.loads(event.get('body', '{}'))
        action = body.get('action', 'generate') 
        
        # 2. Extract common profile configuration
        student_data = body.get('student_profile', {})
        student_id = student_data.get('student_id', 'anonymous')
        target_exam = student_data.get('target_exam', 'UPSC')
        
        # 3. Fetch historical profile from DynamoDB 
        student = get_student_profile(student_id=student_id, target_exam=target_exam)
        
        if 'seen_question_counts' in student_data:
            student.seen_question_counts.update(student_data['seen_question_counts'])

        # ==========================================
        # ROUTE 0: FETCH PROFILE (NEW)
        # ==========================================
        if action == 'get_profile':
            print(f"📖 Fetching profile for {student_id} ({target_exam})")
            return _build_response({
                'message': 'Profile fetched successfully',
                'profile': student.model_dump()
            })

        # ==========================================
        # ROUTE 1: TEST GENERATION
        # ==========================================
        elif action == 'generate':
            config_data = body.get('test_config', {})
            config = TestConfig(
                target_subject=config_data.get('target_subject', 'Economy'),
                target_topic=config_data.get('target_topic', 'All Syllabus'),
                target_difficulty=config_data.get('target_difficulty', 3),
                num_questions=config_data.get('num_questions', 5),
                adaptive_mode=config_data.get('adaptive_mode', True),
                override_topics=config_data.get('override_topics') # <-- Accept UI overrides
            )
            
            # If the user selected specific checkboxes, we inject them directly 
            # into the initial state so the Orchestrator knows to use them!
            start_exploitation = config.override_topics if config.override_topics else []
            
            initial_state = {
                "profile": student,
                "config": config,
                "current_question_index": 0,
                "generation_attempts": 0,
                "selected_questions": [],
                "draft_batch": [],
                "rejected_batch": [],
                "current_batch_target": config.num_questions,
                "exploitation_topics": start_exploitation, # <-- Injected here
                "exploration_topics": []
            }
            
            print(f"🧠 Invoking GENERATOR Graph for {config.num_questions} questions (Exam: {target_exam})...")
            final_state = generator_app.invoke(initial_state)
            
            selected = final_state.get("selected_questions", [])
            output_test = [q.model_dump() for q in selected] 
            
            return _build_response({
                'message': 'Test generated successfully',
                'questions': output_test
            })

        # ==========================================
        # ROUTE 2: TEST EVALUATION
        # ==========================================
        elif action == 'evaluate':
            raw_questions = body.get('questions', [])
            student_answers = body.get('student_answers', {})
            
            questions = [Question(**q) for q in raw_questions]
            
            evaluation_state = {
                "profile": student,
                "questions": questions,
                "student_answers": student_answers,
                "graded_results": [],
                "score_percentage": 0.0,
                "study_plan": ""
            }
            
            print(f"🧠 Invoking EVALUATOR Graph for {len(questions)} questions (Exam: {target_exam})...")
            final_state = evaluator_app.invoke(evaluation_state)
            
            return _build_response({
                'message': 'Test evaluated successfully',
                'score_percentage': final_state.get('score_percentage'),
                'graded_results': final_state.get('graded_results'),
                'study_plan': final_state.get('study_plan'),
                'updated_profile': final_state.get('profile').model_dump()
            })
            
        else:
            raise ValueError(f"Unknown action requested: {action}")

    except Exception as e:
        print(f"❌ Error in Lambda execution: {str(e)}")
        return _build_response({'error': str(e)}, status_code=500)

def _build_response(body_dict, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body_dict, default=str)
    }