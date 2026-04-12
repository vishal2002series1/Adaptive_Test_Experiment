import json
from decimal import Decimal
from schema import StudentProfile, TestConfig, Question
from graph import app as generator_app
from evaluator_graph import evaluator_app 
from workbook_graph import workbook_app
from vector_store import save_questions_to_db

# 👉 THE FIX: Added get_presigned_url to the import list
from db import (
    get_student_profile, 
    get_all_student_profiles,
    get_student_test_history, 
    get_cached_workbook,
    save_pending_test,
    get_pending_test,
    delete_pending_test,
    save_master_question,
    get_presigned_url
)

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

def _build_response(body_dict, status_code=200):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps(body_dict, cls=DecimalEncoder)
    }

def lambda_handler(event, context):
    """
    AWS Lambda entry point. 
    Expects a JSON payload with an 'action' key.
    """
    print("🚀 Received request from Lambda Function URL")
    
    try:
        body = json.loads(event.get('body', '{}'))
        action = body.get('action', 'generate') 
        
        student_data = body.get('student_profile', {})
        student_id = student_data.get('student_id', 'anonymous')
        target_exam = student_data.get('target_exam', 'UPSC')
        
        student = get_student_profile(student_id=student_id, target_exam=target_exam)
        
        if 'seen_question_counts' in student_data:
            student.seen_question_counts.update(student_data['seen_question_counts'])

        # ==========================================
        # ROUTE 0: FETCH PROFILE 
        # ==========================================
        if action == 'get_profile':
            print(f"📖 Fetching profile for {student_id} ({target_exam})")
            return _build_response({
                'message': 'Profile fetched successfully',
                'profile': student.model_dump()
            })

        # ==========================================
        # ROUTE 0.5: FETCH ALL PROFILES (DASHBOARD)
        # ==========================================
        elif action == 'get_all_profiles':
            print(f"📊 Fetching all active profiles for {student_id}")
            profiles = get_all_student_profiles(student_id)
            return _build_response({
                'message': 'Dashboard fetched successfully',
                'profiles': profiles
            })

        # ==========================================
        # ROUTE 1: FETCH TEST HISTORY
        # ==========================================
        elif action == 'get_history':
            print(f"📖 Fetching test history for {student_id}")
            history_logs = get_student_test_history(student_id)
            return _build_response({
                'message': 'History fetched successfully',
                'history': history_logs
            })

        # ==========================================
        # ROUTE 2: GET OR GENERATE WORKBOOK
        # ==========================================
        elif action == 'get_workbook':
            config_data = body.get('workbook_config', {})
            subject = config_data.get('subject', 'General')
            topic = config_data.get('topic', 'General')
            sub_topic = config_data.get('sub_topic')
            difficulty = config_data.get('difficulty_level', 3)

            if not sub_topic:
                raise ValueError("A sub_topic must be provided to fetch a workbook.")

            print(f"📚 Requesting Workbook: {target_exam} -> {sub_topic} (Lvl {difficulty})")
            
            cached_wb = get_cached_workbook(target_exam, sub_topic, difficulty)
            if cached_wb:
                print("⚡ Cache hit! Returning instant workbook.")
                return _build_response({
                    'message': 'Workbook fetched from cache',
                    'workbook': cached_wb
                })
                
            print("⏳ Cache miss. Invoking Workbook Generator Graph...")
            wb_state = {
                "target_exam": target_exam,
                "subject": subject,
                "topic": topic,
                "sub_topic": sub_topic,
                "difficulty_level": difficulty
            }
            final_wb_state = workbook_app.invoke(wb_state)
            
            return _build_response({
                'message': 'Workbook generated successfully',
                'workbook': final_wb_state['final_workbook'].model_dump()
            })

        # ==========================================
        # NEW ROUTE: FETCH EXAM SYLLABUS
        # ==========================================
        elif action == 'get_syllabus':
            exam_name = body.get('exam_name', target_exam)
            print(f"📖 Fetching syllabus for {exam_name}")
            
            try:
                with open('syllabus_maps.json', 'r') as f:
                    syllabus_data = json.load(f)
                    
                if exam_name in syllabus_data:
                    return _build_response({
                        'message': 'Syllabus fetched successfully',
                        'exam': exam_name,
                        'syllabus': syllabus_data[exam_name]
                    })
                else:
                    return _build_response({
                        'error': f'Syllabus for {exam_name} not found in syllabus_maps.json'
                    }, status_code=404)
            except FileNotFoundError:
                return _build_response({'error': 'syllabus_maps.json file missing'}, status_code=500)
                
        # ==========================================
        # NEW ROUTE: INGEST HITL QUESTIONS
        # ==========================================
        elif action == 'ingest_questions':
            questions_data = body.get('questions', [])
            print(f"📥 Received {len(questions_data)} approved questions for ingestion.")
            
            success_count = 0
            for q_dict in questions_data:
                try:
                    # 1. Upload to S3 & Save to Master DynamoDB
                    save_master_question(q_dict)
                    
                    # 2. Rehydrate into Pydantic model
                    q_obj = Question(**q_dict) 
                    
                    # 3. Embed & Insert into OpenSearch Vector DB (reuses your existing logic!)
                    save_questions_to_db([q_obj])
                    
                    success_count += 1
                except Exception as e:
                    print(f"⚠️ Failed to ingest question {q_dict.get('id')}: {e}")
                    
            return _build_response({
                'message': f'Successfully ingested {success_count} out of {len(questions_data)} questions into the master bank and Vector DB.'
            })

        # ==========================================
        # UPGRADED ROUTE: FETCH STRUCTURED PROGRESS TREE
        # ==========================================
        elif action == 'get_progress_tree':
            print(f"🌳 Fetching structured progress tree for {student_id} ({target_exam})")
            
            try:
                with open('syllabus_maps.json', 'r') as f:
                    syllabus_data = json.load(f)
            except FileNotFoundError:
                return _build_response({'error': 'syllabus_maps.json file missing'}, status_code=500)
                
            exam_syllabus = syllabus_data.get(target_exam)
            if not exam_syllabus:
                return _build_response({'error': f'Syllabus for {target_exam} not found'}, status_code=404)

            # 1. Fast lookup dictionary for subtopics (Resilient Case-Insensitive Matching)
            prof_lookup = {}
            for p in student.proficiencies:
                # We lowercase and strip the sub_topic to protect against LLM casing changes
                key = p.sub_topic.strip().lower()
                prof_lookup[key] = {
                    "score": p.score,
                    "questions_attempted": p.questions_attempted,
                    "last_tested": p.last_tested
                }
            
            # 2. Build Tree and Calculate Roll-Up Aggregations (Approach C: Bayesian 0.5 Prior)
            progress_tree = []
            total_syllabus_subtopics = 0
            total_syllabus_score = 0.0
            total_attempted_subtopics = 0

            for subject, topics in exam_syllabus.items():
                subject_node = {
                    "subject": subject,
                    "subject_score": 0.0,
                    "subject_coverage_pct": 0.0,
                    "topics": []
                }
                subject_subtopic_count = 0
                subject_score_sum = 0.0
                subject_attempted_count = 0

                for topic, sub_topics in topics.items():
                    topic_node = {
                        "topic": topic,
                        "topic_score": 0.0,
                        "topic_coverage_pct": 0.0,
                        "sub_topics": []
                    }
                    topic_subtopic_count = len(sub_topics)
                    topic_score_sum = 0.0
                    topic_attempted_count = 0

                    for sub_topic in sub_topics:
                        total_syllabus_subtopics += 1
                        
                        # 👉 THE FIX: Normalize the syllabus key to match the database key
                        lookup_key = sub_topic.strip().lower()
                        
                        # APPROACH C IMPLEMENTATION: Default unseen score is 0.5
                        stats = prof_lookup.get(lookup_key, {
                            "score": 0.5, # <-- THE BAYESIAN PRIOR
                            "questions_attempted": 0,
                            "last_tested": None
                        })
                        
                        if stats["questions_attempted"] > 0:
                            topic_attempted_count += 1
                            total_attempted_subtopics += 1
                            subject_attempted_count += 1

                        # Add to Topic Sums
                        topic_score_sum += stats["score"]
                        topic_node["sub_topics"].append({
                            "sub_topic": sub_topic,
                            "progress": stats
                        })

                    # Calculate Topic Average & Coverage
                    if topic_subtopic_count > 0:
                        topic_node["topic_score"] = topic_score_sum / topic_subtopic_count
                        topic_node["topic_coverage_pct"] = (topic_attempted_count / topic_subtopic_count) * 100
                    
                    subject_node["topics"].append(topic_node)
                    
                    # Roll up to Subject Sums
                    subject_subtopic_count += topic_subtopic_count
                    subject_score_sum += topic_score_sum

                # Calculate Subject Average & Coverage
                if subject_subtopic_count > 0:
                    subject_node["subject_score"] = subject_score_sum / subject_subtopic_count
                    subject_node["subject_coverage_pct"] = (subject_attempted_count / subject_subtopic_count) * 100
                
                progress_tree.append(subject_node)
                
                # Roll up to Global Sums
                total_syllabus_score += subject_score_sum

            # 3. Calculate True Overall Readiness (Syllabus Coverage Weighted with 0.5 baseline)
            true_overall_readiness = (total_syllabus_score / total_syllabus_subtopics) if total_syllabus_subtopics > 0 else 0.5
            overall_coverage_pct = (total_attempted_subtopics / total_syllabus_subtopics) * 100 if total_syllabus_subtopics > 0 else 0.0

            return _build_response({
                'message': 'Progress tree constructed successfully',
                'true_overall_readiness': true_overall_readiness,
                'overall_coverage_pct': overall_coverage_pct,
                'total_syllabus_nodes': total_syllabus_subtopics,
                'attempted_nodes': total_attempted_subtopics,
                'progress_tree': progress_tree
            })
            
        # ==========================================
        # ROUTE 3: TEST GENERATION & INTERCEPTOR
        # ==========================================
        elif action == 'generate':
            config_data = body.get('test_config', {})
            config = TestConfig(
                target_subject=config_data.get('target_subject', 'Economy'),
                target_topic=config_data.get('target_topic', 'All Syllabus'),
                target_difficulty=config_data.get('target_difficulty'), 
                num_questions=config_data.get('num_questions', 5),
                adaptive_mode=config_data.get('adaptive_mode', True),
                override_topics=config_data.get('override_topics') 
            )
            
            start_exploitation = config.override_topics if config.override_topics else []
            
            # 🛑 PHASE 2: THE INTERCEPTOR ENGINE
            pending_test = get_pending_test(student_id, target_exam)
            
            if pending_test:
                pending_config = pending_test.get('test_config', {})
                pending_qs = pending_test.get('questions', [])
                
                # STRICT MATCH VALIDATION
                is_match = (
                    pending_config.get('target_subject') == config.target_subject and
                    pending_config.get('target_topic') == config.target_topic and
                    pending_config.get('target_difficulty') == config.target_difficulty and
                    pending_config.get('adaptive_mode') == config.adaptive_mode and
                    pending_config.get('override_topics') == config.override_topics
                )
                
                if is_match:
                    r_len = config.num_questions
                    p_len = len(pending_qs)
                    
                    if r_len <= p_len:
                        # ⚡ INSTANT RESUME (Cost: $0)
                        print(f"🔄 Resuming session: Slicing {r_len} questions from {p_len} pending.")
                        output_test = pending_qs[:r_len]
                        
                        # 👉 THE FIX: Attach temporary S3 URLs for rendering images
                        for q in output_test:
                            if q.get('s3_image_key'):
                                q['presigned_image_url'] = get_presigned_url(q['s3_image_key'])
                                
                        save_pending_test(student_id, target_exam, config_data, output_test)
                        
                        return _build_response({
                            'message': 'Session restored successfully',
                            'questions': output_test,
                            'session_restored': True
                        })
                    else:
                        # 🧩 DELTA GENERATION (Partial Cost)
                        delta = r_len - p_len
                        print(f"🔄 Resuming session: Found {p_len} pending. Generating {delta} more.")
                        
                        exclude_ids = [q['id'] for q in pending_qs]
                        
                        # Rehydrate the Pydantic objects to inject memory into LangGraph
                        pending_objects = [Question(**q) for q in pending_qs]
                        
                        initial_state = {
                            "profile": student,
                            "config": config, 
                            "current_question_index": 0,
                            "generation_attempts": 0,
                            "selected_questions": pending_objects, 
                            "draft_batch": [],
                            "rejected_batch": [],
                            "current_batch_target": delta,
                            "exploitation_topics": start_exploitation, 
                            "exploration_topics": [],
                            "exclude_ids": exclude_ids
                        }
                        
                        final_state = generator_app.invoke(initial_state)
                        
                        # Graph handles the concatenation natively now
                        output_test = [q.model_dump() for q in final_state.get("selected_questions", [])]
                        
                        # 👉 THE FIX: Attach temporary S3 URLs for rendering images
                        for q in output_test:
                            if q.get('s3_image_key'):
                                q['presigned_image_url'] = get_presigned_url(q['s3_image_key'])
                                
                        save_pending_test(student_id, target_exam, config_data, output_test)
                        
                        return _build_response({
                            'message': 'Session restored and expanded successfully',
                            'questions': output_test,
                            'session_restored': True
                        })
                else:
                    print("⚠️ Pending test config mismatch. Overwriting with new request.")

            # ⚙️ FRESH GENERATION (If no pending test or mismatch)
            initial_state = {
                "profile": student,
                "config": config,
                "current_question_index": 0,
                "generation_attempts": 0,
                "selected_questions": [],
                "draft_batch": [],
                "rejected_batch": [],
                "current_batch_target": config.num_questions,
                "exploitation_topics": start_exploitation, 
                "exploration_topics": [],
                "exclude_ids": []
            }
            
            print(f"🧠 Invoking GENERATOR Graph for {config.num_questions} questions (Exam: {target_exam})...")
            final_state = generator_app.invoke(initial_state)
            
            output_test = [q.model_dump() for q in final_state.get("selected_questions", [])] 
            
            # 👉 THE FIX: Attach temporary S3 URLs for rendering images
            for q in output_test:
                if q.get('s3_image_key'):
                    q['presigned_image_url'] = get_presigned_url(q['s3_image_key'])
                    
            # Save the fresh test to Pending cache
            save_pending_test(student_id, target_exam, config_data, output_test)
            
            return _build_response({
                'message': 'Test generated successfully',
                'questions': output_test,
                'session_restored': False
            })

        # ==========================================
        # ROUTE 4: TEST EVALUATION & PURGE
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
            
            # 🛑 PHASE 3: THE PURGE
            delete_pending_test(student_id, target_exam)
            
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