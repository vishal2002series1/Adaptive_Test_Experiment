import os
import json
import math
import time
from upstash_vector import Index
from schema import Question, StudentProfile
from typing import List, Optional

# 👉 NEW: Import the centralized Bedrock client
from bedrock_client import bedrock_runtime

# Initialize Upstash Index using Env Vars
try:
    upstash_url = os.environ.get("UPSTASH_VECTOR_REST_URL")
    upstash_token = os.environ.get("UPSTASH_VECTOR_REST_TOKEN")
    if upstash_url and upstash_token:
        index = Index(url=upstash_url, token=upstash_token)
    else:
        index = None
except Exception as e:
    print(f"⚠️ Upstash Init Error: {e}")
    index = None

def get_embedding(text: str) -> List[float]:
    """Generates embeddings using Amazon Titan Text Embeddings V2 via assumed Bedrock role."""
    try:
        # Titan V2 Payload
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024, # Titan V2 standard output
            "normalize": True
        })
        
        response = bedrock_runtime.invoke_model(
            body=body,
            modelId="amazon.titan-embed-text-v2:0",
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get('body').read())
        return response_body.get('embedding')
        
    except Exception as e:
        print(f"⚠️ Error generating embedding with Amazon Titan: {e}")
        return []

# ==========================================
# 2. THE SAVER (INGESTION & DEDUPLICATION)
# ==========================================

def save_questions_to_db(questions: List[Question]):
    if not index: return

    for q in questions:
        if q.shared_context:
            print(f"   -> 🛡️ Bypassing Deduplication for Grouped Question: {q.id}")
            _index_question(q)
            continue

        embed_text = f"Question: {q.text} Explanation: {q.explanation}"
        vector = get_embedding(embed_text)
        
        if not vector:
            continue
            
        safe_exam = q.metadata.exam.replace("'", "''")
        try:
            res = index.query(
                vector=vector,
                top_k=3,
                include_metadata=True,
                filter=f"exam = '{safe_exam}'"
            )
            
            is_duplicate = False
            if res and len(res) > 0:
                best_hit = res[0]
                score = best_hit.score 
                existing_answer = best_hit.metadata.get('correct_answer')
                
                if score >= 0.98 and existing_answer == q.correct_answer:
                    print(f"   -> ♻️ Duplicate Found! Discarding {q.id} (Matches {best_hit.id} with score {score:.2f})")
                    is_duplicate = True
            
            if not is_duplicate:
                _index_question(q, vector)
                
        except Exception as e:
            print(f"⚠️ Search failed, saving anyway: {e}")
            _index_question(q, vector)

def _index_question(q: Question, vector: List[float] = None):
    if not vector:
        embed_text = f"Question: {q.text} Explanation: {q.explanation}"
        vector = get_embedding(embed_text)
        if not vector: return

    current_time = int(time.time())
    metadata = {
        "id": q.id,
        "exam": q.metadata.exam,
        "subject": q.metadata.subject,
        "topic": q.metadata.topic,
        "sub_topic": q.metadata.sub_topic,
        "difficulty": q.metadata.difficulty_level,
        "correct_answer": q.correct_answer,
        "created_at": current_time,
        "full_json": q.model_dump_json()
    }
    
    ttl_days = getattr(q.metadata, 'ttl_days', None)
    if ttl_days:
        metadata['expires_at'] = current_time + (ttl_days * 86400)
    else:
        metadata['expires_at'] = 4070908800 

    try:
        index.upsert(vectors=[(q.id, vector, metadata)])
        print(f"   -> Embedded and saved question to Upstash: {q.id}")
    except Exception as e:
        print(f"❌ Failed to save question {q.id} to Upstash: {e}")

# ==========================================
# 3. THE RETRIEVER (SEARCH ENGINE)
# ==========================================

def retrieve_best_question(
    target_exam: str, 
    target_subject: str, 
    target_topic: str, 
    target_sub_topic: str, 
    target_difficulty: int,
    student_profile: StudentProfile,
    exclude_ids: List[str]
) -> Optional[Question]:
    if not index: return None

    current_time = int(time.time())
    safe_exam = target_exam.replace("'", "''")
    
    filters = [
        f"exam = '{safe_exam}'",
        f"expires_at >= {current_time}"
    ]
    
    query_parts = [target_exam]
    
    if target_subject and target_subject not in ["All Syllabus", "Entire Syllabus"]:
        safe_subject = target_subject.replace("'", "''")
        filters.append(f"subject = '{safe_subject}'")
        query_parts.append(target_subject)
        
    if target_topic and target_topic not in ["All Syllabus", "General"]:
        safe_topic = target_topic.replace("'", "''")
        filters.append(f"topic = '{safe_topic}'")
        query_parts.append(target_topic)
        
    if target_sub_topic and target_sub_topic not in ["All Syllabus", "General"]:
        clean_sub_topic = str(target_sub_topic).strip().replace("'", "''")
        filters.append(f"sub_topic = '{clean_sub_topic}'")
        query_parts.append(str(target_sub_topic).strip())

    if target_difficulty is not None:
        filters.append(f"difficulty = {target_difficulty}")
        
    if exclude_ids:
        safe_ids = [f"'{eid}'" for eid in exclude_ids]
        filters.append(f"id NOT IN ({', '.join(safe_ids)})")

    filter_string = " AND ".join(filters)

    query_text = " ".join(query_parts)
    if not query_text.strip(): query_text = "general exam questions"
    
    query_vector = get_embedding(query_text)
    if not query_vector: return None

    try:
        res = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            filter=filter_string
        )
        if res and len(res) > 0:
            best_hit = res[0]
            q_json = best_hit.metadata.get('full_json')
            if q_json:
                return Question(**json.loads(q_json))
    except Exception as e:
        print(f"⚠️ Retriever Search Error: {e}")
        
    return None

# ==========================================
# 4. SEMANTIC SNAPPER (DEDUPLICATION ENGINE)
# ==========================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    return dot_product / (norm_v1 * norm_v2)

def semantic_snap_topic(new_topic: str, existing_topics: List[str], similarity_threshold: float = 0.85) -> str:
    if not existing_topics: return new_topic

    new_vector = get_embedding(new_topic)
    if not new_vector: return new_topic 

    best_match = None
    highest_score = -1.0

    for existing_topic in existing_topics:
        existing_vector = get_embedding(existing_topic)
        if not existing_vector: continue
            
        score = cosine_similarity(new_vector, existing_vector)
        if score > highest_score:
            highest_score = score
            best_match = existing_topic

    if highest_score >= similarity_threshold:
        return best_match
    return new_topic