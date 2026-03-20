import os
import json
import boto3
import math
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from google import genai
from google.genai import types 
from schema import Question, StudentProfile
from typing import List, Optional
from dotenv import load_dotenv  

# ==========================================
# 1. CLOUD INITIALIZATION
# ==========================================

api_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key)

host = os.environ.get('OPENSEARCH_ENDPOINT', '').replace('https://', '')
region = os.environ.get('AWS_REGION', 'us-east-1')
service = 'aoss'

INDEX_NAME = "adaptive-questions-v2"

def get_opensearch_client():
    if not host or host == "pending-console-setup":
        return None
        
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

    return OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
        pool_maxsize=20
    )

def get_embedding(text: str) -> List[float]:
    try:
        response = gemini_client.models.embed_content(
            model='gemini-embedding-001', 
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=768) 
        )
        return response.embeddings[0].values
    except Exception as e:
        print(f"⚠️ Error generating embedding with Gemini: {e}")
        return []

def _ensure_index_exists(client):
    if not client.indices.exists(index=INDEX_NAME):
        index_body = {
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 768, 
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "id": {"type": "keyword"},
                    "exam": {"type": "keyword"},
                    "subject": {"type": "keyword"},
                    "topic": {"type": "keyword"},
                    "sub_topic": {"type": "keyword"},
                    "taxonomy_source": {"type": "keyword"},
                    "difficulty": {"type": "integer"},
                    "correct_answer": {"type": "keyword"}
                }
            }
        }
        try:
            client.indices.create(index=INDEX_NAME, body=index_body)
            print(f"✅ Created new index: {INDEX_NAME}")
        except Exception as e:
            pass

# ==========================================
# 2. THE SAVER (INGESTION & DEDUPLICATION)
# ==========================================

def save_questions_to_db(questions: List[Question]):
    client = get_opensearch_client()
    if not client: return
    _ensure_index_exists(client)

    for q in questions:
        if q.shared_context:
            print(f"   -> 🛡️ Bypassing Deduplication for Grouped Question: {q.id}")
            _index_question(client, q)
            continue

        embed_text = f"Question: {q.text} Explanation: {q.explanation}"
        vector = get_embedding(embed_text)
        
        if not vector:
            continue
            
        search_body = {
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {"knn": {"embedding": {"vector": vector, "k": 3}}}
                    ],
                    "filter": [
                        {"term": {"exam": q.metadata.exam}}
                    ]
                }
            }
        }
        
        try:
            response = client.search(index=INDEX_NAME, body=search_body)
            hits = response.get('hits', {}).get('hits', [])
            
            is_duplicate = False
            if hits:
                best_hit = hits[0]
                score = best_hit.get('_score', 0)
                existing_answer = best_hit.get('_source', {}).get('correct_answer')
                
                if score >= 1.95 and existing_answer == q.correct_answer:
                    print(f"   -> ♻️ Duplicate Found! Discarding {q.id} (Matches {best_hit['_source']['id']} with score {score:.2f})")
                    is_duplicate = True
            
            if not is_duplicate:
                _index_question(client, q, vector)
                
        except Exception as e:
            print(f"⚠️ Search failed, saving anyway: {e}")
            _index_question(client, q, vector)

def _index_question(client, q: Question, vector: List[float] = None):
    if not vector:
        embed_text = f"Question: {q.text} Explanation: {q.explanation}"
        vector = get_embedding(embed_text)
        if not vector: return

    doc = json.loads(q.model_dump_json())
    doc['embedding'] = vector
    doc['exam'] = q.metadata.exam
    doc['subject'] = q.metadata.subject
    doc['topic'] = q.metadata.topic
    doc['sub_topic'] = q.metadata.sub_topic
    doc['taxonomy_source'] = q.metadata.taxonomy_source
    doc['difficulty'] = q.metadata.difficulty_level
    doc['correct_answer'] = q.correct_answer
    
    try:
        client.index(index=INDEX_NAME, body=doc)
        print(f"   -> Embedded and saved question: {q.id}")
    except Exception as e:
        print(f"❌ Failed to save question {q.id} to OpenSearch: {e}")

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
    client = get_opensearch_client()
    if not client or not client.indices.exists(index=INDEX_NAME): return None

    # We now filter strictly by the exact 3-tier match
    query = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"exam": target_exam}}
                ],
                "must_not": [{"terms": {"id": exclude_ids}}]
            }
        }
    }
    
    if target_subject and target_subject not in ["All Syllabus", "Entire Syllabus"]:
        query["query"]["bool"]["must"].append({"term": {"subject": target_subject}})
        
    if target_topic and target_topic not in ["All Syllabus", "General"]:
        query["query"]["bool"]["must"].append({"term": {"topic": target_topic}})
        
    if target_sub_topic and target_sub_topic not in ["All Syllabus", "General"]:
        clean_sub_topic = str(target_sub_topic).strip().title()
        query["query"]["bool"]["must"].append({"term": {"sub_topic": clean_sub_topic}})

    # 👉 THE FIX: Strict match the dynamically generated difficulty level
    if target_difficulty is not None:
        query["query"]["bool"]["must"].append({"term": {"difficulty": target_difficulty}})

    try:
        response = client.search(index=INDEX_NAME, body=query)
        hits = response.get('hits', {}).get('hits', [])
        if hits:
            source = hits[0]['_source']
            source.pop('embedding', None)
            
            for field in ['exam', 'subject', 'topic', 'sub_topic', 'taxonomy_source', 'difficulty']:
                source.pop(field, None)
                
            return Question(**source)
    except Exception as e:
        print(f"⚠️ Retriever Search Error: {e}")
        pass
        
    return None

# ==========================================
# 4. SEMANTIC SNAPPER (DEDUPLICATION ENGINE)
# ==========================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def semantic_snap_topic(new_topic: str, existing_topics: List[str], similarity_threshold: float = 0.85) -> str:
    if not existing_topics:
        return new_topic

    print(f"🔄 Semantic Snapper: Checking if '{new_topic}' is a duplicate...")
    
    new_vector = get_embedding(new_topic)
    if not new_vector:
        return new_topic 

    best_match = None
    highest_score = -1.0

    for existing_topic in existing_topics:
        existing_vector = get_embedding(existing_topic)
        if not existing_vector:
            continue
            
        score = cosine_similarity(new_vector, existing_vector)
        if score > highest_score:
            highest_score = score
            best_match = existing_topic

    if highest_score >= similarity_threshold:
        print(f"✨ Semantic Snapper: Snapped '{new_topic}' -> '{best_match}' (Score: {highest_score:.2f})")
        return best_match
    else:
        print(f"🆕 Semantic Snapper: '{new_topic}' is genuinely new. (Highest match was {highest_score:.2f})")
        return new_topic