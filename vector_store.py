import os
import json
import boto3
import math
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from google import genai
from google.genai import types 
from schema import Question, StudentProfile
from typing import List, Optional

# ==========================================
# 1. CLOUD INITIALIZATION
# ==========================================

api_key = os.environ.get("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key)

host = os.environ.get('OPENSEARCH_ENDPOINT', '').replace('https://', '')
region = os.environ.get('AWS_REGION', 'us-east-1')
service = 'aoss'

INDEX_NAME = "upsc-adaptive-questions"

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
                    "topic": {"type": "keyword"},
                    "exam": {"type": "keyword"},
                    "subject": {"type": "keyword"},
                    "difficulty": {"type": "integer"}
                }
            }
        }
        try:
            client.indices.create(index=INDEX_NAME, body=index_body)
        except Exception as e:
            pass

# ==========================================
# 2. THE SAVER (INGESTION ENGINE)
# ==========================================

def save_questions_to_db(questions: List[Question]):
    client = get_opensearch_client()
    if not client: return
    _ensure_index_exists(client)

    for q in questions:
        embed_text = f"Question: {q.text} Explanation: {q.explanation}"
        vector = get_embedding(embed_text)
        
        if not vector:
            continue
            
        doc = json.loads(q.model_dump_json())
        
        doc['embedding'] = vector
        doc['topic'] = q.metadata.topic
        doc['exam'] = q.metadata.exam
        doc['subject'] = q.metadata.subject
        doc['difficulty'] = q.metadata.difficulty_level
        
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
    target_difficulty: int,
    student_profile: StudentProfile,
    exclude_ids: List[str]
) -> Optional[Question]:
    client = get_opensearch_client()
    if not client or not client.indices.exists(index=INDEX_NAME): return None

    query = {
        "size": 1,
        "query": {
            "bool": {
                "must": [
                    {"term": {"exam": target_exam}},
                    {"term": {"subject": target_subject}}
                ],
                "must_not": [{"terms": {"id": exclude_ids}}]
            }
        }
    }
    
    if target_topic != "All Syllabus":
         query["query"]["bool"]["must"].append({"term": {"topic": target_topic}})

    try:
        response = client.search(index=INDEX_NAME, body=query)
        hits = response.get('hits', {}).get('hits', [])
        if hits:
            source = hits[0]['_source']
            source.pop('embedding', None)
            source.pop('topic', None)
            source.pop('exam', None)
            source.pop('subject', None)
            source.pop('difficulty', None)
            return Question(**source)
    except Exception as e:
        pass
        
    return None

# ==========================================
# 4. SEMANTIC SNAPPER (DEDUPLICATION ENGINE)
# ==========================================

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity mathematically without needing numpy."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def semantic_snap_topic(new_topic: str, existing_topics: List[str], similarity_threshold: float = 0.85) -> str:
    """
    Embeds the new topic and compares it to existing topics.
    If similarity > threshold, returns the existing topic (snaps to it).
    Otherwise, returns the new topic.
    """
    if not existing_topics:
        return new_topic

    print(f"🔄 Semantic Snapper: Checking if '{new_topic}' is a duplicate...")
    
    # 1. Convert the new topic into a mathematical vector
    new_vector = get_embedding(new_topic)
    if not new_vector:
        return new_topic 

    best_match = None
    highest_score = -1.0

    # 2. Compare it against all vectors of topics the student has already seen
    for existing_topic in existing_topics:
        existing_vector = get_embedding(existing_topic)
        if not existing_vector:
            continue
            
        score = cosine_similarity(new_vector, existing_vector)
        if score > highest_score:
            highest_score = score
            best_match = existing_topic

    # 3. Snap or Keep
    if highest_score >= similarity_threshold:
        print(f"✨ Semantic Snapper: Snapped '{new_topic}' -> '{best_match}' (Score: {highest_score:.2f})")
        return best_match
    else:
        print(f"🆕 Semantic Snapper: '{new_topic}' is genuinely new. (Highest match was {highest_score:.2f})")
        return new_topic