import os
import time
import boto3
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional
from boto3.dynamodb.conditions import Key
from schema import StudentProfile, ProficiencyRecord

dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

# 1. Existing Profile Table
TABLE_NAME = os.environ.get('STUDENT_TABLE', 'AdaptiveStudentProfiles') 
table = dynamodb.Table(TABLE_NAME)

# 2. History Logs Table
HISTORY_TABLE_NAME = os.environ.get('HISTORY_TABLE', 'AdaptiveTestHistoryLogs')
history_table = dynamodb.Table(HISTORY_TABLE_NAME)

# 3. Workbook Cache Table
WORKBOOK_TABLE_NAME = os.environ.get('WORKBOOK_TABLE', 'AdaptiveWorkbooks')
workbook_table = dynamodb.Table(WORKBOOK_TABLE_NAME)

# 4. Pending Tests Table
PENDING_TESTS_TABLE_NAME = os.environ.get('PENDING_TESTS_TABLE', 'AdaptivePendingTests')
pending_tests_table = dynamodb.Table(PENDING_TESTS_TABLE_NAME)


def get_student_profile(student_id: str, target_exam: str) -> StudentProfile:
    """Fetches a student profile from DynamoDB or creates a default one."""
    try:
        response = table.get_item(Key={
            'student_id': student_id,
            'target_exam': target_exam  
        })
        if 'Item' in response:
            item = response['Item']
            
            # --- THE SAFETY NET: ON-THE-FLY MIGRATION ---
            if 'topic_proficiencies' in item and 'proficiencies' not in item:
                print(f"⚠️ Legacy data detected for {student_id}. Running on-the-fly migration...")
                old_profs = item.pop('topic_proficiencies')
                new_profs = []
                for topic_name, score in old_profs.items():
                    new_profs.append({
                        "subject": "Legacy Subject",
                        "topic": "Legacy Topic",
                        "sub_topic": str(topic_name),
                        "score": float(score),
                        "questions_attempted": 1 
                    })
                item['proficiencies'] = new_profs

            return StudentProfile(**item)
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch from DynamoDB ({e}). Using default profile.")
    
    return StudentProfile(student_id=student_id, target_exam=target_exam)

def save_student_profile(profile: StudentProfile) -> bool:
    """Saves the updated student profile back to DynamoDB."""
    try:
        item = profile.model_dump()
        dynamo_item = json.loads(json.dumps(item), parse_float=Decimal)
        
        table.put_item(Item=dynamo_item)
        print(f"💾 Successfully saved {profile.target_exam} profile for {profile.student_id} to DynamoDB.")
        return True
    except Exception as e:
        print(f"❌ Error saving to DynamoDB: {e}")
        return False

# 👉 NEW: Fetch all profiles for the Dashboard
def get_all_student_profiles(student_id: str) -> List[Dict[str, Any]]:
    """Fetches all ongoing exam profiles for a student to populate the dashboard."""
    try:
        response = table.query(
            KeyConditionExpression=Key('student_id').eq(student_id)
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ Error fetching all profiles from DynamoDB: {e}")
        return []

# ==========================================
# TEST HISTORY LOGIC
# ==========================================

def save_test_history(student_id: str, exam: str, score: float, graded_results: list, study_plan: str) -> bool:
    """Saves a completed test log to the DynamoDB history table."""
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        
        history_record = {
            "student_id": student_id,
            "timestamp": timestamp,
            "exam": exam,
            "score_percentage": score,
            "graded_results": graded_results,
            "study_plan": study_plan
        }
        
        dynamo_item = json.loads(json.dumps(history_record), parse_float=Decimal)
        
        history_table.put_item(Item=dynamo_item)
        print(f"💾 Permanently saved test history log for {student_id} to DynamoDB.")
        return True
    except Exception as e:
        print(f"❌ Error saving test history to DynamoDB: {e}")
        return False

def get_student_test_history(student_id: str) -> List[Dict[str, Any]]:
    """Fetches all historical test logs for a student, newest first."""
    try:
        response = history_table.query(
            KeyConditionExpression=Key('student_id').eq(student_id),
            ScanIndexForward=False  
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ Error fetching test history from DynamoDB: {e}")
        return []

# ==========================================
# DYNAMIC WORKBOOK CACHE LOGIC
# ==========================================

def _generate_topic_key(sub_topic: str, difficulty: int) -> str:
    """Creates a unique deterministic string for DynamoDB Sort Key."""
    clean_sub_topic = str(sub_topic).strip().title().replace(" ", "")
    return f"{clean_sub_topic}#Lvl{difficulty}"

def get_cached_workbook(target_exam: str, sub_topic: str, difficulty: int) -> Optional[Dict[str, Any]]:
    """Checks if a workbook for this topic and difficulty already exists to save LLM costs."""
    topic_key = _generate_topic_key(sub_topic, difficulty)
    print(f"🔍 Checking cache for Workbook: {target_exam} -> {topic_key}")
    try:
        response = workbook_table.get_item(Key={
            'target_exam': target_exam,
            'topic_key': topic_key
        })
        item = response.get('Item')
        
        if not item:
            return None
            
        # 👉 THE FIX: Manually enforce TTL for ghost records
        if item.get('expires_at', 0) and item.get('expires_at', 0) < int(time.time()):
            print(f"⚠️ Found expired workbook for {topic_key}. Ignoring to generate fresh content.")
            return None
            
        return item
    except Exception as e:
        print(f"❌ Error fetching workbook from DynamoDB: {e}")
        return None

def save_cached_workbook(workbook_dict: Dict[str, Any]) -> bool:
    """Saves a newly generated workbook to DynamoDB so future students get it instantly."""
    try:
        # Inject the composite key required by DynamoDB
        topic_key = _generate_topic_key(workbook_dict['sub_topic'], workbook_dict['difficulty_level'])
        workbook_dict['topic_key'] = topic_key
        
        # 👉 THE FIX: Calculate dynamic TTL
        subject = workbook_dict.get('subject', '').lower()
        topic = workbook_dict.get('topic', '').lower()
        dynamic_keywords = ["current affairs", "general awareness", "economy", "science & tech", "banking", "finance", "government schemes"]
        
        is_dynamic = any(k in subject or k in topic for k in dynamic_keywords)
        
        if is_dynamic:
            # Expire in 90 days for dynamic topics
            workbook_dict['expires_at'] = int(time.time()) + (90 * 86400)
            print("⏱️ Dynamic topic detected. Workbook cached with 90-day TTL.")
        else:
            # Expire in 5 years for static topics
            workbook_dict['expires_at'] = int(time.time()) + (1825 * 86400)
            print("⏱️ Static topic detected. Workbook cached with 5-year TTL.")
        
        dynamo_item = json.loads(json.dumps(workbook_dict), parse_float=Decimal)
        
        workbook_table.put_item(Item=dynamo_item)
        print(f"💾 Permanently cached workbook '{topic_key}' to DynamoDB.")
        return True
    except Exception as e:
        print(f"❌ Error saving workbook to DynamoDB: {e}")
        return False

# ==========================================
# PENDING TEST / SESSION RESUME LOGIC
# ==========================================

def save_pending_test(student_id: str, target_exam: str, test_config: Dict[str, Any], questions: List[Dict[str, Any]]) -> bool:
    """Saves an unsubmitted test to DynamoDB with a 1-hour TTL."""
    try:
        expiry_hours = int(os.environ.get('EXPIRY_HOURS', 1))
        expires_at = int(time.time()) + (expiry_hours * 3600) 
        
        pending_record = {
            "student_id": student_id,
            "target_exam": target_exam,
            "test_config": test_config,
            "questions": questions,
            "expires_at": expires_at,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        dynamo_item = json.loads(json.dumps(pending_record), parse_float=Decimal)
        pending_tests_table.put_item(Item=dynamo_item)
        print(f"💾 Saved pending test for {student_id} ({target_exam}). Expires in {expiry_hours} hour(s).")
        return True
    except Exception as e:
        print(f"❌ Error saving pending test: {e}")
        return False

def get_pending_test(student_id: str, target_exam: str) -> Optional[Dict[str, Any]]:
    """Retrieves a pending test if it exists and hasn't expired."""
    try:
        response = pending_tests_table.get_item(Key={
            'student_id': student_id,
            'target_exam': target_exam
        })
        
        item = response.get('Item')
        if not item:
            return None
            
        # Manually enforce TTL in case AWS hasn't physically deleted the ghost record yet
        if item.get('expires_at', 0) < int(time.time()):
            print(f"⚠️ Found expired ghost record for {student_id}. Ignoring.")
            return None
            
        return item
    except Exception as e:
        print(f"❌ Error fetching pending test: {e}")
        return None

def delete_pending_test(student_id: str, target_exam: str) -> bool:
    """Deletes a pending test after successful evaluation or manual overwrite."""
    try:
        pending_tests_table.delete_item(Key={
            'student_id': student_id,
            'target_exam': target_exam
        })
        print(f"🗑️ Deleted pending test for {student_id} ({target_exam}).")
        return True
    except Exception as e:
        print(f"❌ Error deleting pending test: {e}")
        return False