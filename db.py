import os
import boto3
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List
from boto3.dynamodb.conditions import Key
from schema import StudentProfile, ProficiencyRecord

dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

# 1. Existing Profile Table
TABLE_NAME = os.environ.get('STUDENT_TABLE', 'AdaptiveStudentProfiles') 
table = dynamodb.Table(TABLE_NAME)

# 2. 👉 NEW: History Logs Table
HISTORY_TABLE_NAME = os.environ.get('HISTORY_TABLE', 'AdaptiveTestHistoryLogs')
history_table = dynamodb.Table(HISTORY_TABLE_NAME)

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

# ==========================================
# NEW: TEST HISTORY LOGIC
# ==========================================

def save_test_history(student_id: str, exam: str, score: float, graded_results: list, study_plan: str) -> bool:
    """Saves a completed test log to the DynamoDB history table."""
    try:
        # ISO format ensures strict chronological sorting when queried
        timestamp = datetime.now(timezone.utc).isoformat()
        
        history_record = {
            "student_id": student_id,
            "timestamp": timestamp,
            "exam": exam,
            "score_percentage": score,
            "graded_results": graded_results,
            "study_plan": study_plan
        }
        
        # Boto3 Float to Decimal Converter
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
            ScanIndexForward=False  # 👉 Ensures the newest test is at index 0!
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"❌ Error fetching test history from DynamoDB: {e}")
        return []