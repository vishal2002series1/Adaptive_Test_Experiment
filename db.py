import os
import boto3
import json
from decimal import Decimal
from typing import Dict, Any
from schema import StudentProfile, ProficiencyRecord

dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
TABLE_NAME = os.environ.get('STUDENT_TABLE', 'AdaptiveStudentProfiles') 
table = dynamodb.Table(TABLE_NAME)

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
            # If we detect the old flat dictionary, we instantly transform it 
            # into the new structured array so Pydantic validates cleanly.
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
        
        # --- Boto3 Float to Decimal Converter ---
        dynamo_item = json.loads(json.dumps(item), parse_float=Decimal)
        
        table.put_item(Item=dynamo_item)
        print(f"💾 Successfully saved {profile.target_exam} profile for {profile.student_id} to DynamoDB.")
        return True
    except Exception as e:
        print(f"❌ Error saving to DynamoDB: {e}")
        return False