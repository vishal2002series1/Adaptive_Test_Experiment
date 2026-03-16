import os
import boto3
import json
from decimal import Decimal
from typing import Dict, Any
from schema import StudentProfile

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
            # DynamoDB returns Decimals, which Pydantic happily casts back to floats.
            # If 'explored_topics' is missing from old DB records, Pydantic's default_factory handles it automatically.
            return StudentProfile(**response['Item'])
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch from DynamoDB ({e}). Using default profile.")
    
    return StudentProfile(student_id=student_id, target_exam=target_exam)

def save_student_profile(profile: StudentProfile) -> bool:
    """Saves the updated student profile back to DynamoDB."""
    try:
        item = profile.model_dump()
        
        # --- Boto3 Float to Decimal Converter ---
        # Convert the dict to a JSON string, then parse it back into a dict 
        # while forcing all floating-point numbers to become Decimals for DynamoDB.
        dynamo_item = json.loads(json.dumps(item), parse_float=Decimal)
        
        table.put_item(Item=dynamo_item)
        print(f"💾 Successfully saved {profile.target_exam} profile for {profile.student_id} to DynamoDB.")
        return True
    except Exception as e:
        print(f"❌ Error saving to DynamoDB: {e}")
        return False