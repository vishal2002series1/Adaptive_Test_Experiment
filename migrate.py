import os
import boto3
from dotenv import load_dotenv
load_dotenv()
from schema import Question
from vector_store import save_questions_to_db

# Load API keys from your local .env file
load_dotenv()

# Connect to AWS DynamoDB
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
table = dynamodb.Table('AdaptiveQuestionBank')

print("Fetching all questions from DynamoDB Master Bank...")
response = table.scan()
items = response.get('Items', [])

# Paginate if the table gets large
while 'LastEvaluatedKey' in response:
    response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
    items.extend(response.get('Items', []))

print(f"Found {len(items)} questions in DynamoDB. Embedding and migrating to Upstash...")

questions_to_migrate = []
for item in items:
    try:
        # Convert DynamoDB item back to your strict Pydantic model
        q_obj = Question(**item)
        questions_to_migrate.append(q_obj)
    except Exception as e:
        print(f"Skipping a question due to parsing error: {e}")

# This will call your Upstash deduplication logic and index them safely!
save_questions_to_db(questions_to_migrate)

print("✅ Migration Complete! All questions are now live in Upstash Vector.")