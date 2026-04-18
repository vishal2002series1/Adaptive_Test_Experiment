import os
import boto3
from aws_assume_role_lib import assume_role
from botocore.config import Config

# Fallbacks provided for local testing if env vars aren't set
TARGET_ROLE_ARN = os.environ.get("BEDROCK_TARGET_ROLE_ARN", "arn:aws:iam::258574424891:role/CrossAccountAdminRole")
EXTERNAL_ID = os.environ.get("BEDROCK_EXTERNAL_ID", "92F110A6-4E94-4359-82F3-C718DD537623")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Create the auto-refreshing session once at module load
_base_session = boto3.Session()
_assumed_session = assume_role(
    _base_session,
    RoleArn=TARGET_ROLE_ARN,
    RoleSessionName="adaptive-test-bedrock-session",
    ExternalId=EXTERNAL_ID,
)

# 👉 THE FIX: Configure boto3 to wait up to 5 minutes (300s) for Claude to finish 
# generating large reports, and disable the silent retries to prevent loops!
custom_bedrock_config = Config(
    read_timeout=1500,
    connect_timeout=30,
    retries={"max_attempts": 0} 
)

# Export this client to be used by all other files
bedrock_runtime = _assumed_session.client(
    "bedrock-runtime", 
    region_name=REGION, 
    config=custom_bedrock_config
)