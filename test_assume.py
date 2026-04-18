import boto3
import json

session = boto3.Session(profile_name="account1-via-account2")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="us.anthropic.claude-sonnet-4-6",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": "Hello from boto3!"}]
    })
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"])