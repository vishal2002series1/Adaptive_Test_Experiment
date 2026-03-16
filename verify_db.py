import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

host = 'ol4glcqiz2kh3n81yby7.us-east-1.aoss.amazonaws.com'
region = 'us-east-1'
service = 'aoss'
INDEX_NAME = "upsc-adaptive-questions"

# 1. Force a check on Python's active AWS Identity
session = boto3.Session()
credentials = session.get_credentials()
try:
    caller_arn = session.client('sts').get_caller_identity()['Arn']
    print(f"🔑 Python is authenticating to AWS as: {caller_arn}\n")
except Exception as e:
    print(f"❌ Could not verify AWS credentials. Ensure you are logged in. Error: {e}")
    exit()

# 2. Connect to OpenSearch
auth = AWSV4SignerAuth(credentials, region, service)
client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=30
)

# 3. Check if the index actually exists
try:
    print("📂 Checking available OpenSearch Indices...")
    indices = client.cat.indices(format="json")
    if not indices:
        print("   -> No indices found! The database is completely empty.")
        print("   -> This means the Lambda failed to create the index during the last curl test (likely because the Lambda's own security policy was still propagating).")
    else:
        for idx in indices:
            print(f"   -> Found index: {idx['index']} (Docs: {idx['docs.count']})")
except Exception as e:
    print(f"❌ Failed to list indices (Data Access Policy is likely still propagating): {e}\n")

# 4. Search for the Questions
try:
    print(f"\n🔍 Searching for questions in '{INDEX_NAME}'...")
    query = {"query": {"match_all": {}}, "size": 10}
    response = client.search(index=INDEX_NAME, body=query)
    
    total_docs = response['hits']['total']['value']
    print(f"✅ SUCCESS! Found {total_docs} questions stored in the Vector Database.\n")
    
    for hit in response['hits']['hits']:
        source = hit['_source']
        print(f"ID: {hit['_id']} | Topic: {source.get('topic')}")
        print(f"Preview: {source.get('text')[:75]}...\n" + "-"*50)

except Exception as e:
    print(f"❌ Search Error: {e}")