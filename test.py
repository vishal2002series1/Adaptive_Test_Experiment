import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Fetch the key and initialize the client explicitly
# This guarantees it doesn't fail if auto-detection misses it
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def generate_upsc_question(topic, recency_instructions):
    print("Generating grounded question using Gemini Search...")
    
    response = client.models.generate_content(
        model='gemini-3.1-pro-preview',
        contents=f"Generate a UPSC Prelims multi-statement question about {topic}. {recency_instructions}",
        config=types.GenerateContentConfig(
            # Activates native Google Search Grounding
            tools=[{"google_search": {}}],
            temperature=0.2, 
        )
    )
    
    return response.text

# Example usage
draft = generate_upsc_question(
    topic="India's current monetary policy and repo rates", 
    recency_instructions="Use data from the last 3 months."
)
print(draft)