import json
import requests
import random

# 1. Load the generated test from Step 1
try:
    with open('50_question_test.json', 'r') as f:
        response_data = json.load(f)
except FileNotFoundError:
    print("❌ Error: 50_question_test.json not found. Run the curl command first!")
    exit()

questions = response_data.get('questions', [])
if not questions:
    print("❌ Error: No questions found in the JSON file. The generation might have failed.")
    exit()

# 2. Simulate a student taking the test (Randomly picking A, B, C, or D)
student_answers = {q['id']: random.choice(["A", "B", "C", "D"]) for q in questions}

# 3. Build the evaluation payload
payload = {
    "action": "evaluate",
    "student_profile": {
        "student_id": "student_123",
        "target_exam": "UPSC"
    },
    "student_answers": student_answers,
    "questions": questions
}

# 4. Send it to your live AWS architecture
url = "https://2e7o2dai4vbtqzvs7x7qtcwuni0cqrim.lambda-url.us-east-1.on.aws/"
print(f"🚀 Submitting {len(questions)} answers to AWS for evaluation...")
response = requests.post(url, json=payload)

# 5. Save the Strategist's study plan and grading results
with open('50_question_results.json', 'w') as f:
    json.dump(response.json(), f, indent=2)

print("✅ Evaluation complete! Open 50_question_results.json to see the AI study plan.")