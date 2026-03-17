import streamlit as st
import requests

# --- CONFIGURATION ---
API_URL = "https://2e7o2dai4vbtqzvs7x7qtcwuni0cqrim.lambda-url.us-east-1.on.aws/"

st.set_page_config(page_title="AI Adaptive Test Engine", layout="centered")

# --- LATEX FORMATTER ---
def format_latex(text):
    """Cleans LLM hallucinations and converts delimiters to Streamlit-compatible LaTeX."""
    if not text or not isinstance(text, str): 
        return text
        
    # 1. Convert standard LLM math brackets to Streamlit $ signs
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    
    # 2. Scrub LLM escaping hallucinations
    text = text.replace(r'\backslash ', '\\')
    text = text.replace(r'\backslash', '\\')
    text = text.replace(r'^{\wedge}', '^')
    text = text.replace(r'\wedge', '^')
    
    # Fix broken fractions
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    
    return text

# --- STATE MANAGEMENT ---
if 'phase' not in st.session_state:
    st.session_state.phase = 'setup'
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'evaluation' not in st.session_state:
    st.session_state.evaluation = None
if 'student_profile' not in st.session_state:
    st.session_state.student_profile = {}

# ==========================================
# PHASE 1: SETUP & CONFIGURATION
# ==========================================
if st.session_state.phase == 'setup':
    st.title("📚 AI Adaptive Test Engine")
    st.subheader("1. User Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Student ID", value="Physics Student V2")
    with col2:
        target_exam = st.selectbox("Target Exam", ["JEE MAINS", "UPSC", "AWS SOLUTIONS ARCHITECT", "SSC CGL"])
        
    st.subheader("2. Test Configuration")
    adaptive_mode = st.toggle("Enable Adaptive Mode (Targets Weaknesses)", value=True)
    
    col3, col4 = st.columns(2)
    with col3:
        target_subject = st.text_input("Target Subject", value="Physics")
        target_topic = st.text_input("Target Topic", value="Kinematics")
    with col4:
        target_difficulty = st.slider("Difficulty Level", 1, 5, 3)
        num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

    if st.button("Generate Test 🚀", use_container_width=True):
        with st.spinner("Generating highly calibrated questions via LangGraph..."):
            payload = {
                "action": "generate",
                "student_profile": {
                    "student_id": student_id,
                    "target_exam": target_exam
                },
                "test_config": {
                    "target_subject": target_subject,
                    "target_topic": target_topic,
                    "target_difficulty": target_difficulty,
                    "num_questions": num_questions,
                    "adaptive_mode": adaptive_mode
                }
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=900)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.questions = data.get("questions", [])
                    st.session_state.student_profile = payload["student_profile"]
                    st.session_state.user_answers = {} 
                    st.session_state.phase = 'testing'
                    st.rerun()
                else:
                    st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# ==========================================
# PHASE 2: TEST TAKING
# ==========================================
elif st.session_state.phase == 'testing':
    st.title(f"📝 {st.session_state.student_profile['target_exam']} Test")
    st.write("Please answer the following questions.")
    st.divider()
    
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}.** {format_latex(q['text'])}")
        
        options_list = [f"{key}: {format_latex(val)}" for key, val in q['options'].items()]
        
        choice = st.radio(
            f"Select answer for Q{i+1}:", 
            options_list, 
            key=f"radio_{q['id']}",
            index=None 
        )
        
        if choice:
            st.session_state.user_answers[q['id']] = choice.split(":")[0]
            
        st.divider()
        
    if st.button("Submit Exam 📋", type="primary", use_container_width=True):
        if len(st.session_state.user_answers) < len(st.session_state.questions):
            st.warning("Please answer all questions before submitting!")
        else:
            with st.spinner("Evaluating your answers and updating your DynamoDB profile..."):
                eval_payload = {
                    "action": "evaluate",
                    "student_profile": st.session_state.student_profile,
                    "questions": st.session_state.questions,
                    "student_answers": st.session_state.user_answers
                }
                
                try:
                    eval_response = requests.post(API_URL, json=eval_payload, timeout=300)
                    if eval_response.status_code == 200:
                        st.session_state.evaluation = eval_response.json()
                        st.session_state.phase = 'results'
                        st.rerun()
                    else:
                        st.error(f"Evaluation Error: {eval_response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# ==========================================
# PHASE 3: EVALUATION & RESULTS
# ==========================================
elif st.session_state.phase == 'results':
    st.title("🎯 Test Results")
    eval_data = st.session_state.evaluation
    
    score = eval_data.get("score_percentage", 0)
    st.metric(label="Overall Score", value=f"{score}%")
    
    st.subheader("💡 AI Study Plan & Recommendations")
    st.info(eval_data.get("study_plan", "No study plan provided."))
    
    st.subheader("Detailed Breakdown")
    for result in eval_data.get("graded_results", []):
        q_id = result["question_id"]
        original_q = next((q for q in st.session_state.questions if q["id"] == q_id), None)
        
        with st.expander(f"Question: {format_latex(original_q['text'][:60])}..."):
            st.markdown(f"**Full Question:** {format_latex(original_q['text'])}")
            if result["is_correct"]:
                st.success(f"✅ Your Answer: {result['student_answer']} (Correct)")
            else:
                st.error(f"❌ Your Answer: {result['student_answer']} | Correct Answer: {result['correct_answer']}")
            
            st.markdown(f"**Explanation:** {format_latex(original_q['explanation'])}")
            
            st.caption(f"Topic: {result.get('topic', 'N/A')} | Score Delta: {result.get('score_delta', 'N/A')}")

    st.divider()
    if st.button("Take Another Test 🔄"):
        st.session_state.phase = 'setup'
        st.session_state.questions = []
        st.session_state.user_answers = {}
        st.session_state.evaluation = None
        st.rerun()