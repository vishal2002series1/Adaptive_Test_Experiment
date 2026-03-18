import streamlit as st
import requests
import json
import threading
import time

# Safely import Streamlit's context manager for background threading
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    add_script_run_ctx = None

# --- CONFIGURATION ---
API_URL = "https://2e7o2dai4vbtqzvs7x7qtcwuni0cqrim.lambda-url.us-east-1.on.aws/"

st.set_page_config(page_title="AI Adaptive Test Engine", layout="centered")

# --- LATEX FORMATTER ---
def format_latex(text):
    if not text or not isinstance(text, str): 
        return text
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\backslash ', '\\')
    text = text.replace(r'\backslash', '\\')
    text = text.replace(r'^{\wedge}', '^')
    text = text.replace(r'\wedge', '^')
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    return text

# --- DYNAMIC LOADER THREAD ---
def update_loading_text(placeholder, stop_event):
    """Cycles through dynamic agent statuses."""
    messages = [
        "⏳ Planner Agent is analyzing your historical weaknesses...",
        "🔍 Librarian Agent is checking OpenSearch for existing matches...",
        "✍️ Generator Agent is drafting highly calibrated questions...",
        "🧐 Critic Agent is running quality assurance and formatting checks...",
        "🔄 Agents are refining the question batch..."
    ]
    i = 0
    while not stop_event.is_set():
        placeholder.info(messages[i % len(messages)])
        time.sleep(2.5)
        i += 1

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
if 'fetched_profile_data' not in st.session_state:
    st.session_state.fetched_profile_data = None

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
        exam_preset = st.selectbox("Target Exam", ["JEE MAINS", "UPSC", "AWS SOLUTIONS ARCHITECT", "SSC CGL", "Custom Exam..."])
        if exam_preset == "Custom Exam...":
            target_exam = st.text_input("Enter Custom Exam Name", value="CAT Exam")
        else:
            target_exam = exam_preset

    st.divider()

    st.subheader("2. Testing Mode")
    adaptive_mode = st.toggle("Enable Adaptive Learning Mode", value=True)
    
    target_subject = "Entire Syllabus"
    target_topic = "All Syllabus"
    target_difficulty = 3
    num_questions = 5
    selected_override_topics = []

    if not adaptive_mode:
        st.info("Manual Mode: Select exactly what you want to study.")
        col3, col4 = st.columns(2)
        with col3:
            target_subject = st.text_input("Target Subject", value="Quantitative Aptitude")
            target_topic = st.text_input("Target Topic", value="Arithmetic")
        with col4:
            target_difficulty = st.slider("Difficulty Level", 1, 5, 3)
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
    
    else:
        st.success("Adaptive Mode: The AI will target your weak areas across the ENTIRE exam.")
        
        use_specific_subject = st.checkbox("I want to target a specific subject (Optional)")
        if use_specific_subject:
            target_subject = st.text_input("Broad Target Subject", value="Quantitative Aptitude", help="Tell the AI to strictly focus on this specific subject within the exam.")
        
        col_diff, col_num = st.columns(2)
        with col_diff:
            target_difficulty = st.slider("Difficulty Level", 1, 5, 3)
        with col_num:
            num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)

        if st.button("🔍 Load My Learning Profile", type="secondary"):
            with st.spinner("Fetching historical data..."):
                payload = {
                    "action": "get_profile",
                    "student_profile": {"student_id": student_id, "target_exam": target_exam}
                }
                try:
                    res = requests.post(API_URL, json=payload, timeout=30)
                    if res.status_code == 200:
                        st.session_state.fetched_profile_data = res.json().get('profile', {})
                    else:
                        st.error(f"Failed to load profile: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.fetched_profile_data:
            prof_data = st.session_state.fetched_profile_data
            proficiencies = prof_data.get('topic_proficiencies', {})
            
            st.markdown("### 📊 Your Progress Dashboard")
            st.metric("Total Tests Taken", prof_data.get('tests_taken', 0))
            st.progress(prof_data.get('overall_readiness_score', 0.0), text=f"Overall Exam Readiness: {prof_data.get('overall_readiness_score', 0.0):.2f}/1.0")
            
            if not proficiencies:
                st.info("No historical topics found for this exam. The AI will start exploring new topics for you!")
            else:
                st.markdown("#### Topic Mastery & Next Test Selection")
                st.write("Review your progress. The AI has pre-selected your 3 weakest topics to focus on next, but you can change them.")
                
                sorted_topics = sorted(proficiencies.items(), key=lambda x: x[1])
                auto_select_topics = [t[0] for t in sorted_topics[:3]]
                
                for topic, score in proficiencies.items():
                    col_chk, col_prog = st.columns([1, 4])
                    with col_chk:
                        is_checked = st.checkbox(topic, value=(topic in auto_select_topics))
                        if is_checked:
                            selected_override_topics.append(topic)
                    with col_prog:
                        st.progress(score, text=f"{score:.2f}/1.0")

    st.divider()
    if st.button("Generate Test 🚀", use_container_width=True, type="primary"):
        status_placeholder = st.empty()
        stop_event = threading.Event()
        
        if add_script_run_ctx:
            t = threading.Thread(target=update_loading_text, args=(status_placeholder, stop_event))
            add_script_run_ctx(t)
            t.start()
        else:
            status_placeholder.info("⏳ AI Agents are collaborating to build your test...")
            
        payload = {
            "action": "generate",
            "student_profile": {
                "student_id": student_id,
                "target_exam": target_exam
            },
            "test_config": {
                "target_subject": target_subject,
                "target_topic": target_topic if not adaptive_mode else "Auto-Selected",
                "target_difficulty": target_difficulty,
                "num_questions": num_questions,
                "adaptive_mode": adaptive_mode,
                "override_topics": selected_override_topics if adaptive_mode else None 
            }
        }
        
        try:
            response = requests.post(API_URL, json=payload, timeout=900)
            stop_event.set()
            status_placeholder.empty()
            
            if response.status_code == 200:
                data = response.json()
                st.session_state.questions = data.get("questions", [])
                st.session_state.student_profile = payload["student_profile"]
                st.session_state.user_answers = {} 
                st.session_state.phase = 'testing'
                st.rerun()
            else:
                st.error(f"Test Generation Failed: {response.json().get('error', response.text)}")
        except Exception as e:
            stop_event.set()
            status_placeholder.empty()
            st.error(f"Connection Error: {e}")

# ==========================================
# PHASE 2: TEST TAKING
# ==========================================
elif st.session_state.phase == 'testing':
    st.title(f"📝 {st.session_state.student_profile['target_exam']} Test")
    st.write("Please answer the following questions.")
    st.divider()
    
    # Track which shared contexts we've already displayed to avoid repetition
    rendered_contexts = set()
    
    for i, q in enumerate(st.session_state.questions):
        context = q.get('shared_context')
        
        # If the question belongs to a shared block, print the context once
        if context and context not in rendered_contexts:
            st.markdown("### 📖 Shared Context Passage")
            st.info(format_latex(context))
            rendered_contexts.add(context)
            
        st.markdown(f"**Q{i+1}.** {format_latex(q['text'])}")
        options_list = [f"{key}: {format_latex(val)}" for key, val in q['options'].items()]
        
        choice = st.radio(
            f"Select answer for Q{i+1}:", 
            options_list, 
            key=f"radio_{q['id']}", 
            index=None, 
            label_visibility="collapsed"
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
        
        # `text` now solely contains the specific question prompt
        display_text = original_q.get('text', 'Unknown') if original_q else "Unknown"
        
        with st.expander(f"Question: {format_latex(display_text[:60])}..."):
            # Provide a hint if this belonged to a larger context block
            if original_q and original_q.get('shared_context'):
                st.caption("*(This question was part of a Shared Context Passage)*")
                
            st.markdown(f"**Question:** {format_latex(display_text)}")
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
        st.session_state.fetched_profile_data = None 
        st.rerun()