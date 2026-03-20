import streamlit as st
import requests
import json
import threading
import time
from datetime import datetime

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

def parse_iso_date(iso_str):
    """Helper to convert ISO string to readable format"""
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return iso_str

# --- DYNAMIC LOADER THREAD ---
def update_loading_text(placeholder, stop_event, mode="test"):
    """Cycles through dynamic agent statuses based on the mode."""
    if mode == "workbook":
        messages = [
            "🔍 Researcher Agent is finding the best video resources...",
            "✍️ Author Agent is drafting theory and mnemonics...",
            "🎨 Designer Agent is coding the Mermaid.js visualization...",
            "🗂️ Curator Agent is pulling relevant practice questions...",
            "💾 Compiler is caching your workbook to DynamoDB..."
        ]
    else:
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
if 'history_logs' not in st.session_state:
    st.session_state.history_logs = []
if 'current_workbook' not in st.session_state:
    st.session_state.current_workbook = None

# ==========================================
# UI NAVIGATION TABS
# ==========================================
if st.session_state.phase == 'setup':
    tab_setup, tab_history, tab_learn = st.tabs(["🚀 Take a Test", "📂 Test History", "🧠 Study Modules"])
else:
    from contextlib import nullcontext
    tab_setup = nullcontext()
    tab_history = nullcontext()
    tab_learn = nullcontext()

# ==========================================
# PHASE 1: SETUP & CONFIGURATION
# ==========================================
with tab_setup:
    if st.session_state.phase == 'setup':
        st.title("📚 AI Adaptive Test Engine")
        
        st.subheader("1. User Profile")
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID", value="Physics Student V2")
        with col2:
            exam_preset = st.selectbox("Target Exam", ["UPSC CSE Prelims", "UPSC CSE Mains", "SSC CGL", "IBPS PO", "IBPS RRB PO", "RBI Grade B", "GATE CSE", "Custom Exam..."])
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
                manual_difficulty = st.checkbox("Set Difficulty Manually")
                if manual_difficulty:
                    target_difficulty = st.slider("Difficulty Level", 1, 5, 3)
                else:
                    target_difficulty = None
                    st.info("🧠 AI will dynamically set difficulty per topic.")
                    
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
                proficiencies = prof_data.get('proficiencies', [])
                
                st.markdown("### 📊 Your Progress Dashboard")
                st.metric("Total Tests Taken", prof_data.get('tests_taken', 0))
                st.progress(prof_data.get('overall_readiness_score', 0.0), text=f"Overall Exam Readiness: {prof_data.get('overall_readiness_score', 0.0):.2f}/1.0")
                
                last_plan = prof_data.get('last_study_plan')
                if last_plan:
                    with st.expander("📝 View Previous AI Study Plan", expanded=False):
                        st.markdown(last_plan)
                
                if not proficiencies:
                    st.info("No historical topics found for this exam. The AI will start exploring new topics for you!")
                else:
                    st.markdown("#### Sub-Topic Mastery & Next Test Selection")
                    st.write("Review your progress. The AI has pre-selected your 3 weakest sub-topics to focus on next, but you can change them.")
                    
                    sorted_profs = sorted(proficiencies, key=lambda x: x.get('score', 0.0))
                    auto_select_topics = [p.get('sub_topic') for p in sorted_profs[:3]]
                    
                    for p in proficiencies:
                        sub_topic = p.get('sub_topic', 'Unknown')
                        topic = p.get('topic', 'Unknown')
                        score = p.get('score', 0.0)
                        attempts = p.get('questions_attempted', 0)
                        
                        col_chk, col_prog = st.columns([1, 4])
                        with col_chk:
                            is_checked = st.checkbox(f"{sub_topic}", value=(sub_topic in auto_select_topics), key=f"chk_{sub_topic}")
                            if is_checked:
                                selected_override_topics.append(sub_topic)
                        with col_prog:
                            st.progress(score, text=f"{topic} > {sub_topic} | Score: {score:.2f} ({attempts} Qs)")

        st.divider()
        if st.button("Generate Test 🚀", use_container_width=True, type="primary"):
            status_placeholder = st.empty()
            stop_event = threading.Event()
            
            if add_script_run_ctx:
                t = threading.Thread(target=update_loading_text, args=(status_placeholder, stop_event, "test"))
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
                
                # 👉 UPDATE: Added safety net for Test Generation too
                try:
                    res_json = response.json()
                except Exception:
                    st.error(f"Backend returned non-JSON format! Raw Response: {response.text}")
                    res_json = {}
                
                if response.status_code == 200 and res_json:
                    st.session_state.questions = res_json.get("questions", [])
                    st.session_state.student_profile = payload["student_profile"]
                    st.session_state.user_answers = {} 
                    st.session_state.phase = 'testing'
                    st.rerun()
                elif res_json:
                    st.error(f"Test Generation Failed: {res_json.get('error', response.text)}")
            except Exception as e:
                stop_event.set()
                status_placeholder.empty()
                st.error(f"Connection Error: {e}")

# ==========================================
# PHASE 1.5: HISTORY TAB
# ==========================================
with tab_history:
    if st.session_state.phase == 'setup':
        st.title("📂 Your Past Tests")
        st.write("Review your historical performance and analyze past mistakes.")
        
        hist_col1, hist_col2 = st.columns([3, 1])
        with hist_col1:
            history_student_id = st.text_input("Enter Student ID to fetch history", value=student_id, key="hist_id")
        with hist_col2:
            st.write("") 
            st.write("") 
            fetch_history_btn = st.button("Fetch History 🔍", use_container_width=True)
            
        if fetch_history_btn:
            with st.spinner("Fetching test logs from DynamoDB..."):
                payload = {
                    "action": "get_history",
                    "student_profile": {"student_id": history_student_id}
                }
                try:
                    res = requests.post(API_URL, json=payload, timeout=30)
                    if res.status_code == 200:
                        st.session_state.history_logs = res.json().get('history', [])
                        if not st.session_state.history_logs:
                            st.info("No test history found for this Student ID.")
                    else:
                        st.error(f"Failed to load history: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
                    
        if st.session_state.history_logs:
            st.divider()
            for index, log in enumerate(st.session_state.history_logs):
                exam_name = log.get('exam', 'Unknown Exam')
                score = log.get('score_percentage', 0)
                date_str = parse_iso_date(log.get('timestamp', ''))
                
                with st.expander(f"📝 {exam_name} - {date_str} (Score: {score}%)"):
                    if log.get('study_plan'):
                        st.markdown("**AI Study Plan Provided:**")
                        st.info(log.get('study_plan'))
                        
                    st.markdown("**Questions & Answers:**")
                    for q_idx, result in enumerate(log.get('graded_results', [])):
                        question_text = result.get('text', 'Question text missing')
                        is_correct = result.get('is_correct', False)
                        icon = "✅" if is_correct else "❌"
                        
                        st.markdown(f"**Q{q_idx+1}.** {format_latex(question_text)}")
                        st.write(f"{icon} **Your Answer:** {result.get('student_answer')} | **Correct Answer:** {result.get('correct_answer')}")
                        st.caption(f"Taxonomy: {result.get('subject')} > {result.get('topic')} > {result.get('sub_topic')} | Difficulty: {result.get('difficulty')}")
                        
                        with st.popover("Show Explanation"):
                            st.markdown(format_latex(result.get('explanation', 'No explanation provided.')))
                        st.divider()

# ==========================================
# 👉 STUDY MODULES TAB
# ==========================================
with tab_learn:
    if st.session_state.phase == 'setup':
        st.title("🧠 AI Study Modules")
        st.write("Generate interactive workbooks for any topic to master weak areas.")
        
        wb_col1, wb_col2 = st.columns(2)
        with wb_col1:
            wb_subject = st.text_input("Subject", value="General Awareness")
            wb_topic = st.text_input("Topic", value="Economy")
        with wb_col2:
            wb_subtopic = st.text_input("Sub-Topic (Required)", value="Inflation")
            wb_difficulty = st.slider("Target Difficulty Level", 1, 5, 3, key="wb_diff")
            
        if st.button("Generate Study Module 📖", use_container_width=True, type="primary"):
            status_placeholder = st.empty()
            stop_event = threading.Event()
            
            if add_script_run_ctx:
                t = threading.Thread(target=update_loading_text, args=(status_placeholder, stop_event, "workbook"))
                add_script_run_ctx(t)
                t.start()
            else:
                status_placeholder.info("⏳ AI Agents are writing your workbook...")
                
            payload = {
                "action": "get_workbook",
                "student_profile": {
                    "student_id": student_id,
                    "target_exam": target_exam
                },
                "workbook_config": {
                    "subject": wb_subject,
                    "topic": wb_topic,
                    "sub_topic": wb_subtopic,
                    "difficulty_level": wb_difficulty
                }
            }
            
            try:
                response = requests.post(API_URL, json=payload, timeout=900)
                stop_event.set()
                status_placeholder.empty()
                
                # 👉 UPDATE: The Ultimate Safety Net. We now intercept invalid JSON and print the raw string.
                try:
                    res_json = response.json()
                except Exception as json_err:
                    st.error(f"AWS Failed to return JSON! Raw Server Response: {response.text}")
                    res_json = None
                
                if res_json:
                    if response.status_code == 200:
                        st.session_state.current_workbook = res_json.get('workbook')
                        st.success("✅ Workbook loaded successfully!")
                    else:
                        st.error(f"Generation Failed: {res_json.get('error', response.text)}")
            except Exception as e:
                stop_event.set()
                status_placeholder.empty()
                st.error(f"Connection Error: {e}")
                
        # --- DISPLAY THE WORKBOOK ---
        if st.session_state.current_workbook:
            wb = st.session_state.current_workbook
            st.divider()
            st.header(f"Module: {wb.get('sub_topic')}")
            st.caption(f"Difficulty Level: {wb.get('difficulty_level')} | Target Exam: {wb.get('target_exam')}")
            
            tab_theory, tab_visual, tab_tricks, tab_videos = st.tabs(["📖 Theory", "🗺️ Mind Map", "💡 Tricks", "🎥 Videos"])
            
            with tab_theory:
                st.markdown(format_latex(wb.get('theory_markdown', '')))
                
            with tab_visual:
                st.write("Visual breakdown of the concepts:")
                mermaid_code = wb.get('mermaid_graph_code', '')
                if mermaid_code:
                    st.markdown(f"```mermaid\n{mermaid_code}\n```")
                else:
                    st.info("No visualization available for this topic.")
                    
            with tab_tricks:
                st.markdown(format_latex(wb.get('tricks_and_mnemonics', '')))
                
            with tab_videos:
                videos = wb.get('video_references', [])
                if videos:
                    for v in videos:
                        with st.container(border=True):
                            st.markdown(f"**[{v.get('title', 'Video')}]({v.get('url', '#')})**")
                            st.write(v.get('why_watch_this', ''))
                else:
                    st.info("No video recommendations available.")
                    
            q_ids = wb.get('practice_question_ids', [])
            if q_ids:
                st.success(f"🎯 The Curator found **{len(q_ids)}** practice questions matching this module in your database. Switch to 'Take a Test' mode to practice them!")

# ==========================================
# PHASE 2: TEST TAKING
# ==========================================
if st.session_state.phase == 'testing':
    st.title(f"📝 {st.session_state.student_profile['target_exam']} Test")
    st.write("Please answer the following questions.")
    st.divider()
    
    rendered_contexts = set()
    
    for i, q in enumerate(st.session_state.questions):
        context = q.get('shared_context')
        
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
                    
                    try:
                        res_json = eval_response.json()
                    except Exception:
                        st.error(f"AWS returned non-JSON format! Raw: {eval_response.text}")
                        res_json = {}
                        
                    if eval_response.status_code == 200 and res_json:
                        st.session_state.evaluation = res_json
                        st.session_state.phase = 'results'
                        st.rerun()
                    elif res_json:
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
        
        display_text = original_q.get('text', 'Unknown') if original_q else "Unknown"
        
        with st.expander(f"Question: {format_latex(display_text[:60])}..."):
            if original_q and original_q.get('shared_context'):
                st.caption("*(This question was part of a Shared Context Passage)*")
                
            st.markdown(f"**Question:** {format_latex(display_text)}")
            if result["is_correct"]:
                st.success(f"✅ Your Answer: {result['student_answer']} (Correct)")
            else:
                st.error(f"❌ Your Answer: {result['student_answer']} | Correct Answer: {result['correct_answer']}")
            
            st.markdown(f"**Explanation:** {format_latex(original_q['explanation'])}")
            
            if original_q:
                meta = original_q.get('metadata', {})
                st.caption(f"Taxonomy: {meta.get('subject', 'N/A')} > {meta.get('topic', 'N/A')} > {meta.get('sub_topic', 'N/A')} | Source: {meta.get('taxonomy_source', 'N/A')}")
            st.caption(f"Score Delta: {result.get('score_delta', 'N/A')}")

    st.divider()
    if st.button("Take Another Test 🔄"):
        st.session_state.phase = 'setup'
        st.session_state.questions = []
        st.session_state.user_answers = {}
        st.session_state.evaluation = None
        st.session_state.fetched_profile_data = None 
        st.session_state.history_logs = []
        st.session_state.current_workbook = None
        st.rerun()