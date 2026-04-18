import streamlit as st
import requests
import json
import threading
import time
import re
import os
import tempfile
from datetime import datetime
import streamlit.components.v1 as components

# Safely import Streamlit's context manager for background threading
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
except ImportError:
    add_script_run_ctx = None

# # --- CONFIGURATION ---
# API_URL = "https://2e7o2dai4vbtqzvs7x7qtcwuni0cqrim.lambda-url.us-east-1.on.aws/"

# --- CONFIGURATION ---
API_URL = "https://4sbvnmfkgevegvyu5s4aqc7b6y0wipxg.lambda-url.us-east-1.on.aws/"
st.set_page_config(page_title="AI Adaptive Test Engine", layout="centered")

# --- EPHEMERAL STORAGE FOR ECS ---
# Create a secure temporary directory on the ECS container for the uploaded images
TEMP_IMG_DIR = os.path.join(tempfile.gettempdir(), "streamlit_uploaded_images")
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# --- LATEX FORMATTER ---
# --- LATEX FORMATTER ---
def format_latex(text):
    if not text or not isinstance(text, str): 
        return text
        
    # 1. Fix literal escaped newlines and unicode artifacts from JSON
    text = text.replace('\\n', '\n')
    text = text.replace('\\u2014', '—') # Em-dash
    text = text.replace('\\u2013', '–') # En-dash
    text = text.replace('\\u2022', '•') # Bullet point
    text = text.replace('\\u00a0', ' ') # Non-breaking space
    
    # 2. Convert LaTeX Itemize/Enumerate Environments to Markdown Lists
    text = re.sub(r'\\+begin\s*{(?:itemize|enumerate)}', '\n', text)
    text = re.sub(r'\\+end\s*{(?:itemize|enumerate)}', '\n', text)
    text = re.sub(r'\\+item\s*', '* ', text)
    
    # 3. Handle LaTeX line breaks and arrows
    text = re.sub(r'\\+newline', '\n', text)
    text = re.sub(r'\\+rightarrow', '→', text)
    text = re.sub(r'\\+leftarrow', '←', text)
    text = re.sub(r'\\+Rightarrow', '⇒', text)
    text = re.sub(r'\\+Leftarrow', '⇐', text)
    
    # Fix multiple backslashes acting as line breaks (e.g., \\\\textbf -> \n\n\textbf)
    text = text.replace(r'\\\\\\', '\n\n\\') 
    
    # 4. Convert LaTeX text styling to Markdown (using highly robust regex)
    text = re.sub(r'\\+textbf{(.*?)}', r'**\1**', text, flags=re.DOTALL)
    text = re.sub(r'\\+textit{(.*?)}', r'*\1*', text, flags=re.DOTALL)
    text = re.sub(r'\\+texttt{(.*?)}', r'`\1`', text, flags=re.DOTALL)
    
    # 5. Handle math blocks and escaping
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')
    text = text.replace(r'\backslash ', '\\')
    text = text.replace(r'\backslash', '\\')
    text = text.replace(r'^{\wedge}', '^')
    text = text.replace(r'\wedge', '^')
    text = text.replace(r'\{', '{').replace(r'\}', '}')
    
    return text

def parse_iso_date(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return iso_str

# --- DYNAMIC SYLLABUS LOADER & UI COMPONENTS ---
@st.cache_data
def load_syllabus_map():
    try:
        with open("syllabus_maps.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def dynamic_select(label, options, default_value, key):
    opts = [str(opt) for opt in options if opt] 
    if "Other..." not in opts:
        opts.append("Other...")
        
    try:
        index = opts.index(default_value)
        default_text = ""
    except ValueError:
        index = opts.index("Other...")
        default_text = default_value if default_value else ""
        
    selected = st.selectbox(label, opts, index=index, key=f"{key}_select")
    
    if selected == "Other...":
        return st.text_input(f"Enter Custom {label}", value=default_text, key=f"{key}_text")
    return selected

def render_markmap(markdown_content: str):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: sans-serif; background-color: transparent; overflow: hidden; }}
            svg {{ width: 100vw; height: 100vh; }}
            .markmap-toolbar {{ position: absolute; bottom: 20px; right: 20px; background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-lib@0.16.0"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-view@0.16.0"></script>
        <script src="https://cdn.jsdelivr.net/npm/markmap-toolbar@0.16.0"></script>
    </head>
    <body>
        <textarea id="md-content" style="display:none;">{markdown_content}</textarea>
        <svg id="markmap"></svg>
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const markdown = document.getElementById('md-content').value;
                const {{ markmap }} = window;
                const {{ Transformer, Markmap, Toolbar }} = markmap;
                
                const transformer = new Transformer();
                const {{ root }} = transformer.transform(markdown);
                
                const mm = Markmap.create('#markmap', {{ autoFit: true, duration: 500 }}, root);
                
                const toolbar = new Toolbar();
                toolbar.attach(mm);
                const toolbarEl = toolbar.render();
                document.body.appendChild(toolbarEl);
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=600, scrolling=False)

def update_loading_text(placeholder, stop_event, mode="test"):
    if mode == "workbook":
        messages = [
            "🔍 Researcher Agent is finding the best video resources...",
            "✍️ Author Agent is drafting theory and mnemonics...",
            "🎨 Designer Agent is mapping the interactive mind map...",
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

def request_workbook_generation(student_id, target_exam, subject, topic, sub_topic, difficulty):
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
        "student_profile": {"student_id": student_id, "target_exam": target_exam},
        "workbook_config": {
            "subject": subject,
            "topic": topic,
            "sub_topic": sub_topic,
            "difficulty_level": difficulty
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=900)
        stop_event.set()
        status_placeholder.empty()
        
        try:
            res_json = response.json()
        except Exception:
            st.error(f"AWS Failed to return JSON! Raw Server Response: {response.text}")
            return None
        
        if res_json and response.status_code == 200:
            return res_json.get('workbook')
        elif res_json:
            st.error(f"Generation Failed: {res_json.get('error', response.text)}")
            return None
    except Exception as e:
        stop_event.set()
        status_placeholder.empty()
        st.error(f"Connection Error: {e}")
        return None

def render_workbook_ui(wb):
    st.divider()
    st.header(f"Module: {wb.get('sub_topic')}")
    st.caption(f"Difficulty Level: {wb.get('difficulty_level')} | Target Exam: {wb.get('target_exam')}")
    
    tab_theory, tab_visual, tab_tricks, tab_videos, tab_practice = st.tabs(["📖 Theory", "🗺️ Mind Map", "💡 Tricks", "🎥 Videos", "📝 Practice"])
    
    with tab_theory:
        st.markdown(format_latex(wb.get('theory_markdown', '')))
        
    with tab_visual:
        st.write("💡 **Interactive Mind Map:** Use your mouse to drag/zoom, and click on the circles to expand or collapse learning branches for active recall!")
        map_data = wb.get('mermaid_graph_code', '')
        if map_data:
            render_markmap(map_data)
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
            
    with tab_practice:
        questions = wb.get('practice_questions', [])
        if questions:
            st.write("Test your understanding of this module:")
            for i, q_dict in enumerate(questions):
                with st.container(border=True):
                    st.markdown(f"**Q{i+1}.** {format_latex(q_dict.get('text', ''))}")
                    
                    options = q_dict.get('options', {})
                    for key, val in options.items():
                        st.write(f"**{key}:** {format_latex(val)}")
                        
                    with st.expander("👀 Show Answer & Explanation"):
                        st.success(f"**Correct Answer: {q_dict.get('correct_answer')}**")
                        st.markdown(format_latex(q_dict.get('explanation', '')))
        else:
            st.info("No specific practice questions are mapped to this module yet. Try generating more tests in this subject!")

# --- STATE MANAGEMENT ---
if 'phase' not in st.session_state: st.session_state.phase = 'setup'
if 'questions' not in st.session_state: st.session_state.questions = []
if 'user_answers' not in st.session_state: st.session_state.user_answers = {}
if 'evaluation' not in st.session_state: st.session_state.evaluation = None
if 'student_profile' not in st.session_state: st.session_state.student_profile = {}
if 'fetched_profile_data' not in st.session_state: st.session_state.fetched_profile_data = None
if 'history_logs' not in st.session_state: st.session_state.history_logs = []

if 'current_workbook' not in st.session_state: st.session_state.current_workbook = None
if 'active_history_wb' not in st.session_state: st.session_state.active_history_wb = None
if 'active_results_wb' not in st.session_state: st.session_state.active_results_wb = None
if 'session_restored' not in st.session_state: st.session_state.session_restored = False

# Admin HITL States
if 'admin_mode' not in st.session_state: st.session_state.admin_mode = "Student Platform"
if 'review_queue' not in st.session_state: st.session_state.review_queue = []
if 'review_decisions' not in st.session_state: st.session_state.review_decisions = {}
if 'current_review_idx' not in st.session_state: st.session_state.current_review_idx = 0

# --- DASHBOARD CALLBACK FUNCTION ---
def load_scope_to_form(exam, scope, subject="Entire Syllabus", topic="All Syllabus"):
    st.session_state.dash_exam = exam
    st.session_state.dash_scope = scope
    st.session_state.dash_subject = subject
    st.session_state.dash_topic = topic
    st.toast(f"✅ Loaded {scope} configuration into the test builder!")

# Initialize default states for the form
if 'dash_exam' not in st.session_state: st.session_state.dash_exam = "UPSC CSE Prelims"
if 'dash_scope' not in st.session_state: st.session_state.dash_scope = "Full Exam"
if 'dash_subject' not in st.session_state: st.session_state.dash_subject = "Entire Syllabus"
if 'dash_topic' not in st.session_state: st.session_state.dash_topic = "All Syllabus"
if 'dashboard_data' not in st.session_state: st.session_state.dashboard_data = None

# ==========================================
# APP ROUTING (SIDEBAR)
# ==========================================
with st.sidebar:
    st.title("Navigation")
    st.session_state.admin_mode = st.radio("Select Portal:", ["Student Platform", "Admin Portal (HITL)"])

if st.session_state.admin_mode == "Student Platform":
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
            
            # --- THE DASHBOARD ---
            st.subheader("👤 Student Identity")
            col_id, col_dash = st.columns([3, 1])
            with col_id:
                student_id = st.text_input("Student ID", value="Physics Student V2")
            with col_dash:
                st.write("") 
                st.write("")
                if st.button("📊 Load My Dashboard", use_container_width=True):
                    with st.spinner("Fetching ongoing tests..."):
                        try:
                            res = requests.post(API_URL, json={"action": "get_all_profiles", "student_profile": {"student_id": student_id}}, timeout=30)
                            if res.status_code == 200:
                                st.session_state.dashboard_data = res.json().get('profiles', [])
                                if not st.session_state.dashboard_data:
                                    st.warning("No ongoing tests found. Start a new one below!")
                        except Exception as e:
                            st.error(f"Connection error: {e}")

            if st.session_state.dashboard_data:
                st.markdown("### 🏆 Your Ongoing Progress")
                for profile in st.session_state.dashboard_data:
                    exam_name = profile.get('target_exam', 'Unknown')
                    overall_score = profile.get('overall_readiness_score', 0.0)
                    
                    with st.container(border=True):
                        col_exam_title, col_exam_btn = st.columns([3, 1])
                        with col_exam_title:
                            st.markdown(f"#### 🎓 {exam_name}")
                            st.progress(overall_score, text=f"Overall Readiness: {overall_score*100:.0f}%")
                        with col_exam_btn:
                            st.button(f"Resume Exam", key=f"res_{exam_name}", on_click=load_scope_to_form, args=(exam_name, "Full Exam"), use_container_width=True)
                        
                        # Group by Subject
                        profs = profile.get('proficiencies', [])
                        subjects = sorted(list(set([p.get('subject') for p in profs if p.get('subject')])))
                        
                        if subjects:
                            for subj in subjects:
                                subj_profs = [p for p in profs if p.get('subject') == subj]
                                subj_score = sum([p.get('score', 0) for p in subj_profs]) / len(subj_profs) if subj_profs else 0
                                
                                with st.expander(f"📘 Subject: {subj} (Mastery: {subj_score*100:.0f}%)"):
                                    col_subj_title, col_subj_btn = st.columns([3, 1])
                                    with col_subj_title:
                                        st.progress(subj_score)
                                    with col_subj_btn:
                                        st.button(f"Study Subject", key=f"res_{exam_name}_{subj}", on_click=load_scope_to_form, args=(exam_name, "Specific Subject", subj), use_container_width=True)
                                    
                                    # Group by Topic
                                    topics = sorted(list(set([p.get('topic') for p in subj_profs if p.get('topic')])))
                                    for top in topics:
                                        top_profs = [p for p in subj_profs if p.get('topic') == top]
                                        top_score = sum([p.get('score', 0) for p in top_profs]) / len(top_profs) if top_profs else 0
                                        
                                        col_top_title, col_top_btn = st.columns([3, 1])
                                        with col_top_title:
                                            st.markdown(f"**Topic: {top}**")
                                            st.progress(top_score, text=f"Score: {top_score*100:.0f}%")
                                        with col_top_btn:
                                            st.button(f"Study Topic", key=f"res_{exam_name}_{subj}_{top}", on_click=load_scope_to_form, args=(exam_name, "Specific Topic", subj, top), use_container_width=True)

            st.divider()

            # --- THE TEST BUILDER FORM ---
            st.subheader("⚙️ Configure Next Test")
            
            # 👉 Load Syllabus Data
            syllabus_map = load_syllabus_map()
            available_exams_in_json = list(syllabus_map.keys())
            base_presets = ["UPSC CSE Prelims", "UPSC CSE Mains", "SSC CGL", "IBPS PO", "IBPS RRB PO", "RBI Grade B", "GATE CSE"]
            
            # Merge JSON exams with presets, keeping order and removing duplicates
            exam_options = list(dict.fromkeys(available_exams_in_json + base_presets)) + ["Custom Exam..."]
            
            try:
                default_index = exam_options.index(st.session_state.dash_exam)
                target_exam = st.selectbox("Target Exam", exam_options, index=default_index)
            except ValueError:
                target_exam = st.selectbox("Target Exam", exam_options, index=len(exam_options)-1)
                target_exam = st.text_input("Enter Custom Exam Name", value=st.session_state.dash_exam)

            # 👉 Fetch Available Subjects based on the selected exam
            exam_data = syllabus_map.get(target_exam, {})
            subject_options = list(exam_data.keys())

            adaptive_mode = st.toggle("Enable Adaptive Learning Mode", value=True)
            selected_override_topics = []

            if not adaptive_mode:
                st.info("Manual Mode: Select exactly what you want to study.")
                col3, col4 = st.columns(2)
                with col3:
                    # 👉 Dynamic Subject Dropdown
                    target_subject = dynamic_select("Target Subject", subject_options, st.session_state.dash_subject, "man_subj")
                    
                    # 👉 Dynamic Topic Dropdown
                    topic_options = list(exam_data.get(target_subject, {}).keys())
                    target_topic = dynamic_select("Target Topic", topic_options, st.session_state.dash_topic, "man_top")
                with col4:
                    target_difficulty = st.slider("Difficulty Level", 1, 5, 3)
                    num_questions = st.number_input("Number of Questions", min_value=1, max_value=20, value=5)
            
            else:
                st.success("Adaptive Mode: AI targets your weaknesses. Choose your scope below.")
                
                scope_options = ["Full Exam", "Specific Subject", "Specific Topic"]
                scope_index = scope_options.index(st.session_state.dash_scope)
                adaptive_scope = st.radio("Adaptive Scope:", scope_options, index=scope_index, horizontal=True)
                
                if adaptive_scope == "Specific Subject":
                    target_subject = dynamic_select("Target Subject", subject_options, st.session_state.dash_subject, "adp_subj_only")
                    target_topic = "All Syllabus"
                elif adaptive_scope == "Specific Topic":
                    col_sub, col_top = st.columns(2)
                    with col_sub:
                        target_subject = dynamic_select("Target Subject", subject_options, st.session_state.dash_subject, "adp_subj_full")
                    with col_top:
                        topic_options = list(exam_data.get(target_subject, {}).keys())
                        target_topic = dynamic_select("Target Topic", topic_options, st.session_state.dash_topic, "adp_top_full")
                else:
                    target_subject = "Entire Syllabus"
                    target_topic = "All Syllabus"
                
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
                    
                    # Filter the displayed UI progress bars based on selected scope
                    display_profs = proficiencies
                    if adaptive_scope != "Full Exam":
                        display_profs = [p for p in display_profs if p.get('subject', '').lower() == target_subject.lower()]
                    if adaptive_scope == "Specific Topic":
                        display_profs = [p for p in display_profs if p.get('topic', '').lower() == target_topic.lower()]
                    
                    if not display_profs:
                        st.info(f"No historical topics found for the requested scope ({adaptive_scope}). The AI will start exploring new topics here!")
                    else:
                        st.markdown("#### Sub-Topic Mastery & Next Test Selection")
                        st.write("Review your progress within this scope. The AI has pre-selected your weakest sub-topics.")
                        
                        sorted_profs = sorted(display_profs, key=lambda x: x.get('score', 0.0))
                        auto_select_topics = [p.get('sub_topic') for p in sorted_profs[:3]]
                        
                        for p in display_profs:
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
                        "target_topic": target_topic, 
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
                    
                    try:
                        res_json = response.json()
                    except Exception:
                        st.error(f"Backend returned non-JSON format! Raw Response: {response.text}")
                        res_json = {}
                    
                    if response.status_code == 200 and res_json:
                        st.session_state.questions = res_json.get("questions", [])
                        st.session_state.student_profile = payload["student_profile"]
                        st.session_state.user_answers = {} 
                        
                        st.session_state.session_restored = res_json.get("session_restored", False)
                        
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
            
            if st.session_state.active_history_wb:
                if st.button("⬅️ Back to Test History"):
                    st.session_state.active_history_wb = None
                    st.rerun()
                render_workbook_ui(st.session_state.active_history_wb)
                
            else:
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
                                
                            mistakes = [r for r in log.get('graded_results', []) if not r.get('is_correct')]
                            unique_mistakes = list({m['sub_topic']: m for m in mistakes}.values())
                            
                            if unique_mistakes:
                                st.markdown("**🎯 Actionable Remediation:**")
                                btn_cols = st.columns(3)
                                for i, m in enumerate(unique_mistakes):
                                    if btn_cols[i%3].button(f"📖 Study {m['sub_topic']}", key=f"hist_wb_{index}_{i}"):
                                        wb = request_workbook_generation(
                                            student_id=history_student_id,
                                            target_exam=exam_name,
                                            subject=m.get('subject', 'General'),
                                            topic=m.get('topic', 'General'),
                                            sub_topic=m['sub_topic'],
                                            difficulty=m.get('difficulty', 3)
                                        )
                                        if wb:
                                            st.session_state.active_history_wb = wb
                                            st.rerun()
                                st.divider()
                                
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
    # STUDY MODULES TAB
    # ==========================================
    with tab_learn:
        if st.session_state.phase == 'setup':
            st.title("🧠 AI Study Modules")
            st.write("Generate interactive workbooks for any topic to master weak areas.")
            
            # 👉 Load specific syllabus tree for this tab
            syllabus_map = load_syllabus_map()
            exam_data_learn = syllabus_map.get(target_exam, {}) if 'target_exam' in locals() else {}
            subject_options_learn = list(exam_data_learn.keys())
            
            wb_col1, wb_col2 = st.columns(2)
            with wb_col1:
                wb_subject = dynamic_select("Subject", subject_options_learn, "General Awareness", "wb_subj")
                topic_options_learn = list(exam_data_learn.get(wb_subject, {}).keys())
                wb_topic = dynamic_select("Topic", topic_options_learn, "Economy", "wb_top")
            with wb_col2:
                subtopic_options_learn = exam_data_learn.get(wb_subject, {}).get(wb_topic, [])
                wb_subtopic = dynamic_select("Sub-Topic (Required)", subtopic_options_learn, "Inflation", "wb_sub")
                wb_difficulty = st.slider("Target Difficulty Level", 1, 5, 3, key="wb_diff")
                
            if st.button("Generate Study Module 📖", use_container_width=True, type="primary"):
                wb = request_workbook_generation(student_id, target_exam, wb_subject, wb_topic, wb_subtopic, wb_difficulty)
                if wb:
                    st.session_state.current_workbook = wb
                    st.success("✅ Workbook loaded successfully!")
                    
            if st.session_state.current_workbook:
                render_workbook_ui(st.session_state.current_workbook)

    # ==========================================
    # PHASE 2: TEST TAKING
    # ==========================================
    if st.session_state.phase == 'testing':
        st.title(f"📝 {st.session_state.student_profile['target_exam']} Test")
        
        if st.session_state.get('session_restored'):
            st.success("ℹ️ Welcome back! We restored your unsubmitted test session.")
        else:
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
            if q.get('Requires_Diagram') and q.get('presigned_image_url'):
                st.image(q.get('presigned_image_url'), caption="Reference Diagram")
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
                        eval_response = requests.post(API_URL, json=eval_payload, timeout=1500)
                        
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
        
        mistakes = [r for r in eval_data.get("graded_results", []) if not r.get('is_correct')]
        unique_mistakes = list({m['sub_topic']: m for m in mistakes}.values())
        
        if unique_mistakes:
            st.divider()
            st.subheader("🎯 Recommended Remediation Modules")
            st.write("Click a topic you failed to instantly generate a targeted study workbook.")
            btn_cols = st.columns(3)
            for i, m in enumerate(unique_mistakes):
                if btn_cols[i%3].button(f"📖 Study {m['sub_topic']}", key=f"res_wb_{i}"):
                    wb = request_workbook_generation(
                        student_id=st.session_state.student_profile['student_id'],
                        target_exam=st.session_state.student_profile['target_exam'],
                        subject=m.get('subject', 'General'),
                        topic=m.get('topic', 'General'),
                        sub_topic=m['sub_topic'],
                        difficulty=m.get('difficulty', 3)
                    )
                    if wb:
                        st.session_state.active_results_wb = wb
                        st.rerun()

        # Show the active workbook if the user clicked one
        if st.session_state.active_results_wb:
            render_workbook_ui(st.session_state.active_results_wb)
            if st.button("❌ Close Module", key="close_res_wb"):
                st.session_state.active_results_wb = None
                st.rerun()
                
        st.divider()
        st.subheader("Detailed Breakdown")
        for result in eval_data.get("graded_results", []):
            q_id = result["question_id"]
            original_q = next((q for q in st.session_state.questions if q["id"] == q_id), None)
            
            display_text = original_q.get('text', 'Unknown') if original_q else "Unknown"
            
            with st.expander(f"Question: {format_latex(display_text[:60])}..."):
                if original_q and original_q.get('shared_context'):
                    st.caption("*(This question was part of a Shared Context Passage)*")
                    
                st.markdown(f"**Question:** {format_latex(display_text)}")

                if original_q and original_q.get('Requires_Diagram') and original_q.get('presigned_image_url'):
                    st.image(original_q.get('presigned_image_url'), caption="Reference Diagram")

                    
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
            st.session_state.active_history_wb = None
            st.session_state.active_results_wb = None
            st.session_state.session_restored = False
            st.rerun()

elif st.session_state.admin_mode == "Admin Portal (HITL)":
    # ==========================================
    # ADMIN PORTAL: HITL REVIEW PIPELINE
    # ==========================================
    st.title("🛠️ HITL Question Review Portal")
    st.write("Upload a generated JSON batch and any associated images, review them, and approve questions for the database.")
    
    # 1. File Uploaders
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        uploaded_file = st.file_uploader("Upload Generated Questions JSON", type=["json"])
    with col_up2:
        uploaded_images = st.file_uploader("Upload Images (Optional SVGs/PNGs)", type=["svg", "png", "jpg", "jpeg"], accept_multiple_files=True)
        
    # Process uploaded images to temporary directory
    if uploaded_images:
        for img_file in uploaded_images:
            img_path = os.path.join(TEMP_IMG_DIR, img_file.name)
            with open(img_path, "wb") as f:
                f.write(img_file.getbuffer())
        st.toast(f"✅ Securely stored {len(uploaded_images)} images for review.")

    # Process uploaded JSON
    if uploaded_file is not None and not st.session_state.review_queue:
        try:
            data = json.load(uploaded_file)
            st.session_state.review_queue = data
            st.session_state.review_decisions = {}
            st.session_state.current_review_idx = 0
            st.success(f"Loaded {len(data)} questions for review!")
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            
    # 2. The Review Loop
    if st.session_state.review_queue:
        total_q = len(st.session_state.review_queue)
        current_idx = st.session_state.current_review_idx
        
        # --- Bulk Actions ---
        with st.expander("Bulk Actions", expanded=False):
            col_all_acc, col_all_rej = st.columns(2)
            with col_all_acc:
                if st.button("✅ Accept All Questions", use_container_width=True):
                    for q in st.session_state.review_queue:
                        st.session_state.review_decisions[q['id']] = True
                    st.rerun()
            with col_all_rej:
                if st.button("❌ Reject All Questions", use_container_width=True):
                    for q in st.session_state.review_queue:
                        st.session_state.review_decisions[q['id']] = False
                    st.rerun()
        
        st.divider()
        
        # --- Free Navigation ---
        col_prev, col_status, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous", disabled=(current_idx == 0), use_container_width=True):
                st.session_state.current_review_idx -= 1
                st.rerun()
        with col_status:
            accepted_count = sum(1 for v in st.session_state.review_decisions.values() if v == True)
            st.markdown(f"<div style='text-align: center; margin-top: 10px;'><b>Reviewing {current_idx + 1} of {total_q} | {accepted_count} Accepted</b></div>", unsafe_allow_html=True)
        with col_next:
            if st.button("Next ➡️", disabled=(current_idx == total_q - 1), use_container_width=True):
                st.session_state.current_review_idx += 1
                st.rerun()
                
        st.progress((current_idx + 1) / total_q)
        
        # Retrieve the current question
        q = st.session_state.review_queue[current_idx]
        q_id = q.get('id')
        meta = q.get('metadata', {})
        current_status = st.session_state.review_decisions.get(q_id, None)
        
        with st.container(border=True):
            # Dynamic Status Badge
            if current_status is True:
                st.success("Current Status: ✅ **ACCEPTED**")
            elif current_status is False:
                st.error("Current Status: ❌ **REJECTED**")
            else:
                st.warning("Current Status: ⏳ **PENDING REVIEW**")
                
            st.caption(f"**Exam:** {meta.get('exam')} | **Subject:** {meta.get('subject')} | **Topic:** {meta.get('topic')} > {meta.get('sub_topic')} | **Difficulty:** {meta.get('difficulty_level')}")
            st.divider()
            
            # Render LaTeX Question Text
            st.markdown(format_latex(q.get('text', '')), unsafe_allow_html=True)
            
            # SAFELY RENDER UPLOADED IMAGE (Extracting filename from the JSON path)
            if q.get('Requires_Diagram') and q.get('local_image_path'):
                original_path = q.get('local_image_path')
                img_filename = os.path.basename(original_path) # Extracts just "MICROS_....svg"
                img_path = os.path.join(TEMP_IMG_DIR, img_filename)
                
                if os.path.exists(img_path):
                    if img_path.endswith('.svg'):
                        with open(img_path, 'r') as f:
                            svg_content = f.read()
                        components.html(svg_content, height=400, scrolling=True)
                    else:
                        st.image(img_path, caption="Generated Diagram")
                else:
                    st.warning(f"⚠️ Image file '{img_filename}' not found in your uploads. Please upload it above.")
            
            # Render Options
            options = q.get('options', {})
            for key, val in options.items():
                is_correct = "✅" if key == q.get('correct_answer') else "⚪"
                st.markdown(f"{is_correct} **{key}:** {format_latex(val)}", unsafe_allow_html=True)
                
            st.divider()
            with st.expander("Show Explanation"):
                st.markdown(format_latex(q.get('explanation', '')), unsafe_allow_html=True)
        
        # --- Individual Action Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state.review_decisions[q_id] = False
                if current_idx < total_q - 1:
                    st.session_state.current_review_idx += 1
                st.rerun()
        with col2:
            if st.button("✅ Accept", use_container_width=True):
                st.session_state.review_decisions[q_id] = True
                if current_idx < total_q - 1:
                    st.session_state.current_review_idx += 1
                st.rerun()
                
        st.divider()
        
        # 4. Final Submission
        accepted_count = sum(1 for v in st.session_state.review_decisions.values() if v == True)
        if st.button(f"📤 Submit {accepted_count} Accepted Questions to AWS", use_container_width=True, type="primary"):
            
            if accepted_count == 0:
                st.warning("No questions accepted yet!")
            else:
                import base64
                
                # Filter only accepted questions
                payload_batch = [q for q in st.session_state.review_queue if st.session_state.review_decisions.get(q['id']) == True]
                
                # Progress bar for visual feedback during large uploads
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner("Encoding images and preparing batches..."):
                    for q in payload_batch:
                        # If image exists, encode it to Base64
                        if q.get('Requires_Diagram') and q.get('local_image_path'):
                            img_filename = os.path.basename(q['local_image_path'])
                            img_path = os.path.join(TEMP_IMG_DIR, img_filename)
                            
                            if os.path.exists(img_path):
                                with open(img_path, "rb") as f:
                                    encoded = base64.b64encode(f.read()).decode("utf-8")
                                    q['image_base64'] = encoded
                                    q['image_filename'] = img_filename

                # --- THE FIX: BATCH THE UPLOADS TO BYPASS AWS 6MB LIMIT ---
                CHUNK_SIZE = 10  # Safe size to keep payload under 6MB
                total_chunks = (len(payload_batch) + CHUNK_SIZE - 1) // CHUNK_SIZE
                
                upload_failed = False
                questions_uploaded = 0
                
                for i in range(0, len(payload_batch), CHUNK_SIZE):
                    chunk = payload_batch[i:i + CHUNK_SIZE]
                    current_chunk = (i // CHUNK_SIZE) + 1
                    
                    status_text.text(f"Uploading batch {current_chunk} of {total_chunks} to AWS...")
                    
                    try:
                        res = requests.post(API_URL, json={"action": "ingest_questions", "questions": chunk}, timeout=120)
                        if res.status_code == 200:
                            questions_uploaded += len(chunk)
                            progress_bar.progress(questions_uploaded / len(payload_batch))
                        else:
                            st.error(f"Batch {current_chunk} Failed: {res.text}")
                            upload_failed = True
                            break
                    except Exception as e:
                        st.error(f"Connection Error on batch {current_chunk}: {e}")
                        upload_failed = True
                        break
                
                if not upload_failed:
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Successfully uploaded all {questions_uploaded} questions to AWS!")
                    st.balloons()