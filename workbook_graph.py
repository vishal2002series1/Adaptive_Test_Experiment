import os
import json
import uuid
from typing import TypedDict, List, Dict, Optional, Any
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END

# Import schemas and DB functions
from schema import LearningWorkbook, VideoReference, StudentProfile
from db import save_cached_workbook
from vector_store import retrieve_best_question

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# ==========================================
# 1. GRAPH STATE DEFINITION
# ==========================================

class WorkbookState(TypedDict):
    target_exam: str
    subject: str
    topic: str
    sub_topic: str
    difficulty_level: int
    
    # Generated Components
    video_references: List[Dict[str, str]]
    theory_markdown: str
    tricks_and_mnemonics: str
    mermaid_graph_code: str
    
    # 👉 UPDATED: Store both IDs and Full Questions
    practice_question_ids: List[str]
    practice_questions: List[Dict[str, Any]]
    
    # Final Output
    final_workbook: Optional[LearningWorkbook]

def clean_json_response(text: str) -> str:
    """UI-Safe JSON cleaner avoiding literal backticks"""
    text = text.strip()
    marker = chr(96) * 3 # Dynamically creates the triple backtick
    
    if text.startswith(marker + "json\n"):
        text = text[8:]
    elif text.startswith(marker + "json"):
        text = text[7:]
    elif text.startswith(marker):
        text = text[3:]
        
    if text.endswith(marker):
        text = text[:-3]
        
    return text.strip()

# ==========================================
# 2. AGENT NODES
# ==========================================

def researcher_node(state: WorkbookState) -> dict:
    print(f"🔍 Researcher: Sourcing video references for {state['sub_topic']}...")
    
    prompt = f"""
    You are an expert educational researcher for the {state['target_exam']} exam.
    Find or highly recommend 2 specific, realistic educational video titles and search-friendly URLs (like YouTube) that teach the concept of:
    Taxonomy: {state['subject']} > {state['topic']} > {state['sub_topic']}
    Difficulty Level: {state['difficulty_level']} out of 5.
    
    Output strictly as a JSON array matching this schema:
    [
        {{
            "title": "Video Title",
            "url": "https://www.youtube.com/results?search_query=...",
            "why_watch_this": "Brief explanation of why this is perfect for difficulty level {state['difficulty_level']}"
        }}
    ]
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json")
        )
        videos = json.loads(clean_json_response(response.text))
        return {"video_references": videos}
    except Exception as e:
        print(f"❌ Researcher Error: {e}")
        return {"video_references": []}

def author_node(state: WorkbookState) -> dict:
    print(f"✍️ Author: Drafting theory and mnemonics (Difficulty {state['difficulty_level']})...")
    
    prompt = f"""
    You are an elite textbook author for the {state['target_exam']} exam.
    Write an educational module for: {state['subject']} > {state['topic']} > {state['sub_topic']}.
    Target Difficulty: {state['difficulty_level']} out of 5.
    
    DIFFICULTY RULES:
    - Level 1-2: Focus on simple definitions, core concepts, and beginner-friendly analogies.
    - Level 3: Focus on application, formulas, and standard exam scenarios.
    - Level 4-5: Focus on edge cases, complex synthesis, and expert-level nuance.
    
    REQUIREMENTS:
    Output strictly as a JSON object with two keys:
    1. "theory_markdown": A beautifully formatted markdown string (use bolding, headers, and bullet points) explaining the theory. MUST use $ for inline math and $$ for block math.
    2. "tricks_and_mnemonics": A markdown string detailing shortcuts, memory tricks, or common traps to avoid. MUST use $ for inline math and $$ for block math.
    
    JSON format:
    {{
        "theory_markdown": "### The Core Concept\\n...",
        "tricks_and_mnemonics": "### Exam Shortcuts\\n..."
    }}
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, response_mime_type="application/json")
        )
        content = json.loads(clean_json_response(response.text))
        return {
            "theory_markdown": content.get("theory_markdown", "Theory content unavailable."),
            "tricks_and_mnemonics": content.get("tricks_and_mnemonics", "No tricks available.")
        }
    except Exception as e:
        print(f"❌ Author Error: {e}")
        return {"theory_markdown": "Error generating theory.", "tricks_and_mnemonics": ""}

def designer_node(state: WorkbookState) -> dict:
    print(f"🎨 Designer: Creating Interactive Markmap hierarchy...")
    
    prompt = f"""
    You are an expert educational data visualizer. 
    Your task is to create a comprehensive, highly readable Mind Map for the topic: {state['sub_topic']} (Context: {state['target_exam']}).
    
    Instead of code, you must output a pure Nested Markdown List. 
    
    RULES:
    1. Start with a single H1 (#) representing the core topic.
    2. Use H2 (##) for main branches (e.g., Causes, Types, Impacts).
    3. Use H3 (###) and bullet points (-) for deeper details.
    4. Keep node text concise (1-5 words). If you need to explain more, nest it deeper.
    5. ONLY output the markdown list. Do not include introductory text.
    
    EXAMPLE FORMAT:
    # Inflation
    ## Causes
    ### Demand-Pull
    - High Money Supply
    - Population Growth
    ### Cost-Push
    - Supply Chain Shocks
    ## Types
    ### Creeping
    - Less than 3%
    """
    try:
        response = client.models.generate_content(
            model='gemini-3.1-pro-preview',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        # We don't need clean_json_response here because it's just raw markdown!
        markdown_mindmap = response.text.strip()
        
        # Clean up stray backticks if the LLM wraps it in a markdown block
        if markdown_mindmap.startswith("```markdown"):
            markdown_mindmap = markdown_mindmap[11:]
        elif markdown_mindmap.startswith("```"):
            markdown_mindmap = markdown_mindmap[3:]
        if markdown_mindmap.endswith("```"):
            markdown_mindmap = markdown_mindmap[:-3]
            
        return {"mermaid_graph_code": markdown_mindmap.strip()}
    except Exception as e:
        print(f"❌ Designer Error: {e}")
        return {"mermaid_graph_code": "# Error\n## Failed to load mind map"}

# 👉 UPDATED: Curator now extracts and returns the full question dictionary
def curator_node(state: WorkbookState) -> dict:
    print(f"🗂️ Curator: Fetching practice questions from Vector DB...")
    
    fetched_ids = []
    fetched_questions = []
    dummy_profile = StudentProfile(student_id="workbook_curator", target_exam=state['target_exam'])
    
    for _ in range(3):
        q = retrieve_best_question(
            target_exam=state['target_exam'],
            target_subject=state['subject'],
            target_topic=state['topic'],
            target_sub_topic=state['sub_topic'],
            target_difficulty=state['difficulty_level'],
            student_profile=dummy_profile,
            exclude_ids=fetched_ids
        )
        if q:
            fetched_ids.append(q.id)
            fetched_questions.append(q.model_dump())
        else:
            break
            
    print(f"✅ Curator attached {len(fetched_ids)} practice questions.")
    return {"practice_question_ids": fetched_ids, "practice_questions": fetched_questions}

# 👉 UPDATED: Compiler adds the full questions to the LearningWorkbook model
def compiler_node(state: WorkbookState) -> dict:
    print("🏗️ Compiler: Assembling final Learning Workbook...")
    
    videos = [VideoReference(**v) for v in state.get('video_references', [])]
    
    workbook = LearningWorkbook(
        id=f"wb_{uuid.uuid4().hex[:8]}",
        target_exam=state['target_exam'],
        sub_topic=state['sub_topic'],
        difficulty_level=state['difficulty_level'],
        theory_markdown=state['theory_markdown'],
        mermaid_graph_code=state['mermaid_graph_code'],
        tricks_and_mnemonics=state['tricks_and_mnemonics'],
        video_references=videos,
        practice_question_ids=state.get('practice_question_ids', []),
        practice_questions=state.get('practice_questions', [])
    )
    
    save_cached_workbook(workbook.model_dump())
    return {"final_workbook": workbook}

# ==========================================
# 3. BUILD THE WORKBOOK LANGGRAPH
# ==========================================

workbook_workflow = StateGraph(WorkbookState)

workbook_workflow.add_node("researcher", researcher_node)
workbook_workflow.add_node("author", author_node)
workbook_workflow.add_node("designer", designer_node)
workbook_workflow.add_node("curator", curator_node)
workbook_workflow.add_node("compiler", compiler_node)

# Sequential Pipeline
workbook_workflow.set_entry_point("researcher")
workbook_workflow.add_edge("researcher", "author")
workbook_workflow.add_edge("author", "designer")
workbook_workflow.add_edge("designer", "curator")
workbook_workflow.add_edge("curator", "compiler")
workbook_workflow.add_edge("compiler", END)

workbook_app = workbook_workflow.compile()
print("✅ workbook_app compiled successfully")