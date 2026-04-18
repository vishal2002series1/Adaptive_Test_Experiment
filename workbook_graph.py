import os
import json
import uuid
from typing import TypedDict, List, Dict, Optional, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import schemas and DB functions
from schema import LearningWorkbook, VideoReference, StudentProfile
from db import save_cached_workbook
from vector_store import retrieve_best_question

# 👉 NEW: Import the centralized Bedrock client
from bedrock_client import bedrock_runtime

load_dotenv()

# Constants for Bedrock Claude
CLAUDE_MODEL_ID = os.environ.get("MODEL_ID", "us.anthropic.claude-sonnet-4-6")
ANTHROPIC_VERSION = "bedrock-2023-05-31"

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
    
    # Stored Questions
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

def invoke_claude(prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
    """Helper function to invoke Claude 3.5 Sonnet via Bedrock."""
    body = {
        "anthropic_version": ANTHROPIC_VERSION,
        "max_tokens": 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system_prompt:
        body["system"] = system_prompt

    response = bedrock_runtime.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ==========================================
# 2. AGENT NODES
# ==========================================

def researcher_node(state: WorkbookState) -> dict:
    print(f"🔍 Researcher: Sourcing video references for {state['sub_topic']}...")
    
    system_prompt = f"You are an expert educational researcher for the {state['target_exam']} exam. Output ONLY valid JSON."
    prompt = f"""
    Find or highly recommend 2 specific, realistic educational video titles and search-friendly URLs (like YouTube) that teach the concept of:
    Taxonomy: {state['subject']} > {state['topic']} > {state['sub_topic']}
    Difficulty Level: {state['difficulty_level']} out of 5.
    
    Output strictly as a JSON array matching this schema (do NOT use markdown formatting, just the raw JSON):
    [
        {{
            "title": "Video Title",
            "url": "https://www.youtube.com/results?search_query=...",
            "why_watch_this": "Brief explanation of why this is perfect for difficulty level {state['difficulty_level']}"
        }}
    ]
    """
    try:
        response_text = invoke_claude(prompt=prompt, system_prompt=system_prompt, temperature=0.3)
        videos = json.loads(clean_json_response(response_text))
        return {"video_references": videos}
    except Exception as e:
        print(f"❌ Researcher Error: {e}")
        return {"video_references": []}

def author_node(state: WorkbookState) -> dict:
    print(f"✍️ Author: Drafting theory and mnemonics (Difficulty {state['difficulty_level']})...")
    
    system_prompt = f"You are an elite textbook author for the {state['target_exam']} exam. Output ONLY valid JSON."
    prompt = f"""
    Write an educational module for: {state['subject']} > {state['topic']} > {state['sub_topic']}.
    Target Difficulty: {state['difficulty_level']} out of 5.
    
    DIFFICULTY RULES:
    - Level 1-2: Focus on simple definitions, core concepts, and beginner-friendly analogies.
    - Level 3: Focus on application, formulas, and standard exam scenarios.
    - Level 4-5: Focus on edge cases, complex synthesis, and expert-level nuance.
    
    REQUIREMENTS:
    Output strictly as a JSON object with two keys (do NOT use markdown formatting, just the raw JSON):
    1. "theory_markdown": A beautifully formatted markdown string (use bolding, headers, and bullet points) explaining the theory. MUST use $ for inline math and $$ for block math.
    2. "tricks_and_mnemonics": A markdown string detailing shortcuts, memory tricks, or common traps to avoid. MUST use $ for inline math and $$ for block math.
    
    JSON format:
    {{
        "theory_markdown": "### The Core Concept\\n...",
        "tricks_and_mnemonics": "### Exam Shortcuts\\n..."
    }}
    """
    try:
        response_text = invoke_claude(prompt=prompt, system_prompt=system_prompt, temperature=0.4)
        content = json.loads(clean_json_response(response_text))
        return {
            "theory_markdown": content.get("theory_markdown", "Theory content unavailable."),
            "tricks_and_mnemonics": content.get("tricks_and_mnemonics", "No tricks available.")
        }
    except Exception as e:
        print(f"❌ Author Error: {e}")
        return {"theory_markdown": "Error generating theory.", "tricks_and_mnemonics": ""}

def designer_node(state: WorkbookState) -> dict:
    print(f"🎨 Designer: Creating Interactive Markmap hierarchy...")
    
    system_prompt = f"You are an expert educational data visualizer preparing high-yield study materials for the {state['target_exam']} exam. Output ONLY a pure Nested Markdown List."
    prompt = f"""
    Your task is to create a highly focused, readable Mind Map for the following exact syllabus concept:
    
    Taxonomy: {state['subject']} > {state['topic']} > {state['sub_topic']}
    
    CRITICAL INSTRUCTIONS:
    1. Strict Exam Alignment: You must anchor the mind map strictly to the context of the '{state['target_exam']}' exam. 
    2. No Domain Drift: Keep the content exclusively focused on the provided taxonomy. Do not hallucinate or drift into unrelated subjects.
    3. Exam Utility: Ensure the nodes highlight the most testable concepts, relationships, mechanisms, or distinctions that a student must memorize.
    
    RULES:
    1. Start with a single H1 (#) representing the core topic ({state['sub_topic']}).
    2. Use H2 (##) for main branches (e.g., Core Principles, Components, Frameworks, Causes).
    3. Use H3 (###) and bullet points (-) for deeper details.
    4. Keep node text concise (1-5 words). If you need to explain more, nest it deeper.
    5. ONLY output the markdown list. Do not include introductory text.
    """
    try:
        response_text = invoke_claude(prompt=prompt, system_prompt=system_prompt, temperature=0.2)
        markdown_mindmap = response_text.strip()
        
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

workbook_workflow.set_entry_point("researcher")
workbook_workflow.add_edge("researcher", "author")
workbook_workflow.add_edge("author", "designer")
workbook_workflow.add_edge("designer", "curator")
workbook_workflow.add_edge("curator", "compiler")
workbook_workflow.add_edge("compiler", END)

workbook_app = workbook_workflow.compile()
print("✅ workbook_app compiled successfully")