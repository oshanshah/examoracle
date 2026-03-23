import os
import json
import re
import time
from pathlib import Path

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import pdfplumber

# =========================
# CONFIG
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0.1,
)

# =========================
# SAFE LLM CALL
# =========================
def safe_llm_call(prompt, retries=3):
    for i in range(retries):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            if "rate_limit" in str(e):
                wait = 5 + i * 2
                print(f"⏳ Rate limit... retrying in {wait}s")
                time.sleep(wait)
            else:
                return f"Error: {e}"
    return "LLM failed"

# =========================
# LOGGER
# =========================
logs = []

def log(step, msg, detail=None):
    entry = f"[{step}] {msg}"
    if detail:
        entry += f"\n    → {detail}"
    print(entry)
    logs.append(entry)

# =========================
# TOOL 1: CALCULATOR
# Normalizes weightage to sum to exactly 100%
# =========================
def calculator_tool(weightage):
    log("TOOL_CALL", "Calculator tool invoked", "Normalizing weightage to sum to 100%")
    total = sum(weightage.values())
    if total == 0:
        return weightage
    normalized = {k: round(v / total * 100) for k, v in weightage.items()}
    diff = 100 - sum(normalized.values())
    if diff != 0:
        top_key = max(normalized, key=normalized.get)
        normalized[top_key] += diff
    log("TOOL_RESULT", "Calculator tool complete", f"Total = {sum(normalized.values())}%")
    return normalized

# =========================
# TOOL 2: TOPIC COVERAGE CHECKER
# Checks how many weightage topics appear in the final paper
# =========================
def topic_coverage_tool(weightage, paper_text):
    log("TOOL_CALL", "Topic coverage checker invoked", "Scanning paper for topic mentions")
    paper_lower = paper_text.lower()
    covered = []
    missed = []
    for topic in weightage.keys():
        if topic.lower() in paper_lower:
            covered.append(topic)
        else:
            missed.append(topic)
    total = len(weightage)
    pct = round(len(covered) / total * 100) if total > 0 else 0
    result = {
        "covered": covered,
        "missed": missed,
        "coverage_pct": pct
    }
    log("TOOL_RESULT", "Topic coverage check complete",
        f"{len(covered)}/{total} topics covered ({pct}%)")
    if missed:
        log("COVERAGE_WARN", "Some topics missing from paper", ", ".join(missed))
    return result

# =========================
# PDF READER
# =========================
def read_pdfs(subject, folder):
    log("PDF", "Scanning folder for relevant PDFs",
        f"Filtering files containing subject keyword: '{subject}'")

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".pdf") and subject.lower() in f.lower()
    ]

    if not files:
        log("PDF_WARNING",
            "No subject-specific PDFs found",
            "System will rely on fallback knowledge")
        return {}

    data = {}

    for f in files:
        try:
            with pdfplumber.open(Path(folder)/f) as pdf:
                text = " ".join(p.extract_text() or "" for p in pdf.pages)
                data[f] = text
                log("PDF_LOAD",
                    f"Loaded file: {f}",
                    f"Extracted {len(text)} characters")
        except Exception as e:
            data[f] = f"ERROR: {e}"

    return data

# =========================
# NODES
# =========================

def pdf_node(state):
    data = read_pdfs(state["subject"], state["folder"])
    return {**state, "pdf_data": data}


def analysis_node(state):
    log("ANALYSIS_START",
        "Beginning topic extraction",
        "Combining PDF content and sending to LLM")

    if not state["pdf_data"]:
        return {**state, "has_data": False}

    combined = " ".join(state["pdf_data"].values())[:3000]

    log("ANALYSIS_DATA",
        "Prepared input for AI model",
        f"Trimmed content length: {len(combined)} characters")

    prompt = f"""
    Extract important exam topics with weightage.

    Return ONLY JSON:
    {{"topic": percentage}}

    Text:
    {combined}
    """

    raw = safe_llm_call(prompt)

    try:
        weightage = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
        weightage = calculator_tool(weightage)
        log("ANALYSIS_RESULT",
            "Weightage extracted and normalized",
            str(weightage))
    except:
        weightage = {}
        log("ANALYSIS_FAIL",
            "No meaningful weightage extracted",
            "Triggering fallback mechanism")

    return {
        **state,
        "weightage": weightage,
        "has_data": bool(weightage)
    }


def fallback_node(state):
    log("FALLBACK",
        "Switching to fallback mode",
        "Using general AI knowledge instead of PDFs")

    prompt = f"""
    Give important topics for {state['subject']} exam.

    Return JSON:
    {{"topic": percentage}}
    """

    raw = safe_llm_call(prompt)

    try:
        weightage = json.loads(re.search(r"\{.*\}", raw).group())
        weightage = calculator_tool(weightage)
    except:
        weightage = {}

    return {**state, "weightage": weightage}


def paper_node(state):
    log("GENERATOR",
        "Generating predicted exam paper",
        "Using extracted topic weightage to structure sections")

    prompt = f"""
    Subject: {state['subject']}

    Topics:
    {state['weightage']}

    Generate a 70-mark university exam paper with the following sections.
    ALL questions must be descriptive/written answer questions. Do NOT include MCQs, fill in the blanks, or true/false questions anywhere.

    Section A (10 marks): 5 short answer questions x 2 marks each.
    Each question should ask to define, state, or briefly explain a concept in 2-3 sentences.

    Section B (20 marks): 4 medium answer questions x 5 marks each.
    Each question should ask to explain or describe a concept in a paragraph with examples.

    Section C (40 marks): 4 long answer questions x 10 marks each.
    Each question should ask to analyze, compare, discuss, or evaluate a topic in detail.

    Format each question as:
    Q1. [question text] (2 marks)
    """

    paper = safe_llm_call(prompt)

    log("GENERATOR_DONE",
        "Exam paper generated successfully",
        "Running topic coverage check...")

    # === TOOL 2: TOPIC COVERAGE CHECKER ===
    coverage = topic_coverage_tool(state["weightage"], paper)

    return {**state, "paper": paper, "coverage": coverage}


# =========================
# GRAPH
# =========================

def build_graph():
    g = StateGraph(dict)

    g.add_node("pdf", pdf_node)
    g.add_node("analysis", analysis_node)
    g.add_node("fallback", fallback_node)
    g.add_node("paper", paper_node)

    g.set_entry_point("pdf")

    g.add_edge("pdf", "analysis")

    def route(state):
        return "fallback" if not state.get("has_data") else "paper"

    g.add_conditional_edges("analysis", route, {
        "fallback": "fallback",
        "paper": "paper"
    })

    g.add_edge("fallback", "paper")
    g.add_edge("paper", END)

    return g.compile()

# =========================
# MAIN FUNCTION
# =========================

def run_exam_oracle(subject, folder):
    global logs
    logs = []

    result = build_graph().invoke({
        "subject": subject,
        "folder": folder,
        "weightage": {}
    })

    return {
        "weightage": result.get("weightage"),
        "paper": result.get("paper"),
        "coverage": result.get("coverage"),
        "logs": logs
    }