import json
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import (
    SentenceTransformer,
)  # only if you want to reuse your model anywhere
import sympy as sp


# ---------- Tool functions ----------
def search_arxiv(query: str) -> str:
    """
    Try to use the 'arxiv' library if present; else return a dummy snippet.
    """
    try:
        import arxiv  # pip install arxiv

        results = arxiv.Search(
            query=query, max_results=1, sort_by=arxiv.SortCriterion.Relevance
        ).results()
        paper = next(results, None)
        if not paper:
            return f"No arXiv results for: {query}"
        title = paper.title
        authors = ", ".join(a.name for a in paper.authors)
        return f"[arXiv] {title} — {authors} ({paper.published.date()})\n{paper.summary[:400]}..."
    except Exception:
        return f"[arXiv snippet related to '{query}']"


def calculate(expression: str) -> str:
    """
    Safe(ish) math via sympy.
    """
    try:
        expr = sp.sympify(expression)
        val = sp.N(expr)
        return str(val)
    except Exception as e:
        return f"Error: {e}"


# A simple registry so routing is trivial when you add more tools
TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
    "calculate": calculate,
}

# ---------- LLM prompting helpers ----------
SYSTEM_INSTRUCTIONS = """You are a helpful assistant with tool-calling.
If the user's request is best answered by a tool, return ONLY a JSON object:
{"function":"<one of: search_arxiv, calculate>","arguments":{...}}
- For math: {"function":"calculate","arguments":{"expression":"2+2"}}
- For arXiv: {"function":"search_arxiv","arguments":{"query":"quantum entanglement"}}
If no tool is needed, reply with plain text. Do NOT wrap JSON in backticks.
"""


def build_messages(user_text: str):
    # Adapt to your Llama 3 chat client; most accept a list of messages
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_text},
    ]


def llama3_chat_model(user_text: str) -> str:
    """
    Plug in your Week-3 Llama 3 call here and return a string.
    The string can be either JSON (function call) or normal text.
    """
    # raise NotImplementedError("Connect this to your Llama 3 client and return the raw text")
    # --- Optional local heuristic so you can test without an LLM:
    t = user_text.lower()
    if any(
        k in t
        for k in [
            "integrate",
            "derive",
            "simplify",
            "+",
            "-",
            "*",
            "/",
            "^",
            "sqrt",
            "sin",
            "cos",
            "=",
        ]
    ):
        return json.dumps(
            {"function": "calculate", "arguments": {"expression": user_text}}
        )
    if any(k in t for k in ["arxiv", "paper", "preprint", "find a paper", "search"]):
        q = user_text.replace("arxiv", "").replace("search", "").strip()
        return json.dumps(
            {"function": "search_arxiv", "arguments": {"query": q or user_text}}
        )
    return "Sure — what exactly would you like to know?"


# ---------- Routing ----------
def route_llm_output(llm_output: str) -> Dict[str, Any]:
    """
    If llm_output is JSON with {"function":..., "arguments": {...}}, run the tool and return:
    {"type":"tool_result","name":..., "arguments":..., "result":...}
    Else return {"type":"text","text": llm_output}
    """
    try:
        payload = json.loads(llm_output)
        func = payload.get("function")
        args = payload.get("arguments", {}) or {}
        if func in TOOL_REGISTRY:
            result = TOOL_REGISTRY[func](**args)
            return {
                "type": "tool_result",
                "name": func,
                "arguments": args,
                "result": result,
            }
        else:
            return {"type": "text", "text": f"Error: Unknown function '{func}'"}
    except (json.JSONDecodeError, TypeError):
        return {"type": "text", "text": llm_output}


# ---------- FastAPI ----------
class VoiceQuery(BaseModel):
    text: str


app = FastAPI(title="Week 6 Voice Agent (Function Calling)")


@app.post("/api/voice-query")
def voice_query_endpoint(req: VoiceQuery):
    user_text = (req.text or "").strip()
    if not user_text:
        raise HTTPException(400, "text is required")
    llm_raw = llama3_chat_model(user_text)
    routed = route_llm_output(llm_raw)
    # Here you would TTS(routed_text) and stream audio back; we just return JSON.
    if routed["type"] == "tool_result":
        return {
            "user_text": user_text,
            "llm_raw": llm_raw,
            "tool_called": routed["name"],
            "tool_args": routed["arguments"],
            "tool_output": routed["result"],
            "final_response": routed["result"],
        }
    else:
        return {
            "user_text": user_text,
            "llm_raw": llm_raw,
            "final_response": routed["text"],
        }


# ---------- Minimal test harness for deliverables ----------
if __name__ == "__main__":
    tests = [
        "What is 2^10 + 3*7?",
        "Search arXiv for retrieval-augmented generation in healthcare",
        "Tell me a fun fact about black holes",
    ]
    for t in tests:
        llm_raw = llama3_chat_model(t)
        routed = route_llm_output(llm_raw)
        print("\n---")
        print("User:", t)
        print("LLM raw:", llm_raw)
        if routed["type"] == "tool_result":
            print("Tool:", routed["name"])
            print("Args:", routed["arguments"])
            print("Tool output:", routed["result"][:300])
            print("Final:", routed["result"][:300])
        else:
            print("Final:", routed["text"])
    print("\nRun API with: uvicorn week6_agent:app --reload")
