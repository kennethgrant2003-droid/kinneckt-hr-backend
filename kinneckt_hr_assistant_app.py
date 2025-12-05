"""
Kinneckt HR Assistant - Full Production Backend
Groq + Retrieval + Mediation Mode + Memory + Roles + Admin + Saving Conversations
"""

import os
import uuid
import pickle
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
)

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
#                   CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "kb_index.pkl"

GROQ_MODEL = "GROQ_MODEL = "openai/gpt-oss-20b"
"   # Change if needed


# ============================================================
#                   FLASK APP + STATIC
# ============================================================

app = Flask(__name__, static_folder="static")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

kb_index = None
sessions = {}  # session_id → dict


# ============================================================
#                KNOWLEDGE BASE RETRIEVAL
# ============================================================

def load_kb():
    global kb_index
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            kb_index = pickle.load(f)
        print(f"[KB] Loaded KB from {INDEX_PATH}")
    else:
        print(f"[KB] WARNING: KB not found at {INDEX_PATH}")
        kb_index = None


def search_kb(query: str, top_k: int = 3):
    if not kb_index or not query.strip():
        return []

    vectorizer = kb_index["vectorizer"]
    matrix = kb_index["matrix"]
    chunks = kb_index["chunks"]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]

    top_idx = sims.argsort()[::-1][:top_k]

    results = []
    for i in top_idx:
        c = chunks[int(i)]
        results.append({
            "source": c["source"],
            "page": c["page"],
            "text": c["text"],
            "score": float(sims[int(i)])
        })
    return results


# ============================================================
#                 SESSION MANAGEMENT
# ============================================================

def get_or_create_session(session_id, company_id, role):
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "company_id": company_id,
            "role": role,
            "history": [],
            "created_at": datetime.utcnow(),
        }
    else:
        if company_id:
            sessions[session_id]["company_id"] = company_id
        if role:
            sessions[session_id]["role"] = role

    return session_id, sessions[session_id]


# ============================================================
#                    ROLE BEHAVIOR
# ============================================================

def build_role_instructions(role: str) -> str:
    role = (role or "").lower()

    if role == "hr":
        return (
            "The user is in HR / People Ops. Provide structure, documentation, scripts, "
            "and compliance-aware guidance."
        )
    if role == "manager":
        return (
            "The user is a manager. Help them structure conversations, coach employees, "
            "and prevent escalation."
        )
    if role == "employee":
        return (
            "The user is an employee. Help them explain their situation neutrally, "
            "identify impact, and communicate professionally."
        )

    return "Ask the user whether they are HR, a manager, or an employee."


def build_mode_instructions(mode: str) -> str:
    if (mode or "").lower() == "mediation":
        return """
For MEDIATION MODE:
You MUST output the response EXACTLY in this structure:

1. Risk & Role Notes
2. Neutral Situation Summary
3. Key Issues Identified
4. Mediation Meeting Agenda
5. Suggested Scripts for the Mediator
6. Suggested Agreements
7. Follow-Up Plan
8. HR Documentation Summary

Each section must contain helpful, detailed, copy-paste-ready guidance.
Do NOT skip or reorder sections.
"""
    return "Respond normally in chat mode."


# ============================================================
#                 AI GENERATION FUNCTION
# ============================================================

def generate_chat_reply(user_message, role, company_id, kb_snippets, history, mode):

    role_instructions = build_role_instructions(role)
    mode_instructions = build_mode_instructions(mode)

    # Build KB context text
    context = "\n\n".join(
        [
            f"From {s['source']} (page {s['page']}): {s['text']}"
            for s in kb_snippets
        ]
    ) or "No HR reference snippets found."

    # Build conversation memory
    past_turns = []
    for h in history[-6:]:
        past_turns.append({"role": h["role"], "content": h["content"]})

    system_prompt = f"""
You are Kinneckt, an HR Assistant Chatbot.

Tone:
- Warm, empathetic, neutral.
- No legal advice.
- No blame.
- Focus on behavior, communication, impact, and next steps.

Role instructions:
{role_instructions}

Mode instructions:
{mode_instructions}

Safety:
If the user mentions harassment, discrimination, violence, threats, or self-harm:
- Recommend escalation.
- Provide documentation guidance.
- State “This is not legal advice.”

"""

    user_content = f"""
User message:
\"\"\"{user_message}\"\"\"

Company: {company_id}

Relevant HR snippets:
{context}

Follow all structure rules if mediation mode is active.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(past_turns)
    messages.append({"role": "user", "content": user_content})

    result = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
    )

    return result.choices[0].message.content


# ============================================================
#                 FRONTEND HTML (Kinneckt UI)
# ============================================================

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Kinneckt HR Assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { background: #0D1117; color: white; font-family: Arial; padding: 20px; }
.chat-box { background: #111827; padding: 15px; border-radius: 10px; height: 70vh; overflow-y: auto; }
.msg { padding: 10px; border-radius: 10px; margin: 8px 0; max-width: 70%; }
.user { background: #10B981; color: black; margin-left: auto; }
.bot { background: #374151; }
</style>
</head>

<body>
<h2>Kinneckt HR Assistant</h2>
<img src="/static/kinneckt_logo.png" width="90"/><br><br>

<div class="chat-box" id="chat"></div>

<textarea id="msg" style="width:100%;height:60px;"></textarea><br>
<button onclick="send('chat')">Send</button>
<button onclick="send('mediation')">Mediation Plan</button>

<script>
const chat = document.getElementById("chat");

function add(sender, text){
    let div = document.createElement("div");
    div.className = "msg " + sender;
    div.innerText = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

add("bot","Hi, I'm the Kinneckt HR Assistant. How can I support you?");

async function send(mode){
    let text = document.getElementById("msg").value;
    document.getElementById("msg").value = "";
    add("user", text);

    let res = await fetch("/api/chat", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify({ message:text, mode:mode, session_id:"web" })
    });

    let data = await res.json();
    add("bot", data.reply);
}
</script>

</body>
</html>
"""


# ============================================================
#                    ADMIN DASHBOARD
# ============================================================

@app.route("/admin")
def admin():
    rows = ""
    for sid, s in sessions.items():
        rows += f"<tr><td>{sid}</td><td>{s['company_id']}</td><td>{s['role']}</td><td>{len(s['history'])}</td></tr>"

    return f"""
    <h2>Kinneckt Admin Dashboard</h2>
    <table border=1 cellpadding=5>
        <tr><th>Session</th><th>Company</th><th>Role</th><th>Messages</th></tr>
        {rows}
    </table>
    """


# ============================================================
#                    CHAT ENDPOINT
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)

    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or None
    role = (data.get("role") or "").strip().lower()
    company_id = (data.get("company_id") or "default").strip()
    mode = (data.get("mode") or "chat").strip().lower()

    if not user_message:
        return jsonify({"reply": "Please enter a message."})

    session_id, session = get_or_create_session(session_id, company_id, role)

    kb_snippets = search_kb(user_message)

    reply_text = generate_chat_reply(
        user_message=user_message,
        role=role,
        company_id=company_id,
        kb_snippets=kb_snippets,
        history=session["history"],
        mode=mode,
    )

    session["history"].append({"role": "user", "content": user_message})
    session["history"].append({"role": "assistant", "content": reply_text})

    return jsonify({"reply": reply_text, "session_id": session_id})


# ============================================================
#                    HOME ROUTE
# ============================================================

@app.route("/")
def home():
    return render_template_string(PAGE_TEMPLATE)


# ============================================================
#                    MAIN LAUNCH
# ============================================================

if __name__ == "__main__":
    load_kb()
    print("[INFO] Starting Kinneckt HR Assistant backend...")
    app.run(host="0.0.0.0", port=5000, debug=True)
