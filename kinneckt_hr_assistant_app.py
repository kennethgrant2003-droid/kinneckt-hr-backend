"""
Kinneckt HR Assistant Backend (Flask)
- Groq LLM + Retrieval (KB) + Session Memory
- Mobile endpoint: POST /api/chat
- Web UI: GET /
- Admin dashboard: GET /admin
- Health check: GET /health
- Creator attribution: answers "who created this app" etc.
"""

import os
import uuid
import pickle
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template_string

# --- Optional CORS (recommended for mobile) ---
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except Exception:
    CORS_AVAILABLE = False

# --- Groq + retrieval ---
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "kb_index.pkl"

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

app = Flask(__name__)

if CORS_AVAILABLE:
    # Allow calls from anywhere (fine for MVP).
    # For production, restrict to your domains.
    CORS(app)

# Groq client (only if key exists)
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if client:
    print("[LLM] Groq client initialized.")
else:
    print("[LLM] WARNING: GROQ_API_KEY missing. /api/chat will return an error.")

kb_index = None

# In-memory sessions
# sessions[session_id] = {
#   "company_id": str,
#   "role": str,
#   "history": [{"role": "user"/"assistant", "content": str}],
#   "created_at": datetime
# }
sessions: dict[str, dict] = {}


# =========================
# KB LOAD + SEARCH
# =========================
def load_kb():
    global kb_index
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            kb_index = pickle.load(f)
        print(f"[KB] Loaded knowledge base from {INDEX_PATH}")
    else:
        kb_index = None
        print(f"[KB] WARNING: {INDEX_PATH} not found. Run: python build_hr_kb.py")


def search_kb(query: str, top_k: int = 3):
    if not kb_index or not query.strip():
        return []

    vectorizer = kb_index["vectorizer"]
    matrix = kb_index["matrix"]
    chunks = kb_index["chunks"]

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_indices = sims.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append(
            {
                "source": c.get("source", "unknown"),
                "page": c.get("page", "?"),
                "text": c.get("text", ""),
                "score": float(sims[int(idx)]),
            }
        )
    return results


# =========================
# SESSION
# =========================
def get_or_create_session(session_id: str | None, company_id: str | None, role: str | None):
    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = {
            "company_id": company_id or "default",
            "role": role or "unknown",
            "history": [],
            "created_at": datetime.utcnow(),
        }
    else:
        if company_id:
            sessions[session_id]["company_id"] = company_id
        if role:
            sessions[session_id]["role"] = role

    return session_id, sessions[session_id]


# =========================
# CREATOR ATTRIBUTION
# =========================
def is_creator_question(text: str) -> bool:
    t = (text or "").strip().lower()
    triggers = [
        "who created", "who built", "who made", "who developed",
        "who is the developer", "who is the creator", "who designed",
        "who owns this app", "who owns kinneckt", "who made kinneckt",
        "who built kinneckt", "created by", "built by", "developer name",
        "who is kenneth", "who is kenneth grant", "who is granted solutions",
    ]
    return any(p in t for p in triggers)

def creator_reply() -> str:
    return (
        "Kinneckt was created by Kenneth Grant, Founder of Granted Solutions.\n\n"
        "Kinneckt provides confidential, AI-powered HR guidance for employees, managers, and HR teams—"
        "focused on clarity, communication, and practical next steps."
    )


# =========================
# ROLE + MODE INSTRUCTIONS
# =========================
def build_role_instructions(role: str) -> str:
    role = (role or "").lower()
    if role in ("hr", "people ops", "people_ops", "people"):
        return (
            "The user is in HR / People Ops / Leadership. "
            "Prioritize structure, documentation, scripts, action plans, and follow-up steps."
        )
    if role in ("manager", "leader", "supervisor"):
        return (
            "The user is a manager or team lead. "
            "Help them structure conversations, ask good questions, and prevent escalation."
        )
    if role in ("employee", "individual contributor", "staff"):
        return (
            "The user is an employee. Help them describe what’s happening neutrally, "
            "explain impact, and ask for support."
        )
    return (
        "The user’s role is not clear. Ask if they are HR, manager, or employee and adapt accordingly."
    )


def build_mode_instructions(mode: str) -> str:
    mode = (mode or "chat").lower()
    if mode == "mediation":
        return (
            "The user requested a structured mediation plan. "
            "Use this exact structure:\n"
            "1. Risk & Role Notes\n"
            "2. Neutral Situation Summary\n"
            "3. Key Issues Identified\n"
            "4. Mediation Meeting Agenda\n"
            "5. Suggested Scripts for the Mediator\n"
            "6. Suggested Agreements\n"
            "7. Follow-Up Plan\n"
            "8. HR Documentation Summary (if appropriate)\n"
            "Make it detailed and copy-paste friendly."
        )
    return "Normal chat mode."


# =========================
# GROQ RESPONSE
# =========================
def generate_chat_reply(
    user_message: str,
    role: str,
    company_id: str,
    kb_snippets: list[dict],
    history: list[dict],
    mode: str = "chat",
) -> str:

    if not client:
        return "Server is missing GROQ_API_KEY. Add it in Render Environment Variables and redeploy."

    role_instructions = build_role_instructions(role)
    mode_instructions = build_mode_instructions(mode)

    context_parts = []
    for snip in kb_snippets[:3]:
        context_parts.append(
            f"From {snip.get('source','unknown')} (page {snip.get('page','?')}): {snip.get('text','')}"
        )
    context_text = "\n\n".join(context_parts) if context_parts else "No HR snippets retrieved."

    system_prompt = f"""
You are Kinneckt, an HR Assistant.

You do NOT give legal advice.
You do NOT decide who is right or wrong.
You focus on clarity, behavior, communication, and next steps.

Safety:
If user describes harassment, discrimination, threats, violence, self-harm, stalking,
or protected class concerns, advise escalation to HR/legal/emergency services as appropriate,
and remind: "This is not legal advice."

Role behavior:
{role_instructions}

Mode instructions:
{mode_instructions}

Style:
Warm, calm, neutral, professional, high EQ.
Use short headers and clear bullets.
"""

    user_content = f"""
Company (context only): {company_id}

User message:
\"\"\"{user_message}\"\"\"

HR reference snippets:
{context_text}
"""

    messages = [{"role": "system", "content": system_prompt}]

    for m in history[-6:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_content})

    chat_completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
    )
    return chat_completion.choices[0].message.content


# =========================
# SIMPLE WEB UI
# =========================
PAGE_TEMPLATE = """
<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Kinneckt HR Assistant</title></head>
<body style="font-family: Arial; padding: 16px;">
  <h2>Kinneckt HR Assistant</h2>
  <p>Use the mobile app for best experience. This page is a simple test UI.</p>
  <form id="f">
    <input id="msg" style="width: 70%;" placeholder="Type..." />
    <button type="submit">Send</button>
  </form>
  <pre id="out" style="white-space: pre-wrap; margin-top: 12px;"></pre>

<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = document.getElementById('msg').value;
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: msg, role:'employee', company_id:'default', mode:'chat'})
  });
  const data = await res.json();
  document.getElementById('out').textContent = data.reply;
});
</script>
</body>
</html>
"""


# =========================
# ROUTES
# =========================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}), 200


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    rows_html = []
    for sid, sess in sessions.items():
        role = sess.get("role", "unknown")
        company_id = sess.get("company_id", "default")
        created = sess.get("created_at", datetime.utcnow())
        history = sess.get("history", [])
        message_count = len(history)

        preview = ""
        for m in reversed(history):
            if m.get("role") == "user":
                preview = (m.get("content", "")[:120]).replace("<", "&lt;").replace(">", "&gt;")
                break

        rows_html.append(
            f"<tr><td>{sid}</td><td>{company_id}</td><td>{role}</td>"
            f"<td>{message_count}</td><td>{created.isoformat()}Z</td><td>{preview}</td></tr>"
        )

    html = f"""
    <html><head><title>Kinneckt Admin</title></head>
    <body style="font-family: Arial; padding: 16px; background:#f3f4f6;">
      <h2>Kinneckt Admin Dashboard</h2>
      <table border="1" cellpadding="6" style="background:#fff; border-collapse:collapse; width:100%;">
        <tr><th>Session</th><th>Company</th><th>Role</th><th>Messages</th><th>Created</th><th>Last user msg</th></tr>
        {''.join(rows_html)}
      </table>
    </body></html>
    """
    return html


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        print("[API] JSON parse error:", e)
        return jsonify({"reply": "I had trouble reading your request JSON."}), 400

    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id")
    role = (data.get("role") or "unknown").strip()
    company_id = (data.get("company_id") or "default").strip()
    mode = (data.get("mode") or "chat").strip().lower()

    # ✅ Creator question handled here (no app resubmission needed)
    if is_creator_question(user_message):
        return jsonify({"reply": creator_reply()}), 200

    # Create or get session
    session_id, sess = get_or_create_session(session_id, company_id, role)

    # Store user message in memory
    sess["history"].append({"role": "user", "content": user_message})

    # Retrieval
    kb_snips = search_kb(user_message, top_k=3)

    # LLM reply (mediation works if mode == "mediation")
    reply_text = generate_chat_reply(
        user_message=user_message,
        role=sess["role"],
        company_id=sess["company_id"],
        kb_snippets=kb_snips,
        history=sess["history"],
        mode=mode,
    )

    # Store assistant message
    sess["history"].append({"role": "assistant", "content": reply_text})

    return jsonify({"reply": reply_text, "session_id": session_id}), 200


if __name__ == "__main__":
    load_kb()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

