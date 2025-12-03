"""
Kinneckt HR Assistant - Groq + Retrieval + Memory + Roles + Admin

Features:
- Chat-style HR assistant UI (Flask + HTML/JS)
- Uses Groq (GPT-OSS) as the "brain"
- Real HR policy retrieval from PDFs (TF-IDF index)
- Conversation memory per user/session
- Multi-user support via session_id
- Role modes: HR / Manager / Employee
- Button to generate a full structured Mediation Plan
- Simple SaaS-style admin dashboard at /admin

Prereqs:
- Environment variable: GROQ_API_KEY
- Groq Python SDK: pip install groq
- Flask: pip install flask
- Retrieval: pip install pypdf scikit-learn numpy
- Index built by: python build_hr_kb.py
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

# --------- CONFIG ---------

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "kb_index.pkl"

# Use a Groq model that actually exists for your account
GROQ_MODEL = "openai/gpt-oss-20b"

# --------- GLOBALS ---------

app = Flask(__name__)

# Safe Groq client setup
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("[LLM] Groq client initialized.")
    except Exception as e:
        print("[LLM] ERROR initializing Groq client:", e)
        client = None
else:
    print("[LLM] WARNING: GROQ_API_KEY not set. Running in fallback/test mode only.")
    client = None

kb_index = None  # loaded at startup

# Simple in-memory "sessions" store
# sessions[session_id] = {
#   "company_id": str,
#   "role": "employee"|"manager"|"hr"|"unknown",
#   "history": [{"role": "user"/"assistant", "content": str}, ...],
#   "created_at": datetime
# }
sessions: dict[str, dict] = {}


# --------- KNOWLEDGE BASE (RETRIEVAL) ---------


def load_kb():
    global kb_index
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            kb_index = pickle.load(f)
        print(f"[KB] Loaded knowledge base from {INDEX_PATH}")
    else:
        print(f"[KB] WARNING: {INDEX_PATH} not found. Run: python build_hr_kb.py")
        kb_index = None


def search_kb(query: str, top_k: int = 3):
    """Return top_k most relevant chunks from the HR PDFs."""
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
                "source": c["source"],
                "page": c["page"],
                "text": c["text"],
                "score": float(sims[int(idx)]),
            }
        )
    return results


# --------- SESSION / USER MANAGEMENT ---------


def get_or_create_session(session_id: str | None, company_id: str | None, role: str | None):
    """Return session dict and a guaranteed session_id."""
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
        # Update company/role if provided
        if company_id:
            sessions[session_id]["company_id"] = company_id
        if role:
            sessions[session_id]["role"] = role

    return session_id, sessions[session_id]


# --------- AI BRAIN (GROQ) ---------


def build_role_instructions(role: str) -> str:
    role = (role or "").lower()
    if role in ("hr", "people ops", "people_ops", "people"):
        return (
            "The user is in HR / People Ops / Leadership. "
            "Prioritize structure, documentation, scripts, action plans, and follow-up steps. "
            "Make it easy to store your answer as HR documentation."
        )
    if role in ("manager", "leader", "supervisor"):
        return (
            "The user is a manager or team lead. "
            "Help them structure conversations with their team, ask good questions, "
            "and prevent escalation. Provide mediation steps, sample phrases, and "
            "clear expectations for follow-up."
        )
    if role in ("employee", "individual contributor", "staff"):
        return (
            "The user is an employee in a conflict or with an HR concern. "
            "Help them describe what’s happening neutrally, explain impact, and ask for support. "
            "Give examples of how to talk to their manager or HR."
        )
    return (
        "The user’s role is not clear. Gently invite them to clarify whether they are HR, "
        "a manager, or an employee, and adapt your guidance accordingly."
    )


def build_mode_instructions(mode: str) -> str:
    mode = (mode or "chat").lower()
    if mode == "mediation":
        return (
            "The user requested a structured mediation plan. "
            "Use the full structured format:\n"
            "1. Risk & Role Notes\n"
            "2. Neutral Situation Summary\n"
            "3. Key Issues Identified\n"
            "4. Mediation Meeting Agenda\n"
            "5. Suggested Scripts for the Mediator\n"
            "6. Suggested Agreements\n"
            "7. Follow-Up Plan\n"
            "8. HR Documentation Summary (if appropriate)\n"
            "Make it detailed but clear and practical, ready to copy-paste."
        )
    return (
        "The user is in normal chat mode. Focus on answering their question, but still use a "
        "clear structure and invite them to share role, risk, and desired outcome."
    )


def generate_chat_reply(
    user_message: str,
    role: str,
    company_id: str,
    kb_snippets: list[dict],
    history: list[dict],
    mode: str = "chat",
) -> str:
    """
    Use Groq to generate an HR-aware reply as Kinneckt.
    - Includes role instructions
    - Includes HR policy snippets from PDFs
    - Includes recent conversation history for memory
    """

    if client is None:
        # No LLM available – fallback content
        return (
            "(AI TEMPORARILY UNAVAILABLE)\n\n"
            "I’m currently running in test mode and can’t reach the AI model. "
            "However, here are some general next steps you can consider:\n\n"
            "- Write down what happened with dates, times, and who was involved.\n"
            "- Note how this is affecting your work (focus, deadlines, relationships).\n"
            "- Decide whether you want to speak with your manager, HR, or both.\n"
            "- Prepare 2–3 concrete examples you can share in a calm, factual way.\n\n"
            "Please also consider checking your company’s HR, complaint, or ethics policies "
            "for formal options. This is not legal advice."
        )

    role_instructions = build_role_instructions(role)
    mode_instructions = build_mode_instructions(mode)

    # Build a short context from HR PDFs
    context_parts = []
    for snip in kb_snippets[:3]:
        context_parts.append(
            f"From {snip.get('source','unknown')} (page {snip.get('page','?')}): "
            f"{snip.get('text','')}"
        )
    context_text = (
        "\n\n".join(context_parts)
        if context_parts
        else "No specific HR reference snippets were retrieved for this question."
    )

    system_prompt = f"""
You are Kinneckt, an HR Assistant Chatbot designed for organizations who use Kinneckt.

You support:
- HR / People Ops / Leadership
- Managers and team leads
- Employees experiencing workplace issues

You DO NOT give legal advice.
You DO NOT decide who is right or wrong.
You focus on clarity, behavior, communication, and next steps.

Tone:
- Warm, calm, neutral, and professional
- High emotional intelligence and empathy
- Validate emotions (“That sounds frustrating, let’s walk through options.”)
- Avoid judgmental language or labels.

Safety & Escalation:
- If the user describes harassment, discrimination, threats, violence, self-harm, stalking,
  or anything involving a protected class, you MUST:
  - Flag that this may need escalation to HR, leadership, legal, or emergency services.
  - Explicitly say: “This is not legal advice. Please consult your HR team or legal counsel.”
  - Help them organize dates, specific behaviors, impact on work, witnesses, and evidence.

Role-based behavior:
{role_instructions}

Default response structure (unless they ask for something else):
1) Risk & Role Check
2) Neutral Situation Summary (from the user’s perspective, not as fact)
3) Key Issues Identified (e.g., expectations, communication, tone, workload, boundaries)
4) Conversation / Mediation Plan (agenda + example scripts)
5) Suggested Agreements & Follow-Up
6) (Optional) HR Documentation Block (if they are HR/manager or explicitly want documentation)

Conversation mode instructions:
{mode_instructions}

You NEVER:
- Say someone is “guilty”, “abusive”, or “to blame”.
- Make legal determinations.
- Give legal advice.

You ALWAYS:
- Focus on behavior, impact, communication options, and next steps.
- Encourage respectful, professional wording.
- Remind the user that you are not a lawyer or therapist.
"""

    user_content = f"""
Organization/company id (for context only): {company_id}

User's message:
\"\"\"{user_message}\"\"\"

Relevant HR reference snippets (from this organization's HR manuals/resources):
{context_text}

Please respond as Kinneckt using the structure and role guidance above.
Be specific, practical, and copy-paste friendly.
"""

    # Use recent history (last ~6 turns) for memory
    message_history = []
    for m in history[-6:]:
        if m["role"] in ("user", "assistant"):
            message_history.append({"role": m["role"], "content": m["content"]})

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(message_history)
    messages.append({"role": "user", "content": user_content})

    chat_completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1200,
    )

    reply_text = chat_completion.choices[0].message.content
    return reply_text or "I wasn’t able to generate a detailed response. Please try again, or speak directly with HR."


# --------- HTML TEMPLATE (CHAT + ROLES + MEDIATION BUTTON) ---------

PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Kinneckt HR Assistant</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --kinneckt-navy: #0b2545;
      --kinneckt-navy-dark: #050f1f;
      --kinneckt-teal: #18d4b0;
      --kinneckt-teal-soft: #33e2c0;
      --kinneckt-bg: #020617;
      --kinneckt-text: #e5e7eb;
      --kinneckt-muted: #9ca3af;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #0b2545 0, #020617 45%, #000000 100%);
      color: var(--kinneckt-text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: stretch;
    }

    .app-shell {
      width: 100%;
      max-width: 480px;
      height: 100vh;
      background: linear-gradient(180deg, rgba(11,37,69,0.98), rgba(2,6,23,0.98));
      box-shadow: 0 20px 60px rgba(0,0,0,0.7);
      display: flex;
      flex-direction: column;
      border-radius: 0;
    }

    @media (min-width: 768px) {
      .app-shell {
        margin: 1.5rem 0;
        border-radius: 22px;
        overflow: hidden;
        height: calc(100vh - 3rem);
      }
    }

    .app-header {
      display: flex;
      align-items: center;
      padding: 0.75rem 1rem;
      background: linear-gradient(90deg, var(--kinneckt-navy-dark), var(--kinneckt-navy));
      border-bottom: 1px solid rgba(148,163,184,0.2);
      gap: 0.7rem;
    }

    .logo-circle {
      width: 40px;
      height: 40px;
      border-radius: 999px;
      background: radial-gradient(circle at 30% 30%, #51ffe0, var(--kinneckt-teal));
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 0 20px rgba(24,212,176,0.9);
      overflow: hidden;
    }

    .logo-circle img {
      width: 32px;
      height: 32px;
      object-fit: contain;
    }

    .brand-text {
      display: flex;
      flex-direction: column;
    }

    .brand-name {
      font-weight: 700;
      font-size: 1rem;
      letter-spacing: 0.04em;
    }

    .brand-subtitle {
      font-size: 0.7rem;
      color: var(--kinneckt-muted);
    }

    .pill-row {
      display: flex;
      gap: 0.35rem;
      margin-left: auto;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }

    .pill {
      font-size: 0.7rem;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.6);
      color: var(--kinneckt-muted);
      background: rgba(15,23,42,0.75);
      white-space: nowrap;
    }

    .pill strong {
      color: var(--kinneckt-teal-soft);
      font-weight: 600;
    }

    .content-shell {
      display: flex;
      flex-direction: column;
      flex: 1;
      overflow: hidden;
    }

    .top-controls {
      padding: 0.5rem 0.8rem;
      border-bottom: 1px solid rgba(148,163,184,0.2);
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      background: radial-gradient(circle at top, rgba(24,212,176,0.11), transparent 60%);
    }

    .field-group {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
      min-width: 0;
      flex: 1 1 140px;
    }

    .field-label {
      font-size: 0.7rem;
      color: var(--kinneckt-muted);
    }

    select, input[type="text"] {
      font-size: 0.8rem;
      padding: 0.35rem 0.55rem;
      border-radius: 999px;
      border: 1px solid rgba(148,163,184,0.6);
      background: rgba(15,23,42,0.95);
      color: var(--kinneckt-text);
      outline: none;
    }

    select:focus, input[type="text"]:focus {
      border-color: var(--kinneckt-teal);
      box-shadow: 0 0 0 1px rgba(24,212,176,0.5);
    }

    .mode-chip {
      align-self: flex-end;
      font-size: 0.72rem;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      background: rgba(15,23,42,0.9);
      border: 1px solid rgba(148,163,184,0.6);
      color: var(--kinneckt-muted);
      display: inline-flex;
      gap: 0.25rem;
      align-items: center;
      white-space: nowrap;
    }

    .mode-dot {
      width: 7px;
      height: 7px;
      border-radius: 999px;
      background: var(--kinneckt-teal);
      box-shadow: 0 0 8px rgba(24,212,176,0.9);
    }

    .chat-window {
      flex: 1;
      overflow-y: auto;
      padding: 0.75rem 0.75rem 0.5rem 0.75rem;
      background: radial-gradient(circle at top, rgba(24,212,176,0.12), rgba(15,23,42,1));
      scrollbar-width: thin;
    }

    .chat-window::-webkit-scrollbar {
      width: 6px;
    }

    .chat-window::-webkit-scrollbar-track {
      background: transparent;
    }

    .chat-window::-webkit-scrollbar-thumb {
      background: rgba(148,163,184,0.4);
      border-radius: 999px;
    }

    .message {
      margin-bottom: 0.6rem;
      max-width: 80%;
      padding: 0.55rem 0.75rem;
      border-radius: 14px;
      font-size: 0.9rem;
      white-space: pre-wrap;
      line-height: 1.35;
      position: relative;
    }

    .message.user {
      margin-left: auto;
      background: linear-gradient(135deg, var(--kinneckt-teal), var(--kinneckt-teal-soft));
      color: #020617;
      border-bottom-right-radius: 4px;
      box-shadow: 0 10px 25px rgba(24,212,176,0.4);
    }

    .message.bot {
      margin-right: auto;
      background: rgba(15,23,42,0.95);
      color: var(--kinneckt-text);
      border-bottom-left-radius: 4px;
      border: 1px solid rgba(148,163,184,0.3);
      box-shadow: 0 12px 22px rgba(15,23,42,0.6);
    }

    .message.bot::before,
    .message.user::before {
      content: "";
      position: absolute;
      width: 9px;
      height: 9px;
      bottom: -4px;
    }

    .message.bot::before {
      left: 12px;
      background: rgba(15,23,42,0.95);
      border-left: 1px solid rgba(148,163,184,0.3);
      border-bottom: 1px solid rgba(148,163,184,0.3);
      transform: rotate(45deg);
    }

    .message.user::before {
      right: 12px;
      background: linear-gradient(135deg, var(--kinneckt-teal), var(--kinneckt-teal-soft));
      transform: rotate(45deg);
    }

    .assistant-label {
      font-size: 0.7rem;
      color: var(--kinneckt-muted);
      margin-bottom: 0.15rem;
      margin-left: 2px;
    }

    .input-shell {
      padding: 0.55rem 0.75rem 0.7rem 0.75rem;
      border-top: 1px solid rgba(148,163,184,0.3);
      background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(2,6,23,1));
    }

    .input-row {
      display: flex;
      gap: 0.55rem;
      align-items: flex-end;
    }

    textarea {
      flex: 1;
      resize: none;
      font-family: inherit;
      font-size: 0.9rem;
      padding: 0.45rem 0.6rem;
      border-radius: 14px;
      border: 1px solid rgba(148,163,184,0.7);
      background: rgba(15,23,42,0.95);
      color: var(--kinneckt-text);
      max-height: 80px;
      min-height: 48px;
      outline: none;
    }

    textarea::placeholder {
      color: rgba(148,163,184,0.8);
    }

    textarea:focus {
      border-color: var(--kinneckt-teal);
      box-shadow: 0 0 0 1px rgba(24,212,176,0.5);
    }

    .button-column {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    button {
      border: none;
      border-radius: 999px;
      padding: 0.45rem 0.8rem;
      font-weight: 600;
      cursor: pointer;
      font-size: 0.8rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.3rem;
      white-space: nowrap;
    }

    button.primary {
      background: linear-gradient(135deg, var(--kinneckt-teal), var(--kinneckt-teal-soft));
      color: #020617;
      box-shadow: 0 10px 25px rgba(24,212,176,0.45);
    }

    button.secondary {
      background: rgba(15,23,42,1);
      color: var(--kinneckt-text);
      border: 1px solid rgba(148,163,184,0.7);
    }

    button span.icon {
      font-size: 0.9rem;
    }

    button:disabled {
      opacity: 0.6;
      cursor: default;
      box-shadow: none;
    }

    .footer {
      font-size: 0.7rem;
      color: var(--kinneckt-muted);
      margin-top: 0.45rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 0.35rem;
      flex-wrap: wrap;
    }

    .footer a {
      color: var(--kinneckt-teal-soft);
      text-decoration: none;
      font-weight: 500;
    }

    .footer a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div class="logo-circle">
        <img src="/static/kinneckt_logo.png" alt="Kinneckt Logo" />
      </div>
      <div class="brand-text">
        <div class="brand-name">Kinneckt HR Assistant</div>
        <div class="brand-subtitle">Always-on, empathetic HR support</div>
      </div>
      <div class="pill-row">
        <div class="pill"><strong>Confidential</strong></div>
        <div class="pill">No legal advice</div>
      </div>
    </header>

    <div class="content-shell">
      <div class="top-controls">
        <div class="field-group">
          <span class="field-label">Your role</span>
          <select id="role-select">
            <option value="">Not specified</option>
            <option value="employee">Employee</option>
            <option value="manager">Manager / Team Lead</option>
            <option value="hr">HR / People Ops</option>
          </select>
        </div>
        <div class="field-group">
          <span class="field-label">Company / client</span>
          <input id="company-id" type="text" placeholder="e.g., Acme Corp" />
        </div>
        <div class="mode-chip">
          <span class="mode-dot"></span>
          <span>Mode: <span id="mode-label">Chat</span></span>
        </div>
      </div>

      <main id="chat-window" class="chat-window">
        <div class="assistant-label">Assistant</div>
        <div class="message bot">
          Hi, I’m the Kinneckt HR Assistant. 🌱

          In a few sentences, tell me what’s going on at work and how it’s affecting you or your team.
          If you’re comfortable, include whether you’re HR, a manager, or an employee and what a
          “good outcome” would look like for you.
        </div>
      </main>

      <div class="input-shell">
        <div class="input-row">
          <textarea id="user-input" placeholder="Type your message here…"></textarea>
          <div class="button-column">
            <button id="send-btn" class="primary" onclick="sendMessage('chat')">
              <span class="icon">➤</span>
              <span>Send</span>
            </button>
            <button id="mediation-btn" class="secondary" onclick="sendMessage('mediation')">
              <span class="icon">⚖</span>
              <span>Mediation Plan</span>
            </button>
          </div>
        </div>
        <div class="footer">
          <span>Powered by Kinneckt · This is not legal advice.</span>
          <a href="/admin" target="_blank">Open admin dashboard</a>
        </div>
      </div>
    </div>
  </div>

  <script>
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const mediationBtn = document.getElementById('mediation-btn');
    const roleSelect = document.getElementById('role-select');
    const companyInput = document.getElementById('company-id');
    const modeLabel = document.getElementById('mode-label');

    let sessionId = localStorage.getItem('kinneckt_session_id');
    if (!sessionId) {
      sessionId = crypto.randomUUID ? crypto.randomUUID() : (Date.now().toString());
      localStorage.setItem('kinneckt_session_id', sessionId);
    }

    function appendMessage(text, sender) {
      const div = document.createElement('div');
      div.classList.add('message', sender === 'user' ? 'user' : 'bot');

      if (sender === 'bot') {
        const label = document.createElement('div');
        label.classList.add('assistant-label');
        label.textContent = 'Assistant';
        chatWindow.appendChild(label);
      }

      div.textContent = text;
      chatWindow.appendChild(div);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    async function sendMessage(mode) {
      const text = userInput.value.trim();
      if (!text) return;

      appendMessage(text, 'user');
      userInput.value = '';
      sendBtn.disabled = true;
      mediationBtn.disabled = true;
      modeLabel.textContent = mode === 'mediation' ? 'Mediation Plan' : 'Chat';

      appendMessage('Thinking…', 'bot');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: text,
            session_id: sessionId,
            role: roleSelect.value,
            company_id: companyInput.value,
            mode: mode
          })
        });
        const data = await res.json();

        const botMessages = chatWindow.getElementsByClassName('bot');
        if (botMessages.length > 0) {
          const lastBot = botMessages[botMessages.length - 1];
          if (lastBot.textContent === 'Thinking…') {
            chatWindow.removeChild(lastBot.previousSibling); // assistant-label
            chatWindow.removeChild(lastBot);
          }
        }

        appendMessage(data.reply || 'No reply received.', 'bot');
      } catch (err) {
        const botMessages = chatWindow.getElementsByClassName('bot');
        if (botMessages.length > 0) {
          const lastBot = botMessages[botMessages.length - 1];
          if (lastBot.textContent === 'Thinking…') {
            chatWindow.removeChild(lastBot.previousSibling);
            chatWindow.removeChild(lastBot);
          }
        }
        appendMessage('Error talking to server: ' + err, 'bot');
        console.error(err);
      } finally {
        sendBtn.disabled = false;
        mediationBtn.disabled = false;
        userInput.focus();
      }
    }

    userInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage('chat');
      }
    });
  </script>
</body>
</html>
"""


# --------- ADMIN DASHBOARD ---------


@app.route("/admin", methods=["GET"])
def admin_dashboard():
    # Simple SaaS-style admin panel to inspect usage by company + sessions
    rows_html = []
    company_stats: dict[str, dict] = {}

    for sid, sess in sessions.items():
        role = sess.get("role", "unknown")
        company_id = sess.get("company_id", "default")
        created = sess.get("created_at", datetime.utcnow())
        history = sess.get("history", [])
        message_count = len(history)
        preview = ""

        # Last user message preview
        for m in reversed(history):
            if m["role"] == "user":
                preview = m["content"][:120].replace("<", "&lt;").replace(">", "&gt;")
                break

        # Row for the session table
        rows_html.append(
            f"<tr>"
            f"<td>{sid}</td>"
            f"<td>{company_id}</td>"
            f"<td>{role}</td>"
            f"<td>{message_count}</td>"
            f"<td>{created.isoformat()}Z</td>"
            f"<td>{preview}</td>"
            f"</tr>"
        )

        # Aggregate stats per company
        if company_id not in company_stats:
            company_stats[company_id] = {"sessions": 0, "messages": 0}
        company_stats[company_id]["sessions"] += 1
        company_stats[company_id]["messages"] += message_count

    # Build HTML for company summary
    company_rows = []
    for cid, stats in company_stats.items():
        company_rows.append(
            f"<tr>"
            f"<td>{cid}</td>"
            f"<td>{stats['sessions']}</td>"
            f"<td>{stats['messages']}</td>"
            f"</tr>"
        )

    html = f"""
    <html>
    <head>
      <title>Kinneckt Admin Dashboard</title>
      <style>
        body {{
          font-family: Arial, sans-serif;
          padding: 1rem;
          background: #f3f4f6;
        }}
        h1 {{
          margin-top: 0;
        }}
        h2 {{
          margin-top: 1.5rem;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          background: #ffffff;
          margin-top: 0.5rem;
        }}
        th, td {{
          border: 1px solid #e5e7eb;
          padding: 0.4rem 0.5rem;
          font-size: 0.85rem;
        }}
        th {{
          background: #f9fafb;
          text-align: left;
        }}
        tr:nth-child(even) {{
          background: #f3f4f6;
        }}
      </style>
    </head>
    <body>
      <h1>Kinneckt Admin Dashboard</h1>
      <p>In-memory view of active sessions (for demo / SaaS-style monitoring).</p>

      <h2>Company Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Company</th>
            <th>Active Sessions</th>
            <th>Total Messages</th>
          </tr>
        </thead>
        <tbody>
          {''.join(company_rows) if company_rows else '<tr><td colspan="3">No sessions yet.</td></tr>'}
        </tbody>
      </table>

      <h2>Session Detail</h2>
      <table>
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Company</th>
            <th>Role</th>
            <th>Messages</th>
            <th>Created At (UTC)</th>
            <th>Last user message preview</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html) if rows_html else '<tr><td colspan="6">No sessions yet.</td></tr>'}
        </tbody>
      </table>
    </body>
    </html>
    """

    return html


# --------- FLASK ROUTES ---------


@app.route("/", methods=["GET"])
def home():
    return render_template_string(PAGE_TEMPLATE)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Main chat endpoint for the Kinneckt HR Assistant.
    - Accepts JSON from the web UI or mobile app.
    - Requires a company/client name for SaaS usage.
    - Uses the HR LLM brain when available.
    - Falls back to a simple echo-style response if anything goes wrong.
    """
    try:
        data = request.get_json(force=True) or {}
    except Exception as e:
        print("[API] JSON parse error:", e)
        fallback = (
            "(BACKEND ERROR – JSON)\n\n"
            "I had trouble reading your request. Please try again, or update the app if this keeps happening."
        )
        return jsonify({"reply": fallback}), 200  # 200 so frontend doesn't show network error

    user_message = data.get("message", "")
    incoming_session_id = data.get("session_id")
    role = data.get("role") or ""
    raw_company = (data.get("company_id") or "").strip()
    company_id = raw_company or "default"
    mode = data.get("mode") or "chat"

    # Session handling (we pass company_id so sessions are tagged)
    session_id, session = get_or_create_session(incoming_session_id, company_id, role)
    history = session["history"]

    print("[API] /api/chat hit from client")
    print(" message:", user_message)
    print(" role:", role, "company:", company_id, "mode:", mode, "session:", session_id)

    # If company/client not provided, nudge them to enter it
    if company_id == "default":
        reply_text = (
            "Hi, I’m the Kinneckt HR Assistant for Kinneckt.\n\n"
            "To personalize this guidance for your organization or client, "
            "please add your company or client name in the “Company / client” field, "
            "then send your question again.\n\n"
            "For example: “Acme Corp”, “Kinneckt Internal”, or the client you’re supporting."
        )
        return jsonify({"reply": reply_text, "session_id": session_id}), 200

    # If user sent nothing
    if not user_message.strip():
        reply_text = (
            "Hi, I’m the Kinneckt HR Assistant. It looks like your message was empty.\n\n"
            "You can tell me:\n"
            "- Your role (HR, manager, employee)\n"
            "- A brief description of what happened (2–4 sentences)\n"
            "- How it’s affecting you or work\n"
            "- What kind of help you’re hoping for (clarity, mediation, documentation, etc.)"
        )
        return jsonify({"reply": reply_text, "session_id": session_id}), 200

    # Retrieve relevant HR snippets
    kb_snippets = search_kb(user_message)

    # Fallback if LLM call fails
    fallback_reply = (
        "(TEST / FALLBACK RESPONSE FROM BACKEND)\n\n"
        f"You said: {user_message}\n\n"
        f"Role: {role or 'not specified'}\n"
        f"Company: {company_id or 'not specified'}\n"
        f"Mode: {mode}\n"
        "\nIf you see this, the connection between the Kinneckt app and backend is working, "
        "but the AI brain may be unavailable."
    )

    # Try to generate an HR-smart answer
    try:
        reply_text = generate_chat_reply(
            user_message=user_message,
            role=role,
            company_id=company_id,
            kb_snippets=kb_snippets,
            history=history,
            mode=mode,
        )
    except Exception as e:
        print("[API] Error in generate_chat_reply:", e)
        reply_text = fallback_reply

    # Update conversation history
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply_text})

    return jsonify({"reply": reply_text, "session_id": session_id}), 200


if __name__ == "__main__":
    load_kb()
    app.run(host="0.0.0.0", port=5000, debug=True)

