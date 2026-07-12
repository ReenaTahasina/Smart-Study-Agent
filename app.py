import os
import re
import json
import time
import requests
from functools import wraps
from flask import (
    Flask, request, jsonify, render_template,
    redirect, url_for, session, flash
)
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))

# ── IBM watsonx.ai config ─────────────────────────────────────────────────────
WATSONX_URL = os.getenv("WATSONX_URL",
    "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29")
PROJECT_ID  = os.getenv("PROJECT_ID",  "e5338631-1ef1-4dd4-b1ae-d703b5fc87f7")
MODEL_ID    = os.getenv("MODEL_ID",    "ibm/granite-4-h-small")
IBM_API_KEY = os.getenv("IBM_API_KEY", "YxJ5uVqRqc7BFYFgWesb2Vp0H546nl1AxmkqKHJoffQO")

# ── In-memory user store (replace with a real DB for production) ──────────────
_users: dict[str, dict] = {}   # { email: {name, password} }

# ── IAM token cache ───────────────────────────────────────────────────────────
_iam_token_cache: dict = {"token": None, "expiry": 0}


def get_iam_token() -> str:
    if _iam_token_cache["token"] and time.time() < _iam_token_cache["expiry"]:
        return _iam_token_cache["token"]

    if not IBM_API_KEY or IBM_API_KEY == "your-ibm-cloud-api-key-here":
        raise ValueError(
            "IBM API key is not configured. "
            "Set IBM_API_KEY in your .env file."
        )

    try:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": IBM_API_KEY,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        raise ValueError(
            "Cannot reach IBM IAM service. Check your internet connection."
        )
    except requests.exceptions.Timeout:
        raise ValueError(
            "IBM IAM request timed out. Try again in a moment."
        )

    if resp.status_code == 400:
        raise ValueError(
            "IBM API key is invalid or expired (IAM returned 400). "
            "Please generate a new API key at https://cloud.ibm.com/iam/apikeys "
            "and update IBM_API_KEY in your .env file."
        )
    if resp.status_code == 401:
        raise ValueError(
            "IBM API key is unauthorised (IAM returned 401). "
            "Ensure the key has the correct permissions and update .env."
        )
    if not resp.ok:
        raise ValueError(
            f"IBM IAM error {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    _iam_token_cache["token"]  = data["access_token"]
    _iam_token_cache["expiry"] = time.time() + data.get("expires_in", 3600) - 60
    return _iam_token_cache["token"]


def call_granite(system_prompt: str, user_message: str, max_tokens: int = 1024) -> str:
    token = get_iam_token()
    payload = {
        "model_id":   MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature":    0.7,
            "top_p":          0.9,
        },
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/json",
    }
    resp = requests.post(WATSONX_URL, json=payload, headers=headers, timeout=60)
    # Return a clear error message if IBM API responds with an error status
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:300]
        raise ValueError(f"IBM API error {resp.status_code}: {detail}")
    data = resp.json()
    # Safely extract the model reply regardless of response shape
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected IBM API response shape: {data}") from exc


# ── Global JSON error handler — ensures ALL unhandled exceptions return JSON ──
@app.errorhandler(Exception)
def handle_exception(exc):
    """Return JSON for every unhandled server error so the browser never
    receives an HTML 500 page from an /api/ route."""
    import traceback
    if request.path.startswith("/api/"):
        app.logger.error("Unhandled exception on %s: %s", request.path,
                         traceback.format_exc())
        return jsonify({"error": str(exc) or "An unexpected server error occurred."}), 500
    # For non-API routes, let Flask's default handler take over
    raise exc


# ── Auth decorator ────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            # API routes must get JSON, not an HTML redirect page
            if request.path.startswith("/api/"):
                return jsonify({"error": "Session expired. Please log in again."}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def landing():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = _users.get(email)
        if user and user["password"] == password:
            session["user"] = {"email": email, "name": user["name"]}
            return redirect(url_for("dashboard"))
        error = "Invalid email or password."
    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user" in session:
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if not name or not email or not password:
            error = "All fields are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif email in _users:
            error = "An account with that email already exists."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            _users[email] = {"name": name, "password": password}
            session["user"] = {"email": email, "name": name}
            return redirect(url_for("dashboard"))
    return render_template("signup.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", user=session["user"])


# ── AI API routes (all protected) ─────────────────────────────────────────────

@app.route("/api/summary", methods=["POST"])
@login_required
def generate_summary():
    data  = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    level = data.get("level", "intermediate")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    try:
        system = (
            "You are an expert academic tutor. Produce clear, well-structured summaries "
            "with headings, bullet points, and key takeaways suitable for students."
        )
        user = (
            f"Write a comprehensive study summary about: {topic}\n"
            f"Target level: {level}\n"
            "Include: Overview, Key Concepts, Important Details, and Key Takeaways."
        )
        result = call_granite(system, user, max_tokens=1200)
        return jsonify({"summary": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/study_plan", methods=["POST"])
@login_required
def generate_study_plan():
    data  = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    days  = data.get("days", 7)
    hours = data.get("hours_per_day", 2)
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    try:
        system = (
            "You are an expert academic coach. Create detailed, actionable study plans "
            "with daily goals, resources, and progress milestones."
        )
        user = (
            f"Create a {days}-day study plan for: {topic}\n"
            f"Study time available: {hours} hours per day\n"
            "Format as a day-by-day schedule with goals, activities, and milestones."
        )
        result = call_granite(system, user, max_tokens=1400)
        return jsonify({"plan": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/flashcards", methods=["POST"])
@login_required
def generate_flashcards():
    data  = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    count = int(data.get("count", 10))
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    try:
        system = (
            "You are an expert educator. Generate concise, accurate flashcards. "
            "Always respond with ONLY a valid JSON array — no markdown, no extra text. "
            'Format: [{"front": "question", "back": "answer"}, ...]'
        )
        user = (
            f"Generate exactly {count} flashcards for studying: {topic}\n"
            f"Return ONLY a JSON array of {count} objects with 'front' and 'back' keys."
        )
        raw   = call_granite(system, user, max_tokens=1200)
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return jsonify({"flashcards": json.loads(match.group())})
            except json.JSONDecodeError:
                pass
        return jsonify({"flashcards": [], "raw": raw})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/quiz", methods=["POST"])
@login_required
def generate_quiz():
    data  = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    count = int(data.get("count", 5))
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    try:
        system = (
            "You are an expert quiz creator. Generate multiple-choice questions. "
            "Always respond with ONLY a valid JSON array — no markdown, no extra text. "
            'Format: [{"question":"...","options":["A)...","B)...","C)...","D)..."],'
            '"answer":"A","explanation":"..."}]'
        )
        user = (
            f"Generate exactly {count} multiple-choice quiz questions about: {topic}\n"
            f"Return ONLY a JSON array of {count} question objects."
        )
        raw   = call_granite(system, user, max_tokens=1400)
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return jsonify({"quiz": json.loads(match.group())})
            except json.JSONDecodeError:
                pass
        return jsonify({"quiz": [], "raw": raw})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/explain", methods=["POST"])
@login_required
def explain_concept():
    data    = request.get_json(force=True)
    concept = data.get("concept", "").strip()
    style   = data.get("style", "simple")
    if not concept:
        return jsonify({"error": "Concept is required"}), 400
    try:
        style_map = {
            "simple":   "Explain like I'm 10 years old, using simple analogies.",
            "detailed": "Give a thorough, technical explanation with examples.",
            "visual":   "Explain using step-by-step analogies and mental images.",
        }
        system = "You are a brilliant teacher who can explain any concept clearly and engagingly."
        user   = f'{style_map.get(style, style_map["simple"])}\n\nConcept to explain: {concept}'
        result = call_granite(system, user, max_tokens=900)
        return jsonify({"explanation": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chat", methods=["POST"])
@login_required
def study_chat():
    data     = request.get_json(force=True)
    question = data.get("question", "").strip()
    context  = data.get("context", "")
    if not question:
        return jsonify({"error": "Question is required"}), 400
    try:
        system = (
            "You are an intelligent study assistant. Answer academic questions clearly, "
            "provide examples, and encourage deeper understanding."
        )
        user   = f"{('Context: ' + context + chr(10)) if context else ''}{question}"
        result = call_granite(system, user, max_tokens=800)
        return jsonify({"answer": result})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
