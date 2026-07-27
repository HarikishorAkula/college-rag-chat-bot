"""
guardrails.py
Lightweight input/output guardrails for the College Notes RAG Chatbot.
"""

import re
import time

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MIN_QUESTION_LEN = 2
MAX_QUESTION_LEN = 800
MAX_REQUESTS_PER_WINDOW = 15
RATE_LIMIT_WINDOW_SECONDS = 60

INJECTION_PATTERNS = [
    r"ignore (all|any|the)?\s*(previous|prior|above)?\s*instructions",
    r"disregard (all|any|the)?\s*(previous|prior|above)?\s*instructions",
    r"reveal (your|the) (system|hidden)?\s*prompt",
    r"show (me )?(your|the) (system|hidden)?\s*prompt",
    r"what (are|is) your (system )?instructions",
    r"you are now",
    r"act as (a|an) ",
    r"pretend (you are|to be)",
    r"jailbreak",
    r"developer mode",
    r"forget (everything|all) (you (were|are) told|instructions)",
    r"do anything now",
    r"\bDAN\b",
    r"override (your|the) (rules|instructions|guidelines)",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

BLOCKED_TERMS = [
    "kill yourself", "suicide instructions", "how to make a bomb",
    "child sexual", "csam",
]
_BLOCKED_RE = re.compile("|".join(re.escape(t) for t in BLOCKED_TERMS), re.IGNORECASE)

LEAK_PATTERNS = [
    r"GROQ_API_KEY\s*[:=]\s*\S+",
    r"You are a College Notes Assistant\.?",
    r"Priority rules:",
    r"Uploaded Document Context:",
]
_LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


def check_input(question: str):
    if question is None:
        return False, "Please enter a question."

    q = question.strip()

    if len(q) < MIN_QUESTION_LEN:
        return False, "Please enter a valid question."

    if len(q) > MAX_QUESTION_LEN:
        return False, f"Your question is too long (max {MAX_QUESTION_LEN} characters). Please shorten it."

    if _BLOCKED_RE.search(q):
        return False, "This question can't be answered by this assistant. Please ask something related to your notes."

    if _INJECTION_RE.search(q):
        return False, "That looks like an attempt to change how the assistant behaves, which isn't allowed. Please ask a normal study question."

    if re.fullmatch(r"(.)\1{6,}", q.replace(" ", "")):
        return False, "Please enter a real question."

    return True, ""


def sanitize_output(answer: str) -> str:
    if not answer:
        return answer
    cleaned = _LEAK_RE.sub("[redacted]", answer)
    return cleaned


def check_rate_limit(session_state):
    now = time.time()
    timestamps = session_state.get("_rl_timestamps", [])

    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]

    if len(timestamps) >= MAX_REQUESTS_PER_WINDOW:
        wait = int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0]))
        session_state["_rl_timestamps"] = timestamps
        return False, f"You're asking questions too quickly. Please wait {max(wait, 1)} seconds and try again."

    timestamps.append(now)
    session_state["_rl_timestamps"] = timestamps
    return True, ""
