"""In-memory conversation session store.

Each session tracks up to MAX_TURNS of (user, assistant) conversation turns.
Session IDs are arbitrary strings provided by the client.
"""
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

# Maximum conversation turns retained per session
MAX_TURNS: int = 10

# session_id -> deque of (user_message, assistant_message) tuples
_sessions: Dict[str, Deque[Tuple[str, str]]] = {}


def get_history(session_id: str) -> List[Dict[str, str]]:
    """Return conversation history for *session_id* as a list of role/content dicts."""
    if session_id not in _sessions:
        return []
    return [
        {"role": "user", "content": user_msg}
        if i % 2 == 0
        else {"role": "assistant", "content": asst_msg}
        for turn in _sessions[session_id]
        for i, msg in enumerate([turn[0], turn[1]])
        # flatten each turn into two dicts
        for user_msg, asst_msg in [turn]
    ]


def get_turns(session_id: str) -> List[Tuple[str, str]]:
    """Return raw (user, assistant) tuples for *session_id*."""
    return list(_sessions.get(session_id, []))


def add_turn(session_id: str, user_message: str, assistant_message: str) -> None:
    """Append a turn to the session, evicting the oldest if over MAX_TURNS."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=MAX_TURNS)
    _sessions[session_id].append((user_message, assistant_message))


def clear_session(session_id: str) -> None:
    """Remove all history for *session_id*."""
    _sessions.pop(session_id, None)


def list_sessions() -> List[str]:
    """Return all active session IDs."""
    return list(_sessions.keys())
