"""In-memory session manager for the Clinical Screening Assistant.

Sessions are ephemeral (TTL 30 min), keyed by UUID4, and never persisted to disk.
No PII is logged.  Rate limiting is per client IP.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


SESSION_TTL = 1800       # 30 minutes, absolute maximum
IDLE_TTL = 600           # 10 minutes without any request
MAX_SESSIONS_PER_IP = 5
RATE_LIMIT_COUNT = 20    # requests per window
RATE_LIMIT_WINDOW = 60   # seconds


@dataclass
class SessionEntry:
    """One active assistant session."""

    session_id: str
    thread_config: dict    # LangGraph config with thread_id
    created_at: float
    client_ip: str
    last_seen: float = 0.0   # refreshed by every request
    image: Any = None      # PIL Image held temporarily during processing


class SessionManager:
    """Manage assistant sessions, TTL expiry, and per-IP rate limiting."""

    def __init__(
        self,
        ttl: int = SESSION_TTL,
        idle_ttl: float = IDLE_TTL,
        max_per_ip: int = MAX_SESSIONS_PER_IP,
        rate_limit: tuple[int, int] = (RATE_LIMIT_COUNT, RATE_LIMIT_WINDOW),
        on_expire: Optional[Callable[[str], None]] = None,
    ):
        self._lock = threading.RLock()
        self.ttl = ttl
        self.idle_ttl = idle_ttl
        # Called with the session id whenever a session ends (expiry or explicit removal) so that data held
        # elsewhere (graph checkpointer, image cache) is deleted too.
        self.on_expire = on_expire
        self.max_per_ip = max_per_ip
        self.rate_count, self.rate_window = rate_limit
        self._sessions: dict[str, SessionEntry] = {}
        self._ip_hits: dict[str, deque] = defaultdict(deque)

    # ---- housekeeping ----

    def _sweep(self) -> None:
        """Remove expired sessions (cheap: called on every public method)."""
        with self._lock:
            now = time.monotonic()
            expired = [
                sid for sid, s in list(self._sessions.items())
                if now - s.created_at > self.ttl or now - s.last_seen > self.idle_ttl
            ]
            for sid in expired:
                self._sessions.pop(sid, None)
                self._forget(sid)

    # ---- rate limiting ----

    def _check_rate(self, ip: str) -> bool:
        """Return ``True`` if the request is within the rate limit."""
        now = time.monotonic()
        q = self._ip_hits[ip]
        while q and now - q[0] > self.rate_window:
            q.popleft()
        if len(q) >= self.rate_count:
            return False
        q.append(now)
        return True

    # ---- public API ----

    def create_checked(self, client_ip: str) -> tuple[Optional[SessionEntry], Optional[str]]:
        """Create a session, or explain the refusal: ``(entry, None)`` or ``(None, reason)`` with reason
        ``"rate_limited"`` (too many requests per minute) or ``"too_many_sessions"`` (too many open sessions)."""
        self._sweep()

        if not self._check_rate(client_ip):
            return None, "rate_limited"

        active = sum(1 for s in self._sessions.values() if s.client_ip == client_ip)
        if active >= self.max_per_ip:
            return None, "too_many_sessions"

        with self._lock:
            return self._create_locked(client_ip), None

    def create(self, client_ip: str) -> Optional[SessionEntry]:
        """Like :meth:`create_checked` but returns only the entry (``None`` when refused)."""
        return self.create_checked(client_ip)[0]

    def _create_locked(self, client_ip: str) -> Optional[SessionEntry]:
        sid = uuid.uuid4().hex
        entry = SessionEntry(
            session_id=sid,
            thread_config={"configurable": {"thread_id": sid}},
            created_at=time.monotonic(),
            last_seen=time.monotonic(),
            client_ip=client_ip,
        )
        self._sessions[sid] = entry
        return entry

    def get(self, session_id: str) -> Optional[SessionEntry]:
        """Look up a session by ID.  Returns ``None`` if expired or unknown."""
        self._sweep()
        entry = self._sessions.get(session_id)
        if entry is not None:
            entry.last_seen = time.monotonic()
        return entry

    def check_rate(self, client_ip: str) -> bool:
        """Check (and record) a rate-limit hit for *client_ip*."""
        return self._check_rate(client_ip)

    def _forget(self, session_id: str) -> None:
        if self.on_expire is not None:
            try:
                self.on_expire(session_id)
            except Exception:  # cleanup must never break a request
                pass

    def remove(self, session_id: str) -> None:
        """Explicitly delete a session and everything stored for it."""
        self._sessions.pop(session_id, None)
        self._forget(session_id)

    @property
    def active_count(self) -> int:
        self._sweep()
        return len(self._sessions)


def start_sweeper(manager: SessionManager, interval: float = 60.0) -> threading.Thread:
    """Expire sessions on a timer. Without this, expiry only happens when some other request arrives,
    so an idle server would keep a finished patient's answers in memory indefinitely."""

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                manager._sweep()
            except Exception:  # never let the sweeper die
                pass

    thread = threading.Thread(target=loop, daemon=True, name="session-sweeper")
    thread.start()
    return thread
