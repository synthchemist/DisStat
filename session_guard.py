"""Optional cap on simultaneous users, so a small server (e.g. a Raspberry Pi) is not overloaded.

Off by default. Turn on with environment variables:
    DISSOLUTION_MAX_USERS=5         maximum people using the app at once (0 / unset = no limit)
    DISSOLUTION_IDLE_MINUTES=20     a user who has done nothing for this long gives up their place

A person holds a place while their browser tab is open and they have used the page within the
idle window. Closing the tab frees the place straight away.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional


class SessionGuard:
    def __init__(self, max_users: int, idle_seconds: float = 1200.0,
                 is_connected: Optional[Callable[[str], bool]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.max_users, self.idle_seconds = int(max_users), float(idle_seconds)
        self.is_connected, self.clock = is_connected, clock
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        for sid, last in list(self._seen.items()):
            gone = self.is_connected is not None and not self.is_connected(sid)
            if gone or now - last > self.idle_seconds:
                del self._seen[sid]

    def admit(self, session_id: str) -> bool:
        """True if this session may use the app (and refreshes its place); False if the app is full."""
        if self.max_users <= 0:
            return True
        with self._lock:
            now = self.clock()
            self._prune(now)
            if session_id in self._seen or len(self._seen) < self.max_users:
                self._seen[session_id] = now
                return True
            return False

    @property
    def active(self) -> int:
        with self._lock:
            self._prune(self.clock())
            return len(self._seen)


def _env_number(name: str, default: float) -> float:
    """Read a number from an environment variable, else from Streamlit secrets (Community Cloud)."""
    raw = os.environ.get(name)
    if raw is None:
        try:
            import streamlit as st
            raw = st.secrets.get(name)
        except Exception:           # no secrets file configured
            raw = None
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def enforce() -> None:
    """Call near the top of the Streamlit script, after st.set_page_config.
    Shows a 'busy' page and stops the script when the cap is reached."""
    max_users = int(_env_number("DISSOLUTION_MAX_USERS", 0))
    if max_users <= 0:
        return
    import streamlit as st
    from streamlit.runtime import Runtime
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    @st.cache_resource
    def _guard(limit: int, idle_minutes: float) -> SessionGuard:
        def connected(sid: str) -> bool:
            try:
                return bool(Runtime.instance().is_active_session(sid))
            except Exception:       # runtime not reachable: fall back to the idle timeout only
                return True
        return SessionGuard(limit, idle_minutes * 60, connected)

    ctx = get_script_run_ctx()
    if ctx is None:
        return
    guard = _guard(max_users, _env_number("DISSOLUTION_IDLE_MINUTES", 20))
    if not guard.admit(ctx.session_id):
        st.title("Server is busy")
        st.warning(f"The maximum of {max_users} people are using this tool right now. "
                   "Please wait a minute and press the button to try again.")
        st.button("Try again")
        st.stop()
