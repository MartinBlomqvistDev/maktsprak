"""Supabase client initialisation for MaktspråkAI.

Two clients are provided:

- ``supabase``       — anon key, respects Row Level Security.  Use for reads.
- ``supabase_write`` — service-role key, bypasses RLS.  Use for writes.

Clients are created lazily on first access so that importing this module
never raises an error when credentials are absent (useful in tests).
"""

from __future__ import annotations

import os

from supabase import Client, create_client

from ..logger import get_logger

logger = get_logger()

# ---------------------------------------------------------------------------
# Streamlit secret detection
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    _USE_STREAMLIT: bool = get_script_run_ctx() is not None
except Exception:
    _USE_STREAMLIT = False


def _resolve_credentials() -> tuple[str, str, str]:
    """Return (url, anon_key, service_key) from Streamlit Secrets or env vars.

    Raises:
        EnvironmentError: If ``SUPABASE_URL`` or ``SUPABASE_KEY`` are missing.
    """
    if _USE_STREAMLIT:
        url = st.secrets["SUPABASE_URL"]
        anon = st.secrets["SUPABASE_KEY"]
        service = st.secrets.get("SUPABASE_SERVICE_KEY", anon)
    else:
        url = os.environ.get("SUPABASE_URL", "")
        anon = os.environ.get("SUPABASE_KEY", "")
        service = os.environ.get("SUPABASE_SERVICE_KEY", anon)

    if not url or not anon:
        raise OSError(
            "SUPABASE_URL and SUPABASE_KEY must be set in .env or Streamlit Secrets. "
            "See .env.example for the full list of required variables."
        )
    return url, anon, service


# ---------------------------------------------------------------------------
# Lazy client singletons
# ---------------------------------------------------------------------------
_supabase: Client | None = None
_supabase_write: Client | None = None


def _get_read_client() -> Client:
    global _supabase
    if _supabase is None:
        url, anon, _ = _resolve_credentials()
        _supabase = create_client(url, anon)
        logger.debug("Supabase read client initialised.")
    return _supabase


def _get_write_client() -> Client:
    global _supabase_write
    if _supabase_write is None:
        url, _, service = _resolve_credentials()
        _supabase_write = create_client(url, service)
        logger.debug("Supabase write client initialised.")
    return _supabase_write


class _LazyClient:
    """Proxy that forwards attribute access to the underlying Supabase client.

    This allows module-level ``supabase`` and ``supabase_write`` names to
    behave like real clients without triggering the connection at import time.
    """

    def __init__(self, factory):
        object.__setattr__(self, "_factory", factory)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_factory")(), name)


supabase: Client = _LazyClient(_get_read_client)  # type: ignore[assignment]
supabase_write: Client = _LazyClient(_get_write_client)  # type: ignore[assignment]
