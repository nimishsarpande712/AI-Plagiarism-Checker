"""
Supabase database integration for AI Plagiarism Checker.
Handles all database operations: storing analyses, retrieving history,
generating shareable reports, and tracking usage.
"""

import os
import logging
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Supabase client (lazy-loaded)
_supabase_client = None
_supabase_available = False


def _get_client():
    """Lazy-load the Supabase client."""
    global _supabase_client, _supabase_available

    if _supabase_client is not None:
        return _supabase_client

    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set. Database features disabled.")
            _supabase_available = False
            return None

        _supabase_client = create_client(url, key)
        _supabase_available = True
        logger.info("Supabase client initialized successfully")
        return _supabase_client

    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")
        _supabase_available = False
        return None


def is_available():
    """Check if the database is configured and available."""
    if _supabase_available:
        return True
    # Try to initialize
    _get_client()
    return _supabase_available


def db_safe(default=None):
    """Decorator: silently return default if DB is unavailable or errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_available():
                return default
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Database error in {func.__name__}: {e}")
                return default
        return wrapper
    return decorator


# ────────────────────────────────────────────
# Authentication (Supabase Auth)
# ────────────────────────────────────────────

def signup_user(email: str, password: str, full_name: str = "") -> dict:
    """Register a new user via Supabase Auth.

    Returns:
        Dict with user info and session tokens, or {'error': '...'} on failure.
    """
    client = _get_client()
    if not client:
        return {"error": "Database not available"}

    try:
        options = {}
        if full_name:
            options["data"] = {"full_name": full_name}

        response = client.auth.sign_up({
            "email": email,
            "password": password,
            "options": options,
        })

        user = response.user
        session = response.session

        result = {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": (user.user_metadata or {}).get("full_name", ""),
                "created_at": str(user.created_at) if user.created_at else None,
            },
            "message": "Account created successfully",
        }

        # If email confirmation is disabled, session is returned immediately
        if session:
            result["access_token"] = session.access_token
            result["refresh_token"] = session.refresh_token
            result["expires_in"] = session.expires_in

        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Signup error: {error_msg}")
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
            return {"error": "An account with this email already exists"}
        return {"error": f"Signup failed: {error_msg}"}


def login_user(email: str, password: str) -> dict:
    """Log in a user via Supabase Auth.

    Returns:
        Dict with user info and session tokens, or {'error': '...'} on failure.
    """
    client = _get_client()
    if not client:
        return {"error": "Database not available"}

    try:
        response = client.auth.sign_in_with_password({
            "email": email,
            "password": password,
        })

        user = response.user
        session = response.session

        return {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": (user.user_metadata or {}).get("full_name", ""),
                "created_at": str(user.created_at) if user.created_at else None,
            },
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "expires_in": session.expires_in,
        }

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Login error: {error_msg}")
        if "invalid" in error_msg.lower() or "credentials" in error_msg.lower():
            return {"error": "Invalid email or password"}
        return {"error": f"Login failed: {error_msg}"}


def get_user_from_token(access_token: str) -> dict:
    """Validate a Supabase JWT and return the user.

    Returns:
        User dict with id, email, full_name, or None if invalid.
    """
    client = _get_client()
    if not client:
        return None

    try:
        response = client.auth.get_user(access_token)
        user = response.user
        if not user:
            return None

        return {
            "id": str(user.id),
            "email": user.email,
            "full_name": (user.user_metadata or {}).get("full_name", ""),
        }

    except Exception as e:
        logger.debug(f"Token validation failed: {e}")
        return None


# ────────────────────────────────────────────
# Analysis CRUD
# ────────────────────────────────────────────

@db_safe(default=None)
def save_analysis(
    input_text: str,
    results: dict,
    input_source: str = "paste",
    original_filename: str = None,
    user_id: str = None,
    session_id: str = None,
) -> dict:
    """Save a completed analysis to the database.

    Args:
        input_text: The text that was analyzed.
        results: The full results dict from the /check endpoint.
        input_source: How the text was provided ('paste', 'pdf', 'docx', 'txt').
        original_filename: Name of uploaded file, if any.
        user_id: Authenticated user ID (optional).
        session_id: Anonymous session ID (optional).

    Returns:
        The inserted row as a dict, or None on failure.
    """
    client = _get_client()
    if not client:
        return None

    # Determine risk level
    ai_prob = results.get("analysis", {}).get("signals", {}).get("score", 0)
    ai_prob_pct = round(ai_prob * 100)
    if ai_prob_pct >= 70:
        risk_level = "high"
    elif ai_prob_pct >= 40:
        risk_level = "medium"
    else:
        risk_level = "low"

    reasons = results.get("analysis", {}).get("reasons", [])
    if any("too short" in r.lower() for r in reasons):
        risk_level = "inconclusive"

    row = {
        "input_text": input_text[:50000],  # Cap at 50k chars
        "input_source": input_source,
        "original_filename": original_filename,
        "user_id": user_id,
        "session_id": session_id,
        # Core metrics
        "perplexity": results.get("perplexity"),
        "burstiness": results.get("burstiness"),
        "entropy": results.get("entropy"),
        "token_count": results.get("token_count"),
        # AI detection
        "is_ai_generated": results.get("is_ai_generated", False),
        "ai_probability": ai_prob_pct / 100.0,
        "confidence": results.get("analysis", {}).get("confidence"),
        "risk_level": risk_level,
        "verdict": results.get("analysis", {}).get("overall", ""),
        "reasons": reasons,
        # Additional metrics
        "style_consistency": results.get("style_consistency"),
        "complexity": results.get("complexity"),
        "variability": results.get("variability"),
        "readability": results.get("readability"),
        # JSONB columns
        "signal_details": results.get("analysis", {}).get("signals"),
        # Performance
        "model_used": results.get("model_name"),
        "inference_time": results.get("inference_time"),
        # Text stats
        "char_count": len(input_text),
        "word_count": len(input_text.split()),
    }

    response = client.table("analyses").insert(row).execute()
    inserted = response.data
    if inserted and len(inserted) > 0:
        logger.info(f"Analysis saved with ID: {inserted[0]['id']}")
        return inserted[0]
    return None


@db_safe(default=[])
def get_analysis_history(
    user_id: str = None,
    session_id: str = None,
    limit: int = 20,
    offset: int = 0,
) -> list:
    """Retrieve analysis history for a user or session.

    Args:
        user_id: Authenticated user ID.
        session_id: Anonymous session ID.
        limit: Max results to return.
        offset: Pagination offset.

    Returns:
        List of analysis records (newest first).
    """
    client = _get_client()
    if not client:
        return []

    query = client.table("analyses").select(
        "id, created_at, input_source, original_filename, "
        "perplexity, burstiness, is_ai_generated, ai_probability, "
        "confidence, risk_level, verdict, token_count, char_count, "
        "model_used, inference_time"
    )

    if user_id:
        query = query.eq("user_id", user_id)
    elif session_id:
        query = query.eq("session_id", session_id)
    else:
        return []

    response = query.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return response.data or []


@db_safe(default=None)
def get_analysis_by_id(analysis_id: str) -> dict:
    """Retrieve a single analysis by its ID."""
    client = _get_client()
    if not client:
        return None

    response = (
        client.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .single()
        .execute()
    )
    return response.data


# ────────────────────────────────────────────
# Shareable Reports
# ────────────────────────────────────────────

@db_safe(default=None)
def create_report(analysis_id: str, expires_in_hours: int = 168) -> dict:
    """Create a shareable report link for an analysis.

    Args:
        analysis_id: The analysis to share.
        expires_in_hours: How long the link stays valid (default 7 days).

    Returns:
        Dict with share_token and expiry info.
    """
    client = _get_client()
    if not client:
        return None

    share_token = uuid.uuid4().hex[:12]
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)).isoformat()

    row = {
        "analysis_id": analysis_id,
        "share_token": share_token,
        "expires_at": expires_at,
    }

    response = client.table("reports").insert(row).execute()
    if response.data:
        return {
            "share_token": share_token,
            "expires_at": expires_at,
            "url": f"/report/{share_token}",
        }
    return None


@db_safe(default=None)
def get_report_by_token(share_token: str) -> dict:
    """Retrieve a report and its associated analysis by share token."""
    client = _get_client()
    if not client:
        return None

    response = (
        client.table("reports")
        .select("*, analyses(*)")
        .eq("share_token", share_token)
        .single()
        .execute()
    )

    if not response.data:
        return None

    report = response.data

    # Check expiry
    if report.get("expires_at"):
        expires = datetime.fromisoformat(report["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires:
            return None  # Expired

    # Increment view count
    try:
        client.table("reports").update(
            {"view_count": (report.get("view_count", 0) or 0) + 1}
        ).eq("id", report["id"]).execute()
    except Exception:
        pass  # Non-critical

    return report


# ────────────────────────────────────────────
# Usage Tracking & Rate Limiting
# ────────────────────────────────────────────

@db_safe(default=None)
def log_usage(ip_address: str, endpoint: str, response_time: float = None, user_id: str = None):
    """Log an API usage event for analytics and rate limiting."""
    client = _get_client()
    if not client:
        return None

    row = {
        "ip_address": ip_address,
        "endpoint": endpoint,
        "response_time": response_time,
        "user_id": user_id,
    }

    client.table("usage_stats").insert(row).execute()


@db_safe(default=False)
def is_rate_limited(ip_address: str, max_requests: int = 30, window_minutes: int = 60) -> bool:
    """Check if an IP has exceeded the rate limit.

    Args:
        ip_address: The client IP.
        max_requests: Max requests allowed in the window.
        window_minutes: Time window in minutes.

    Returns:
        True if rate-limited, False otherwise.
    """
    client = _get_client()
    if not client:
        return False  # Fail open if DB is down

    since = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

    response = (
        client.table("usage_stats")
        .select("id", count="exact")
        .eq("ip_address", ip_address)
        .gte("created_at", since)
        .execute()
    )

    count = response.count if response.count is not None else 0
    return count >= max_requests


# ────────────────────────────────────────────
# Dashboard / Aggregate Stats
# ────────────────────────────────────────────

@db_safe(default={})
def get_global_stats() -> dict:
    """Get aggregate statistics for the dashboard."""
    client = _get_client()
    if not client:
        return {}

    # Total analyses
    total_resp = client.table("analyses").select("id", count="exact").execute()
    total_scans = total_resp.count or 0

    # AI detected count
    ai_resp = (
        client.table("analyses")
        .select("id", count="exact")
        .eq("is_ai_generated", True)
        .execute()
    )
    ai_count = ai_resp.count or 0

    # Today's scans
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).isoformat()
    today_resp = (
        client.table("analyses")
        .select("id", count="exact")
        .gte("created_at", today)
        .execute()
    )
    today_count = today_resp.count or 0

    return {
        "total_scans": total_scans,
        "ai_detected_count": ai_count,
        "human_detected_count": total_scans - ai_count,
        "today_scans": today_count,
        "ai_detection_rate": round(ai_count / max(total_scans, 1) * 100, 1),
    }
