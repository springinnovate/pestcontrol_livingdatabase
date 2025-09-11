"""OpenAI llm tools for each of the trait serarch pipeline tools."""

import json
import re
import time
from typing import Dict, Optional, Tuple, List

from sqlalchemy import select
from sqlalchemy.orm import Session
import tiktoken

from openai import OpenAI
from models import Content

SYSTEM_PROMPT = """
You are a careful classifier. Decide if PAGE_TEXT is valid human-readable content or an error/placeholder.
Definitions:
- VALID_CONTENT: substantive, human-readable text (articles, posts, reports, documentation) with sentences that convey information beyond generic boilerplate.
- INVALID_ERROR: error messages, blocks, placeholders, login/captcha walls, redirects, empty shells, network errors, or pages telling the user to enable JavaScript or similar.

Return a compact JSON object with exactly these keys:
{
  "is_valid": true|false,
  "label": "valid" | "error",
  "error_type": "none" | "http_error" | "bot_block" | "captcha" | "javascript_required" | "timeout" | "empty_or_boilerplate" | "login_wall" | "other",
  "reason": "short justification (<= 20 words)"
}

Rules:
- Prefer false if the text appears to be an error or barrier.
- If mixed content, decide based on whether a typical downstream NLP pipeline could extract meaningful content from it.
- Keep output strictly valid JSON. Do not include extra keys.
""".strip()

QUESTION_PROMPT = """
Classify the provided PAGE_TEXT per the schema. Be decisive. Output only the JSON object.
""".strip()

ERROR_PATTERNS = [
    "enable javascript",
    "you need to enable javascript",
    "please enable javascript",
    "request failed",
    "failed to fetch",
    "network error",
    "timeout",
    "timed out",
    "connection timed out",
    "403 forbidden",
    "forbidden",
    "401 unauthorized",
    "404 not found",
    "page not found",
    "not found",
    "access denied",
    "access is denied",
    "permission denied",
    "too many requests",
    "rate limit",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "captcha",
    "are you a robot",
    "bot detection",
    "bot detected",
    "cloudflare",
    "akamai",
    "imperva",
    "click here if you are not redirected",
    "sign in to continue",
    "login to continue",
    "please log in",
    "error:",
    "an error occurred",
    "internal server error",
    "we've detected unusual activity from your network.",
]


def guess_if_error_text(text: str) -> Optional[bool]:
    if not text or not text.strip():
        return True
    lower = text.lower()
    if any(pat in lower for pat in ERROR_PATTERNS):
        return True
    compact = re.sub(r"\s+", " ", lower).strip()
    if len(compact) < 40:
        # if after cleaning whitespace, if it's less than 40 characters
        # it's probably invalid
        return True
    return None  # unknown -> defer to LLM


def evaluate_validity_with_llm(
    page_text: str, logger
) -> Tuple[Optional[bool], str, str]:
    resp = apply_llm(QUESTION_PROMPT, page_text, logger)
    # resp is expected to be a dict due to response_format='json_object'
    try:
        is_valid = resp.get("is_valid", None)
        if isinstance(is_valid, bool):
            label = resp.get("label", "valid" if is_valid else "error")
            err_type = resp.get("error_type", "none" if is_valid else "other")
            return is_valid, label, err_type
    except Exception:
        pass
    return None, "error", "other"


def backfill_is_valid(
    session: Session,
    logger,
    batch_size: int = 100,
    throttle_sec: float = 0.0,
) -> None:
    q = select(Content).where(Content.is_valid.is_(None))
    result = session.execute(q)
    pending = []
    for content_obj in result.scalars():
        decided = quick_error_heuristic(content_obj.text)
        if decided is None:
            is_valid, label, err_type = evaluate_validity_with_llm(
                content_obj.text, logger
            )
            if is_valid is None:
                # Could not parse/decide; skip without update
                continue
            decided = is_valid
            # Optional: store diagnostics elsewhere if desired
        content_obj.is_valid = 1 if decided else 0
        pending.append(content_obj.id)

        if len(pending) >= batch_size:
            session.commit()
            pending.clear()
            if throttle_sec > 0:
                time.sleep(throttle_sec)

    if pending:
        session.commit()


def create_openai_context():
    """Helper to create openai context."""
    openai_client = OpenAI()
    openai_context = {
        "client": openai_client,
    }
    return openai_context


def truncate_to_token_limit(
    text: str, model: str = "gpt-4o-mini", max_tokens: int = 100000
) -> str:
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return enc.decode(tokens)


def apply_llm(question: str, page_text: str, logger) -> List[Dict[str, str]]:
    """Invoke the question and context on the `SYSTEM_PROMPT`."""
    user_payload = (
        "QUESTION:\n"
        f"{question.strip()}\n\n"
        "PAGE_TEXT:\n"
        f"{page_text.strip()}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    args = {
        "model": "gpt-4o-mini",
        "response_format": {
            "type": "json_object"
        },  # if supported in your SDK/runtime
        "messages": messages,
    }
    openai_context = create_openai_context()
    resp = openai_context["client"].chat.completions.create(**args)
    try:
        content = json.loads(resp.choices[0].message.content)
    except json.decoder.JSONDecodeError:
        logger.exception(f"error with response: {resp}")
        raise
    return content
