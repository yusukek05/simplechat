# -*- coding: utf-8 -*-
"""
Lambda handler for simplechat – custom‑model edition (v2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Calls a self‑hosted FastAPI `/generate` endpoint instead of Amazon Bedrock.
* Keeps the original Bedrock logic (commented‑out) for easy rollback.
* Adds CORS/Content‑Type headers & payload shape **identical** to the original
  implementation so that the front‑end stops throwing the generic
  "Network Error" (axios CORS failure).

Required environment variables
------------------------------
LLM_API_URL      Public URL for POST /generate (e.g. https://xxx.ngrok.app/generate)
LLM_API_TIMEOUT  Seconds to wait for the HTTP call (optional, default 60)
"""

import json
import os
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# (Bedrock imports kept but disabled)
# ---------------------------------------------------------------------------
# import boto3
# from botocore.exceptions import ClientError
# bedrock_client = boto3.client("bedrock-runtime")
# MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

LLM_API_URL: str = os.environ.get("LLM_API_URL", "https://YOUR_NGROK_URL.ngrok-free.app/generate")
TIMEOUT: int = int(os.environ.get("LLM_API_TIMEOUT", "60"))

# Common headers required by the SPA (matches the original implementation)
HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_prompt(messages: List[Dict[str, str]], latest_user_msg: str) -> str:
    """Convert conversation list → single prompt string for the LLM API."""
    prompt_lines: List[str] = ["## 会話履歴"]
    for m in messages:
        prompt_lines.append(f"{m['role']}: {m['content']}")
    prompt_lines.append(f"user: {latest_user_msg}")
    prompt_lines.append("assistant: ")
    return "\n".join(prompt_lines)


def _call_llm(prompt: str) -> str:
    """POST the prompt to external FastAPI and return the generated text."""
    req_body = json.dumps(
        {
            "prompt": prompt,
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        LLM_API_URL,
        data=req_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as resp:
            rsp_json = json.loads(resp.read().decode("utf-8"))
            return rsp_json["generated_text"]
    except urllib.error.HTTPError as http_err:
        err_detail = http_err.read().decode("utf-8", errors="ignore")
        LOGGER.error("LLM_API HTTPError %s — %s", http_err.code, err_detail)
        raise
    except Exception:
        LOGGER.exception("Unexpected error calling LLM_API")
        raise


# ---------------------------------------------------------------------------
# Lambda entry point
# ---------------------------------------------------------------------------

def lambda_handler(event: Dict[str, Any], context):  # noqa: D401
    """Main Lambda handler (API Gateway proxy integration)."""
    LOGGER.info("event=%s", event)

    # 1) Parse request body --------------------------------------------------
    try:
        body = event.get("body", event)  # direct invoke vs API‑Gw
        if isinstance(body, str):
            body = json.loads(body)

        user_message: str = body["message"]
        conversation_history: List[Dict[str, str]] = body.get("conversationHistory", [])
    except (KeyError, ValueError, TypeError):
        LOGGER.exception("Bad request format")
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({"success": False, "error": "bad request"}),
        }

    # 2) Build prompt --------------------------------------------------------
    prompt = _build_prompt(conversation_history, user_message)

    # 3) Call external LLM ---------------------------------------------------
    try:
        assistant_response = _call_llm(prompt)
    except Exception as call_err:
        return {
            "statusCode": 502,
            "headers": HEADERS,
            "body": json.dumps({"success": False, "error": str(call_err)}),
        }

    # 4) Assemble new conversation history ----------------------------------
    new_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]

    # 5) Return payload identical to original spec --------------------------
    return {
        "statusCode": 200,
        "headers": HEADERS,
        "body": json.dumps(
            {
                "success": True,
                "response": assistant_response,
                "conversationHistory": new_history,
            },
            ensure_ascii=False,
        ),
    }

# ---------------------------------------------------------------------------
# Original Bedrock code (kept for reference / rollback)
# ---------------------------------------------------------------------------
"""
<original Bedrock block unchanged>
"""

# End of file
