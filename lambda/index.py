# -*- coding: utf-8 -*-
"""
Lambda handler for simplechat - **custom-model edition**
-------------------------------------------------------
This version calls a *self-hosted* FastAPI inference endpoint (e.g. running on
Google Colab + ngrok) instead of Amazon Bedrock.  The original Bedrock logic is
still present but commented-out so you can quickly switch back if needed.

Required environment variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLM_API_URL      Public URL for POST /generate (default: placeholder)
LLM_API_TIMEOUT  Seconds to wait for the HTTP call (optional, default 60)
# MODEL_ID       (Bedrock) kept for backwards compatibility - not used now

Event payload contract (unchanged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
request  → { "message": str, "conversationHistory": list[ {role, content} ] }
response ← { "assistantResponse": str, "conversationHistory": list[ ... ] }
"""

import json
import os
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any

# --- Bedrock import & client (disabled) -------------------------------------
# import boto3
# from botocore.exceptions import ClientError
# bedrock_client = boto3.client("bedrock-runtime")
# MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

LLM_API_URL: str = os.environ.get("LLM_API_URL", "https://YOUR_NGROK_URL.ngrok-free.app/generate")
TIMEOUT: int = int(os.environ.get("LLM_API_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_prompt(messages: List[Dict[str, str]], latest_user_msg: str) -> str:
    """Flatten conversation into a single prompt string for the LLM-API."""
    prompt_lines: List[str] = ["## 会話履歴"]
    for m in messages:
        prompt_lines.append(f"{m['role']}: {m['content']}")
    prompt_lines.append(f"user: {latest_user_msg}")
    prompt_lines.append("assistant: ")
    return "\n".join(prompt_lines)


def _call_llm(prompt: str) -> str:
    """POST the prompt to FastAPI and return generated text."""
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
        with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
            rsp_json = json.loads(r.read().decode("utf-8"))
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
    """Main Lambda handler (unchanged signature)."""
    LOGGER.info("event=%s", event)

    # --------------------------------------------------
    # 1) Parse request body
    # --------------------------------------------------
    try:
        body = event.get("body", event)  # direct invoke vs API-Gw
        if isinstance(body, str):
            body = json.loads(body)

        user_message: str = body["message"]
        conversation_history: List[Dict[str, str]] = body.get("conversationHistory", [])
    except (KeyError, ValueError, TypeError):
        LOGGER.exception("Bad request format")
        return {"statusCode": 400, "body": "bad request"}

    # --------------------------------------------------
    # 2) Build prompt string for external LLM
    # --------------------------------------------------
    prompt = _build_prompt(conversation_history, user_message)

    # --------------------------------------------------
    # 3) Call external inference endpoint
    # --------------------------------------------------
    try:
        assistant_response = _call_llm(prompt)
    except Exception as e:  # noqa: BLE001
        return {"statusCode": 502, "body": f"LLM backend error: {e}"}

    # --------------------------------------------------
    # 4) Append assistant message to history & return
    # --------------------------------------------------
    new_history = conversation_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": assistant_response}]

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "assistantResponse": assistant_response,
                "conversationHistory": new_history,
            },
            ensure_ascii=False,
        ),
    }

# ---------------------------------------------------------------------------
# Original Bedrock path kept here for reference (commented-out)
# ---------------------------------------------------------------------------
"""
    # ------- Bedrock runtime (original) --------
    try:
        payload = {
            "inputText": user_message,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9,
            },
        }

        response = bedrock_client.invoke_model(
            body=json.dumps(payload).encode("utf-8"),
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        assistant_response = response_body["results"][0]["outputText"]
    except ClientError as ce:
        LOGGER.error("Bedrock error: %s", ce)
        return {"statusCode": 502, "body": str(ce)}
"""

# End of file