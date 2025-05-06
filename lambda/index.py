import json
import os
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

LLM_API_URL: str = os.environ.get(
    "LLM_API_URL",
    "https://NGROK_URL.ngrok-free.app/generate"  # ← 必ず置き換える
)
TIMEOUT: int = int(os.environ.get("LLM_API_TIMEOUT", "60"))

HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,"
    "X-Api-Key,X-Amz-Security-Token",
    "Access-Control-Allow-Methods": "OPTIONS,POST",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _build_prompt(messages: List[Dict[str, str]], latest_user_msg: str) -> str:
    """Return a newline-concatenated prompt without 'user:' / 'assistant:' labels."""
    text_parts: List[str] = [m["content"] for m in messages]
    text_parts.append(latest_user_msg)
    return "\n".join(text_parts)  # ← 修正ポイント


def _call_llm(prompt: str) -> str:
    """POST the prompt to external FastAPI and return generated text."""
    req_body = json.dumps(
        {
            "prompt": prompt.strip(),
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
    """Main Lambda handler (API Gateway proxy integration)."""
    LOGGER.info("event=%s", event)

    # 1) parse request body ---------------------------------------------------
    try:
        body = event.get("body", event)  # direct invoke vs API-Gw
        if isinstance(body, str):
            body = json.loads(body)

        user_message: str = body["message"]
        conversation_history: List[Dict[str, str]] = body.get(
            "conversationHistory", []
        )
    except (KeyError, ValueError, TypeError):
        LOGGER.exception("Bad request format")
        return {
            "statusCode": 400,
            "headers": HEADERS,
            "body": json.dumps({"success": False, "error": "bad request"}),
        }

    # 2) build prompt ---------------------------------------------------------
    prompt = _build_prompt(conversation_history, user_message)

    # 3) call external LLM ----------------------------------------------------
    try:
        assistant_response = _call_llm(prompt)
    except Exception as call_err:
        return {
            "statusCode": 502,
            "headers": HEADERS,
            "body": json.dumps({"success": False, "error": str(call_err)}),
        }

    # 4) assemble new conversation history -----------------------------------
    new_history = conversation_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_response},
    ]

    # 5) return payload (same shape as original) ------------------------------
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