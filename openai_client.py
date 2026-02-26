import json
import time
from typing import Optional, Type

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError


def _should_send_temperature(model: str, temp: float) -> bool:
    # Some models (for example current gpt-5 variants) only accept default temperature.
    if model.lower().startswith("gpt-5"):
        return False
    return True


def _api_error_message(err: Exception, attempt: int, retries: int) -> str:
    prefix = f"(attempt {attempt + 1}/{retries + 1})"
    if isinstance(err, APITimeoutError):
        return f"{prefix} API timeout: request exceeded timeout window."
    if isinstance(err, RateLimitError):
        return f"{prefix} Rate limited by API (429)."
    if isinstance(err, APIConnectionError):
        return f"{prefix} API connection failed. Check network access."
    if isinstance(err, APIStatusError):
        code = getattr(err, "status_code", "unknown")
        details = str(err).strip()
        return f"{prefix} API status error {code}: {details}"
    if isinstance(err, ValidationError):
        return f"{prefix} Schema validation error: {err}"
    if isinstance(err, json.JSONDecodeError):
        return f"{prefix} Invalid JSON returned by model: {err}"
    return f"{prefix} Unexpected error: {err}"


def api_structured(api_key: str, model: str, temp: float, schema_model: Type[BaseModel], sys: str, usr: str):
    client = OpenAI(api_key=api_key, timeout=20.0)
    schema = schema_model.model_json_schema()
    raw: Optional[str] = None
    retries = 1
    last_err = None
    send_temp = _should_send_temperature(model, temp)
    for attempt in range(retries + 1):
        try:
            if hasattr(client, "responses"):
                req = {
                    "model": model,
                    "input": [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                    "text": {"format": {"type": "json_schema", "name": schema_model.__name__, "schema": schema, "strict": True}},
                }
                if send_temp:
                    req["temperature"] = temp
                r = client.responses.create(
                    **req,
                )
                raw = getattr(r, "output_text", None) or ""
                if not raw:
                    for o in getattr(r, "output", []) or []:
                        for c in getattr(o, "content", []) or []:
                            if hasattr(c, "text"):
                                raw += c.text
            else:
                req = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": f"{sys}\nReturn only valid JSON for this schema: {json.dumps(schema, ensure_ascii=False)}"},
                        {"role": "user", "content": usr},
                    ],
                    "response_format": {"type": "json_object"},
                }
                if send_temp:
                    req["temperature"] = temp
                r = client.chat.completions.create(**req)
                choices = getattr(r, "choices", []) or []
                first = choices[0] if choices else None
                message = getattr(first, "message", None)
                raw = (getattr(message, "content", None) or "").strip()
            parsed = schema_model.model_validate(json.loads(raw))
            return parsed, raw, None
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as e:
            if isinstance(e, APIStatusError):
                msg = str(e).lower()
                if "temperature" in msg and "default (1)" in msg and send_temp:
                    # Retry once without temperature when model rejects custom values.
                    send_temp = False
                    continue
            last_err = _api_error_message(e, attempt, retries)
            status_code = getattr(e, "status_code", None)
            retryable_status = status_code is None or int(status_code) >= 500 or int(status_code) == 429
            if attempt < retries and retryable_status:
                time.sleep(1.2 * (attempt + 1))
                continue
            break
        except (json.JSONDecodeError, ValidationError) as e:
            return None, raw, _api_error_message(e, attempt, retries)
        except Exception as e:
            return None, raw, _api_error_message(e, attempt, retries)
    return None, raw, last_err or "Unknown API failure"
