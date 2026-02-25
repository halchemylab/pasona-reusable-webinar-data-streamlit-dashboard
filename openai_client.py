import json
import time
from typing import Optional, Type

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError


def _api_error_message(err: Exception, attempt: int, retries: int) -> str:
    prefix = f"(attempt {attempt + 1}/{retries + 1})"
    if isinstance(err, APITimeoutError):
        return f"{prefix} API timeout: request exceeded timeout window."
    if isinstance(err, RateLimitError):
        return f"{prefix} Rate limited by API (429)."
    if isinstance(err, APIConnectionError):
        return f"{prefix} API connection failed. Check network access."
    if isinstance(err, APIStatusError):
        return f"{prefix} API status error {getattr(err, 'status_code', 'unknown')}."
    if isinstance(err, ValidationError):
        return f"{prefix} Schema validation error: {err}"
    if isinstance(err, json.JSONDecodeError):
        return f"{prefix} Invalid JSON returned by model: {err}"
    return f"{prefix} Unexpected error: {err}"


def api_structured(api_key: str, model: str, temp: float, schema_model: Type[BaseModel], sys: str, usr: str):
    client = OpenAI(api_key=api_key, timeout=45.0)
    schema = schema_model.model_json_schema()
    raw: Optional[str] = None
    retries = 2
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.responses.create(
                model=model,
                temperature=temp,
                input=[{"role": "system", "content": sys}, {"role": "user", "content": usr}],
                text={"format": {"type": "json_schema", "name": schema_model.__name__, "schema": schema, "strict": True}},
            )
            raw = getattr(r, "output_text", None) or ""
            if not raw:
                for o in getattr(r, "output", []) or []:
                    for c in getattr(o, "content", []) or []:
                        if hasattr(c, "text"):
                            raw += c.text
            parsed = schema_model.model_validate(json.loads(raw))
            return parsed, raw, None
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as e:
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
