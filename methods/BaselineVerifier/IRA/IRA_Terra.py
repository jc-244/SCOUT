import json
import os
import threading
import time
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall, Function)

from .IRA_Qwen import IRAQwen, MAX_TOKENS


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), ".env"))
BEDROCK_BASE_URL = os.getenv("BEDROCK_BASE_URL")
BEDROCK_PROJECT = os.getenv("BEDROCK_PROJECT")

_client = None
_client_born = 0.0
_client_lock = threading.Lock()

PRICING = {
    "openai.gpt-5.6-sol":   (5.50, 0.55,  33.00),
    "openai.gpt-5.6-terra": (2.75, 0.275, 16.50),
    "openai.gpt-5.6-luna":  (1.10, 0.11,   6.60),
    "openai.gpt-5.5":       (5.50, 0.55,  33.00),
    "openai.gpt-5.4":       (2.75, 0.275, 16.50),
}


def price_usage(rows, model):
    rate = PRICING.get(model)
    if not rate:
        return None
    p_in, p_cached, p_out = rate
    tin = sum(r.get("input_tokens") or 0 for r in rows)
    tcached = sum(r.get("cached_tokens") or 0 for r in rows)
    tout = sum(r.get("output_tokens") or 0 for r in rows)
    return {"model": model, "calls": len(rows),
            "input_tokens": tin, "cached_tokens": tcached, "output_tokens": tout,
            "reasoning_tokens": sum(r.get("reasoning_tokens") or 0 for r in rows),
            "usd": round(((tin - tcached) * p_in + tcached * p_cached + tout * p_out) / 1e6, 6)}


def bedrock_client():
    global _client, _client_born
    now = time.time()
    if _client is None or now - _client_born > 11 * 3600:
        missing = [k for k, v in (("BEDROCK_BASE_URL", BEDROCK_BASE_URL),
                                  ("BEDROCK_PROJECT", BEDROCK_PROJECT)) if not v]
        if missing:
            raise RuntimeError(f"missing in .env: {', '.join(missing)}")
        from aws_bedrock_token_generator import provide_token
        with _client_lock:
            if _client is None or now - _client_born > 11 * 3600:
                _client = OpenAI(api_key=provide_token(region=os.getenv("AWS_REGION", "us-east-1")),
                                 base_url=BEDROCK_BASE_URL, project=BEDROCK_PROJECT)
                _client_born = time.time()
    return _client

MODEL = "openai.gpt-5.6-terra"
TEMPERATURE = 1.0
TIMEOUT = 180
ATTEMPTS = 5


def _content_items(content):
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    out = []
    for part in content or []:
        t = part.get("type")
        if t == "text":
            out.append({"type": "input_text", "text": part.get("text", "")})
        elif t == "image_url":
            url = part.get("image_url")
            out.append({"type": "input_image",
                        "image_url": url.get("url") if isinstance(url, dict) else url})
        else:
            raise ValueError(f"unsupported chat content part: {t!r}")
    return out


def to_items(messages):
    items = []
    for m in messages:
        role = m.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"],
                          "output": str(m.get("content") or "")})
        elif role == "assistant":
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                items.append({"role": "assistant",
                              "content": [{"type": "output_text", "text": text}]})
            for tc in m.get("tool_calls") or []:
                items.append({"type": "function_call", "call_id": tc["id"],
                              "name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]})
        elif role in ("system", "user"):
            items.append({"role": role, "content": _content_items(m.get("content"))})
        else:
            raise ValueError(f"unsupported chat role: {role!r}")
    return items


def flat_tools(tools):
    out = []
    for t in tools or []:
        f = t["function"]
        out.append({"type": "function", "name": f["name"],
                    "description": f.get("description", ""),
                    "parameters": f.get("parameters")})
    return out


def to_chat_reply(r):
    text, calls = "", []
    for it in (r.output or []):
        t = getattr(it, "type", None)
        if t == "message":
            for c in (getattr(it, "content", None) or []):
                if getattr(c, "type", None) == "output_text":
                    text += getattr(c, "text", "") or ""
        elif t == "function_call":
            calls.append(ChatCompletionMessageFunctionToolCall(
                id=it.call_id, type="function",
                function=Function(name=it.name, arguments=it.arguments or "{}")))
    msg = ChatCompletionMessage(role="assistant", content=text, tool_calls=calls or None)
    if getattr(r, "status", None) == "incomplete":
        finish = "length"
    else:
        finish = "tool_calls" if calls else "stop"
    return msg, finish


def usage_row(r):
    u = getattr(r, "usage", None)
    if u is None:
        return None
    det_in = getattr(u, "input_tokens_details", None)
    det_out = getattr(u, "output_tokens_details", None)
    return {
        "input_tokens": getattr(u, "input_tokens", None),
        "cached_tokens": getattr(det_in, "cached_tokens", 0) if det_in else 0,
        "cache_write_tokens": getattr(det_in, "cache_write_tokens", 0) if det_in else 0,
        "output_tokens": getattr(u, "output_tokens", None),
        "reasoning_tokens": getattr(det_out, "reasoning_tokens", 0) if det_out else 0,
    }


class _Completions:

    def __init__(self, usage_log, responses_create=None):
        self._usage_log = usage_log
        self._create = responses_create or (lambda **kw: bedrock_client().responses.create(**kw))
        self.raw_path = None
        self._n = 0

    def create(self, model, messages, tools=None, **_ignored):
        kw = dict(model=model, input=to_items(messages),
                  max_output_tokens=int(os.getenv("VERIF_MAX_TOKENS", str(MAX_TOKENS))),
                  timeout=TIMEOUT,
                  reasoning={"effort": os.getenv("VERIF_REASONING_EFFORT", "xhigh")},
                  temperature=TEMPERATURE)
        if tools:
            kw["tools"] = flat_tools(tools)
        r = self._respond(kw)
        self._n += 1
        row = usage_row(r)
        if row is not None:
            self._usage_log.append(row)
        msg, finish = to_chat_reply(r)
        self._log_raw(r, row, finish)
        usage = None if row is None else SimpleNamespace(
            prompt_tokens=row["input_tokens"], completion_tokens=row["output_tokens"],
            prompt_tokens_details=SimpleNamespace(cached_tokens=row["cached_tokens"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                               usage=usage)

    def _respond(self, kw):
        last = None
        for attempt in range(ATTEMPTS):
            try:
                r = self._create(**kw)
                err = getattr(getattr(r, "error", None), "message", None)
                if err:
                    raise RuntimeError(f"Responses error: {err}")
                if getattr(r, "status", None) == "failed":
                    raise RuntimeError("Responses request failed (status=failed)")
                return r
            except Exception as e:
                last = e
                print(f"[IRATerra] API error (attempt {attempt + 1}/{ATTEMPTS}): {e}")
                if attempt == ATTEMPTS - 1:
                    raise
                time.sleep(min(5, (attempt + 1) * 2))
        raise RuntimeError(f"terra API failed: {last}")

    def _log_raw(self, r, row, finish):
        if not self.raw_path:
            return
        try:
            out = [o.model_dump(exclude_none=True) for o in (r.output or [])]
            with open(self.raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"call": self._n, "finish_reason": finish,
                                    "status": getattr(r, "status", None),
                                    "output": out, "usage": row}, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[IRATerra] failed to append {self.raw_path}: {e}")


class TerraChatShim:

    def __init__(self, responses_create=None):
        self.usage_log = []
        self.base_url = BEDROCK_BASE_URL
        self.api_key = "EMPTY"
        self.chat = SimpleNamespace(completions=_Completions(self.usage_log, responses_create))


class IRATerra(IRAQwen):

    def __init__(self, env, app="os", model=None, max_images=5, max_steps=30, client=None):
        shim = client or TerraChatShim()
        super().__init__(env, app=app, model=model or os.getenv("VERIF_MODEL", MODEL),
                         max_images=max_images, max_steps=max_steps, client=shim)
        self.usage_log = shim.usage_log

    @property
    def cost(self):
        return price_usage(self.usage_log, self.model_id)
