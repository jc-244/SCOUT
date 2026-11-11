import base64
import io
import json
import logging
import os
import threading
import time
from collections import namedtuple

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .VAGEN_Terra import TerraEvalAgent

logger = logging.getLogger("desktopenv.agent")

MODEL = os.getenv("VERIF_MODEL", "moonshotai.kimi-k2.5")
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), ".env"))
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL")
BEDROCK_PROJECT = os.getenv("BEDROCK_PROJECT")

TEMPERATURE = 1.0
TOP_P = 0.95
PRESENCE_PENALTY = 1.5
TOP_K = 20
TIMEOUT = 180

_client = None
_client_born = 0.0
_client_lock = threading.Lock()


def bedrock_kimi_client():
    global _client, _client_born
    now = time.time()
    if _client is None or now - _client_born > 11 * 3600:
        missing = [k for k, v in (("KIMI_BASE_URL", KIMI_BASE_URL),
                                  ("BEDROCK_PROJECT", BEDROCK_PROJECT)) if not v]
        if missing:
            raise RuntimeError(f"missing in .env: {', '.join(missing)}")
        from aws_bedrock_token_generator import provide_token
        with _client_lock:
            if _client is None or now - _client_born > 11 * 3600:
                _client = OpenAI(api_key=provide_token(region=os.getenv("AWS_REGION", "us-east-1")),
                                 base_url=KIMI_BASE_URL, project=BEDROCK_PROJECT)
                _client_born = time.time()
    return _client


Call = namedtuple("Call", "call_id name arguments")


def _jpeg_url(data_url):
    try:
        head, b64 = data_url.split(",", 1)
        if not head.startswith("data:image/") or ";base64" not in head:
            return data_url
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return data_url


def content_to_chat(content):
    if isinstance(content, str):
        return content
    parts = []
    for c in content or []:
        t = c.get("type")
        if t in ("input_text", "output_text"):
            parts.append({"type": "text", "text": c.get("text", "")})
        elif t == "input_image":
            parts.append({"type": "image_url", "image_url": {"url": _jpeg_url(c.get("image_url", ""))}})
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def _sound_arguments(raw):
    raw = raw or "{}"
    try:
        json.loads(raw)
        return raw
    except Exception:
        return "{}"


def sanitize_history(msgs):
    out = []
    pending = []
    for m in msgs:
        if m.get("role") == "tool":
            out.append(m)
            if m.get("tool_call_id") in pending:
                pending.remove(m["tool_call_id"])
            continue
        for cid in pending:
            out.append({"role": "tool", "tool_call_id": cid, "content": "[tool execution aborted]"})
        pending = [c["id"] for c in (m.get("tool_calls") or [])]
        out.append(m)
    for cid in pending:
        out.append({"role": "tool", "tool_call_id": cid, "content": "[tool execution aborted]"})
    return out


def items_to_messages(items):
    msgs = []
    for it in items:
        kind = it.get("type")
        if kind == "function_call":
            call = {"id": it["call_id"], "type": "function",
                    "function": {"name": it["name"], "arguments": _sound_arguments(it.get("arguments"))}}
            if msgs and msgs[-1].get("role") == "assistant":
                msgs[-1].setdefault("tool_calls", []).append(call)
            else:
                msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            msgs.append({"role": "tool", "tool_call_id": it["call_id"],
                         "content": str(it.get("output", ""))})
        else:
            msgs.append({"role": it["role"], "content": content_to_chat(it.get("content"))})
    return msgs


IMAGE_WINDOW = 10
_IMG_NOTE = "[System Note: image removed to save context]"


def cap_wire_images(msgs, keep=IMAGE_WINDOW):
    first_user = None
    slots = []
    for i, m in enumerate(msgs):
        c = m.get("content")
        if not isinstance(c, list):
            continue
        if first_user is None and m.get("role") == "user":
            first_user = i
        for j, part in enumerate(c):
            if isinstance(part, dict) and part.get("type") == "image_url":
                slots.append((i, j))
    others = [s for s in slots if s[0] != first_user]
    drop = set(others[:-keep]) if len(others) > keep else set()
    for i, j in drop:
        msgs[i]["content"][j] = {"type": "text", "text": _IMG_NOTE}
    for i, m in enumerate(msgs):
        c = m.get("content")
        if isinstance(c, list) and c and all(isinstance(x, dict) and x.get("type") == "text" for x in c):
            m["content"] = "\n".join(x["text"] for x in c)
    return msgs


def tools_to_chat(tools):
    return [{"type": "function",
             "function": {"name": t["name"], "description": t.get("description", ""),
                          "parameters": t.get("parameters", {})}}
            for t in (tools or [])]


class KimiEvalAgent(TerraEvalAgent):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("model", MODEL)
        super().__init__(*args, **kwargs)
        self._reasoning_log = []
        self._replies = []
        self._reply_sources = []

    def _respond(self, items, tools):
        messages = cap_wire_images(sanitize_history(items_to_messages(items)))
        chat_tools = tools_to_chat(tools)
        last = None
        for attempt in range(3):
            try:
                kw = dict(model=self.model_name, messages=messages,
                          max_tokens=self.max_tokens, timeout=TIMEOUT,
                          temperature=TEMPERATURE, top_p=TOP_P,
                          presence_penalty=PRESENCE_PENALTY,
                          extra_body={"top_k": TOP_K})
                if chat_tools:
                    kw["tools"] = chat_tools
                r = bedrock_kimi_client().chat.completions.create(**kw)
                msg = r.choices[0].message
                think = (getattr(msg, "reasoning", None)
                         or getattr(msg, "reasoning_content", None) or "")
                if think:
                    self._reasoning_log.append(think)
                text = msg.content or ""
                calls = [Call(tc.id, tc.function.name, tc.function.arguments or "{}")
                         for tc in (msg.tool_calls or [])]
                source = "content"
                if not text.strip() and not calls and think.strip():
                    text, source = think, "reasoning"
                if text.strip():
                    self._replies.append(text.strip())
                    self._reply_sources.append(source)
                u = getattr(r, "usage", None)
                if u is not None:
                    det_in = getattr(u, "prompt_tokens_details", None)
                    det_out = getattr(u, "completion_tokens_details", None)
                    self._usage_log.append({
                        "input_tokens": getattr(u, "prompt_tokens", None),
                        "cached_tokens": getattr(det_in, "cached_tokens", 0) if det_in else 0,
                        "cache_write_tokens": 0,
                        "output_tokens": getattr(u, "completion_tokens", None),
                        "reasoning_tokens": getattr(det_out, "reasoning_tokens", 0) if det_out else 0,
                    })
                return text.strip(), calls
            except Exception as e:
                last = e
                logger.warning(f"kimi API error (attempt {attempt + 1}/3): {e}")
                time.sleep(5)
        raise RuntimeError(f"kimi API failed after 3 attempts: {last}")
