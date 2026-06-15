import argparse
import copy
import json
import os
import re
import time
import unicodedata
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


UPSTREAM = "http://127.0.0.1:8080"
HOST = "127.0.0.1"
PORT = 8081
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = None
UPSTREAM_TIMEOUT_ENV = "GEMMA_PROXY_UPSTREAM_TIMEOUT_SECONDS"
UPSTREAM_TIMEOUT_SECONDS = None
UPSTREAM_TIMEOUT_MESSAGE = "Gemma upstream request timed out."
EMPTY_VISIBLE_RESPONSE_MESSAGE = (
    "I did not produce a visible final answer because the local model emitted only internal channel text. "
    "Retry the last request or ask me to continue from the visible tool results."
)
SIMPLE_GREETING_RESPONSE_TEXT = "Hi father."
SIMPLE_GREETING_RE = re.compile(r"^\s*(?:hi|hello|hey|yo|sup|hiya|howdy)[.!?]*\s*$", re.IGNORECASE)
FINAL_CHANNEL_RE = re.compile(r"<\|channel\>\s*final(?:\s*<channel\|>|\s+)", re.IGNORECASE)
CHANNEL_MARKER_START = "<|channel>"
CHANNEL_MARKER_END = "<channel|>"
CHANNEL_MARKER_MAX_BUFFER = 512
CHANNEL_MARKER_LOOKBACK = 128
INTERNAL_TOOL_CALL_PREFIX_RE = re.compile(
    r"^\s*<\|tool_call\>call:codex:[A-Za-z0-9_.:-]+(?:\{[^<>\r\n]*\})?<tool_call\|>\s*",
    re.IGNORECASE,
)
INTERNAL_TOOL_CALL_START = "<|tool_call>"
INTERNAL_TOOL_CALL_END = "<tool_call|>"
INTERNAL_TOOL_CALL_MAX_BUFFER = 512
STREAM_TEXT_KEYS = {"content", "delta", "text"}
PROXY_SANITIZED_KEY_FRAGMENTS = ("error", "tool", "proxy", "details")
PROXY_MODEL_TEXT_KEYS = {"content", "delta", "output_text", "text"}
ACTION_OUTPUT_TYPES = {
    "code_interpreter_call",
    "computer_call",
    "custom_tool_call",
    "file_search_call",
    "function_call",
    "local_shell_call",
    "mcp_call",
    "web_search_call",
}
STREAM_CHANNEL_MARKERS = tuple(
    f"<|channel>{role}{separator}<channel|>"
    for role in ("thought", "final")
    for separator in ("", "\n", "\r\n", " ")
)
VISUAL_SYMBOL_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
)
VISUAL_SYMBOL_CODEPOINTS = {0x200D, 0x20E3, 0xFE0E, 0xFE0F}
ABSOLUTE_ACHIEVABLE_RE = re.compile(r"\b100\s*%\s+ach(?:iev|eiv)e?able\b", re.IGNORECASE)
PRODUCTION_READY_RE = re.compile(r"\bproduction[-\s]?ready\b", re.IGNORECASE)
HIDDEN_THOUGHT_TEXT_RE = re.compile(
    r"^\s*(hidden|secret|analysis|reasoning|scratchpad|chain[ -]?of[ -]?thought|internal)\b",
    re.IGNORECASE,
)
UPPERCASE_EMERGENCY_TERMS = {
    "ALERT",
    "CATASTROPHIC",
    "CRISIS",
    "CRITICAL",
    "DANGER",
    "DANGEROUS",
    "DISASTER",
    "EMERGENCY",
    "FATAL",
    "HARMFUL",
    "HARMFULNESS",
    "PANIC",
    "RISK",
    "SEVERE",
    "THREAT",
    "URGENT",
    "WARNING",
}
UPPERCASE_EMERGENCY_RE = re.compile(
    r"\b(" + "|".join(sorted(UPPERCASE_EMERGENCY_TERMS, key=len, reverse=True)) + r")\b"
)


def proxy_timeout_seconds(env=None):
    env = os.environ if env is None else env
    raw_value = env.get(UPSTREAM_TIMEOUT_ENV)
    if raw_value in {None, ""}:
        return DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_UPSTREAM_TIMEOUT_SECONDS
    return parsed if parsed > 0 else DEFAULT_UPSTREAM_TIMEOUT_SECONDS


UPSTREAM_TIMEOUT_SECONDS = proxy_timeout_seconds()


def clean_internal_tool_call_markers(text):
    return INTERNAL_TOOL_CALL_PREFIX_RE.sub("", text, count=1)


def sanitize_behavior_text(text):
    if not isinstance(text, str) or not text:
        return text
    text = "".join(char for char in text if not is_visual_symbol(char))
    text = ABSOLUTE_ACHIEVABLE_RE.sub("likely achievable with verification", text)
    text = PRODUCTION_READY_RE.sub("ready for review", text)
    return UPPERCASE_EMERGENCY_RE.sub(lambda match: match.group(1).lower(), text)


def is_visual_symbol(char):
    codepoint = ord(char)
    if codepoint in VISUAL_SYMBOL_CODEPOINTS:
        return True
    if any(start <= codepoint <= end for start, end in VISUAL_SYMBOL_RANGES):
        return True
    return unicodedata.category(char) == "So" and codepoint >= 0x2100


def clean_channel_markers(text):
    final_text = text_after_last_final_channel(text)
    if final_text is not None:
        return clean_internal_tool_call_markers(final_text)
    if starts_with_channel_marker(text):
        return ""
    return clean_internal_tool_call_markers(text)


def text_after_last_final_channel(text):
    match = None
    for match in FINAL_CHANNEL_RE.finditer(text):
        pass
    if match is None:
        return None
    return text[match.end():]


def starts_with_channel_marker(text):
    return isinstance(text, str) and text.lstrip().lower().startswith(CHANNEL_MARKER_START)


def starts_with_nonfinal_channel(text):
    if not starts_with_channel_marker(text):
        return False
    candidate = text.lstrip().lower()
    role_fragment = candidate[len(CHANNEL_MARKER_START):].lstrip()
    if not role_fragment:
        return False
    if "final".startswith(role_fragment) or role_fragment.startswith("final"):
        return False
    return True


def channel_marker_tail(text):
    return text[-CHANNEL_MARKER_LOOKBACK:]


def is_internal_tool_call_text(text):
    if not isinstance(text, str) or not text.strip():
        return False
    return INTERNAL_TOOL_CALL_PREFIX_RE.match(text) is not None and clean_internal_tool_call_markers(text) == ""


def is_channel_marker_prefix(text):
    candidate = text.lstrip().lower()
    if not candidate:
        return True
    if CHANNEL_MARKER_START.startswith(candidate):
        return True
    if (
        candidate.startswith(CHANNEL_MARKER_START)
        and CHANNEL_MARKER_END not in candidate
        and len(candidate) <= CHANNEL_MARKER_MAX_BUFFER
    ):
        return True
    return any(marker.startswith(candidate) for marker in STREAM_CHANNEL_MARKERS)


def is_internal_tool_call_prefix(text):
    candidate = text.lstrip().lower()
    if not candidate:
        return True
    if INTERNAL_TOOL_CALL_START.startswith(candidate):
        return True
    return (
        candidate.startswith(INTERNAL_TOOL_CALL_START)
        and INTERNAL_TOOL_CALL_END not in candidate
        and "\n" not in candidate
        and "\r" not in candidate
        and len(candidate) <= INTERNAL_TOOL_CALL_MAX_BUFFER
    )


class StreamChannelCleaner:
    def __init__(self):
        self.buffer = ""
        self.pending = True
        self.suppress_until_final = False
        self.suppressed_nonfinal_channel_text_seen = False

    def clean(self, text):
        if self.suppress_until_final:
            self.buffer += text
            final_text = text_after_last_final_channel(self.buffer)
            if final_text is None:
                self.buffer = channel_marker_tail(self.buffer)
                return ""
            self.buffer = ""
            self.pending = False
            self.suppress_until_final = False
            return clean_internal_tool_call_markers(final_text)

        if not self.pending:
            return clean_channel_markers(text)

        self.buffer += text
        final_text = text_after_last_final_channel(self.buffer)
        if final_text is not None:
            self.buffer = ""
            self.pending = False
            return clean_internal_tool_call_markers(final_text)

        if starts_with_nonfinal_channel(self.buffer):
            self.buffer = channel_marker_tail(self.buffer)
            self.suppress_until_final = True
            self.suppressed_nonfinal_channel_text_seen = True
            return ""

        if is_channel_marker_prefix(self.buffer) or is_internal_tool_call_prefix(self.buffer):
            return ""

        cleaned = clean_channel_markers(self.buffer)
        if cleaned != self.buffer:
            self.buffer = ""
            self.pending = False
            return cleaned

        self.pending = False
        result = self.buffer
        self.buffer = ""
        return result


def clean_response_payload(payload):
    cleaned = copy.deepcopy(payload)
    recovered_visible_text = recover_thought_only_visible_response_text(payload)
    clean_response_model_text(cleaned)
    if (
        recovered_visible_text
        and not any(text.strip() for text in response_model_texts(cleaned))
    ):
        apply_visible_response_text(cleaned, recovered_visible_text)
    if has_action_output(cleaned):
        prune_empty_message_output_items(cleaned)
    elif needs_empty_visible_response_fallback(payload, cleaned):
        apply_empty_visible_response_fallback(cleaned)
    return clean_json_strings(cleaned)


def clean_response_model_text(payload):
    if not isinstance(payload, dict):
        return payload
    if isinstance(payload.get("output_text"), str):
        payload["output_text"] = clean_channel_markers(payload["output_text"])
    for item in response_output_items(payload):
        clean_response_output_item_text(item)
    return payload


def clean_response_output_item_text(item):
    if not is_message_output_item(item):
        return item
    for content in item.get("content", []):
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            content["text"] = clean_channel_markers(content["text"])
    return item


def response_model_texts(payload):
    if not isinstance(payload, dict):
        return []
    texts = []
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        texts.append(output_text)
    for item in response_output_items(payload):
        texts.extend(response_message_texts(item))
    return texts


def needs_empty_visible_response_fallback(original, cleaned):
    original_texts = [text for text in response_model_texts(original) if text.strip()]
    if has_action_output(original):
        return False
    if any(text.strip() for text in response_model_texts(cleaned)):
        return False
    if not original_texts:
        return is_completed_response_payload(original)
    return any(
        is_suppressed_nonfinal_channel_text(text) or is_internal_tool_call_text(text)
        for text in original_texts
    )


def is_completed_response_payload(value):
    if not isinstance(value, dict):
        return False
    response = value.get("response")
    if isinstance(response, dict):
        return is_completed_response_payload(response)
    if value.get("status") != "completed":
        return False
    return value.get("object") == "response" or "output" in value or "output_text" in value


def recover_thought_only_visible_response_text(payload):
    candidates = [
        recovered
        for recovered in (
            recover_thought_only_visible_text(text)
            for text in response_model_texts(payload)
        )
        if recovered
    ]
    if not candidates:
        return None
    return candidates[0]


def recover_thought_only_visible_text(text):
    if not isinstance(text, str) or not starts_with_nonfinal_channel(text):
        return None
    if text_after_last_final_channel(text) is not None:
        return None
    marker_index = text.lower().find(CHANNEL_MARKER_END)
    if marker_index < 0:
        return None
    candidate = clean_internal_tool_call_markers(text[marker_index + len(CHANNEL_MARKER_END):]).strip()
    if not candidate or HIDDEN_THOUGHT_TEXT_RE.search(candidate):
        return None
    return candidate


def response_output_items(payload):
    if not isinstance(payload, dict):
        return []
    output = payload.get("output")
    if not isinstance(output, list):
        return []
    return [item for item in output if isinstance(item, dict)]


def response_message_texts(item):
    if not is_message_output_item(item):
        return []
    content_list = item.get("content")
    if not isinstance(content_list, list):
        return []
    return [
        content["text"]
        for content in content_list
        if isinstance(content, dict) and isinstance(content.get("text"), str)
    ]


def has_action_output(value):
    if not isinstance(value, dict):
        return False

    if any(is_action_output_item(item) for item in response_output_items(value)):
        return True
    item = value.get("item")
    if isinstance(item, dict) and is_action_output_item(item):
        return True
    response = value.get("response")
    if isinstance(response, dict) and has_action_output(response):
        return True
    tool_calls = value.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        return True
    function_call = value.get("function_call")
    if isinstance(function_call, dict) and function_call:
        return True
    return False


def is_action_output_item(item):
    return isinstance(item, dict) and is_action_output_type(item.get("type"))


def is_message_output_item(item):
    return (
        isinstance(item, dict)
        and not is_action_output_item(item)
        and (item.get("type") == "message" or isinstance(item.get("content"), list))
    )


def is_action_output_type(item_type):
    if not isinstance(item_type, str):
        return False
    normalized = item_type.strip().lower().replace("-", "_")
    return (
        normalized in ACTION_OUTPUT_TYPES
        or normalized.endswith("_tool_call")
        or normalized.endswith("_call")
    )


def is_suppressed_nonfinal_channel_text(text):
    if not isinstance(text, str) or not text.strip():
        return False
    return starts_with_nonfinal_channel(text) and clean_channel_markers(text).strip() == ""


def apply_empty_visible_response_fallback(value):
    if not isinstance(value, dict):
        return value
    response = value.get("response")
    if isinstance(response, dict):
        apply_empty_visible_response_fallback(response)
        return value
    if not isinstance(value.get("output_text"), str) or not value["output_text"].strip():
        value["output_text"] = EMPTY_VISIBLE_RESPONSE_MESSAGE
    output = value.setdefault("output", [])
    if isinstance(output, list) and not any(is_message_output_item(item) for item in output):
        output.append(build_message_output_item(EMPTY_VISIBLE_RESPONSE_MESSAGE))
    for item in response_output_items(value):
        if not is_message_output_item(item):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str) and not content["text"].strip():
                content["text"] = EMPTY_VISIBLE_RESPONSE_MESSAGE
    return value


def apply_visible_response_text(value, text):
    if not isinstance(value, dict):
        return value
    response = value.get("response")
    if isinstance(response, dict):
        apply_visible_response_text(response, text)
        return value
    value["output_text"] = text
    output = value.setdefault("output", [])
    if isinstance(output, list) and not any(is_message_output_item(item) for item in output):
        output.append(build_message_output_item(text))
    for item in response_output_items(value):
        if not is_message_output_item(item):
            continue
        content_list = item.setdefault("content", [])
        if not content_list:
            content_list.append(
                {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": text,
                }
            )
        for content in content_list:
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                content["text"] = text
    return value


def build_message_output_item(text):
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": text,
            }
        ],
    }


def prune_empty_message_output_items(payload):
    output = payload.get("output") if isinstance(payload, dict) else None
    if not isinstance(output, list):
        return payload
    payload["output"] = [
        item
        for item in output
        if not is_empty_message_output_item(item)
    ]
    return payload


def is_empty_message_output_item(item):
    if not is_message_output_item(item):
        return False
    content_list = item.get("content")
    if not isinstance(content_list, list) or not content_list:
        return True
    for content in content_list:
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return False
            if has_meaningful_non_text_message_content(content):
                return False
        elif content:
            return False
    return True


def has_meaningful_non_text_message_content(content):
    ignored_keys = {"type", "text", "annotations"}
    return any(
        key not in ignored_keys and bool(value)
        for key, value in content.items()
    )


def should_sanitize_proxy_metadata(path):
    normalized = [str(item).lower().replace("-", "_") for item in path]
    if not normalized:
        return False
    if normalized[-1] in PROXY_MODEL_TEXT_KEYS and not any(
        item in {"error", "proxy_message", "tool_output"} or "tool" in item
        for item in normalized[:-1]
    ):
        return False
    return any(
        fragment in item
        for item in normalized
        for fragment in PROXY_SANITIZED_KEY_FRAGMENTS
    )


def clean_json_strings(value, path=()):
    if isinstance(value, str):
        if should_sanitize_proxy_metadata(path):
            return sanitize_behavior_text(clean_channel_markers(value))
        return value
    if isinstance(value, list):
        return [clean_json_strings(item, (*path, index)) for index, item in enumerate(value)]
    if isinstance(value, dict):
        return {key: clean_json_strings(item, (*path, key)) for key, item in value.items()}
    return value


def clean_stream_json_strings(value, cleaner, key=None, path=()):
    if isinstance(value, str):
        if key in STREAM_TEXT_KEYS:
            cleaned = cleaner.clean(value)
        elif should_sanitize_proxy_metadata(path):
            cleaned = clean_channel_markers(value)
        else:
            cleaned = value
        if should_sanitize_proxy_metadata(path):
            return sanitize_behavior_text(cleaned)
        return cleaned
    if isinstance(value, list):
        return [
            clean_stream_json_strings(item, cleaner, path=(*path, index))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            item_key: clean_stream_json_strings(
                item,
                cleaner,
                item_key,
                (*path, item_key),
            )
            for item_key, item in value.items()
        }
    return value


def append_sse_json_event(lines, payload):
    event_type = payload.get("type") if isinstance(payload, dict) else None
    if isinstance(event_type, str) and event_type:
        lines.append(f"event: {event_type}\n")
    lines.append("data: " + json.dumps(payload) + "\n\n")


def split_sse_frames(text):
    frames = []
    current = []
    for line in text.splitlines(keepends=True):
        current.append(line)
        if not line.strip():
            frames.append(current)
            current = []
    if current:
        frames.append(current)
    return frames


def sse_frame_data(frame_lines):
    data_parts = []
    for line in frame_lines:
        if not line.startswith("data:"):
            continue
        value = line[5:]
        if value.startswith(" "):
            value = value[1:]
        data_parts.append(value.rstrip("\r\n"))
    if not data_parts:
        return None
    return "\n".join(data_parts)


def clean_sse_payload(response_body):
    cleaned_lines = []
    stream_cleaners = {}
    pending_empty_message_added = {}
    stream_state = {
        "suppressed_nonfinal_channel_text_seen": False,
        "natural_visible_text_seen": False,
        "action_output_seen": False,
    }
    for frame_lines in split_sse_frames(response_body.decode("utf-8")):
        data = sse_frame_data(frame_lines)
        if data is None:
            cleaned_lines.extend(frame_lines)
            continue
        if data == "[DONE]":
            cleaned_lines.append("data: [DONE]\n\n")
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            cleaned_lines.extend(frame_lines)
            continue
        if has_action_output(payload):
            stream_state["action_output_seen"] = True
        raw_texts = stream_model_texts(payload)
        stream_cleaner = stream_cleaner_for_payload(payload, stream_cleaners)
        cleaned_payload = clean_stream_event_payload(payload, stream_cleaner)
        cleaned_texts = stream_model_texts(cleaned_payload)
        if any(is_suppressed_nonfinal_channel_text(text) for text in raw_texts):
            stream_state["suppressed_nonfinal_channel_text_seen"] = True
        if stream_cleaner is not None and stream_cleaner.suppressed_nonfinal_channel_text_seen:
            stream_state["suppressed_nonfinal_channel_text_seen"] = True
        if should_apply_stream_empty_response_fallback(cleaned_payload, stream_state):
            apply_empty_visible_response_fallback(cleaned_payload)
        synthetic_events = completed_visible_text_stream_events(cleaned_payload, stream_state)
        if synthetic_events:
            synthetic_item_id = stream_event_item_id(synthetic_events[0])
            if synthetic_item_id in pending_empty_message_added:
                pending_payload = pending_empty_message_added.pop(synthetic_item_id)
                append_sse_json_event(cleaned_lines, pending_payload)
            for synthetic_payload in synthetic_events:
                append_sse_json_event(cleaned_lines, synthetic_payload)
            stream_state["natural_visible_text_seen"] = True
        if any(text.strip() for text in cleaned_texts):
            stream_state["natural_visible_text_seen"] = True
        if is_empty_message_added_stream_event(cleaned_payload):
            item_id = stream_event_item_id(cleaned_payload)
            if item_id:
                pending_empty_message_added[item_id] = cleaned_payload
                continue
        if is_empty_text_stream_event(cleaned_payload):
            if cleaned_payload.get("type") == "response.output_item.done":
                pending_empty_message_added.pop(stream_event_item_id(cleaned_payload), None)
            continue
        item_id = stream_event_item_id(cleaned_payload)
        if item_id in pending_empty_message_added:
            pending_payload = pending_empty_message_added.pop(item_id)
            append_sse_json_event(cleaned_lines, pending_payload)
        append_sse_json_event(cleaned_lines, cleaned_payload)
    return "".join(cleaned_lines).encode("utf-8")


def stream_cleaner_for_payload(payload, stream_cleaners):
    item_id = stream_event_item_id(payload)
    if item_id is None:
        item_id = "__default__"
    return stream_cleaners.setdefault(item_id, StreamChannelCleaner())


def stream_event_item_id(payload):
    if not isinstance(payload, dict):
        return None
    item_id = payload.get("item_id")
    if isinstance(item_id, str) and item_id:
        return item_id
    item = payload.get("item")
    if isinstance(item, dict):
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            return item_id
    return None


def clean_stream_event_payload(payload, stream_cleaner):
    if (
        isinstance(payload, dict)
        and payload.get("type") == "response.completed"
        and isinstance(payload.get("response"), dict)
    ):
        cleaned = copy.deepcopy(payload)
        cleaned["response"] = clean_response_payload(payload["response"])
        return clean_json_strings(cleaned)
    return clean_stream_json_strings(payload, stream_cleaner)


def is_empty_text_stream_event(payload):
    if not isinstance(payload, dict):
        return False
    event_type = payload.get("type")
    if event_type == "response.output_text.delta":
        return isinstance(payload.get("delta"), str) and not payload["delta"].strip()
    if event_type == "response.output_text.done":
        return isinstance(payload.get("text"), str) and not payload["text"].strip()
    if event_type == "response.content_part.done":
        part = payload.get("part")
        return (
            isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
            and not part["text"].strip()
        )
    if event_type == "response.output_item.done":
        return is_empty_message_output_item(payload.get("item"))
    return False


def is_empty_message_added_stream_event(payload):
    return (
        isinstance(payload, dict)
        and payload.get("type") == "response.output_item.added"
        and is_empty_message_output_item(payload.get("item"))
    )


def stream_model_texts(value):
    texts = []
    if isinstance(value, str):
        return texts
    if isinstance(value, list):
        for item in value:
            texts.extend(stream_model_texts(item))
        return texts
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PROXY_MODEL_TEXT_KEYS and isinstance(item, str):
                texts.append(item)
            else:
                texts.extend(stream_model_texts(item))
    return texts


def should_apply_stream_empty_response_fallback(payload, stream_state):
    if stream_state["natural_visible_text_seen"]:
        return False
    if stream_state["action_output_seen"]:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("type") != "response.completed":
        return False
    return stream_state["suppressed_nonfinal_channel_text_seen"] or is_completed_response_payload(payload)


def completed_visible_text_stream_events(payload, stream_state):
    if stream_state["natural_visible_text_seen"] or stream_state["action_output_seen"]:
        return []
    if not isinstance(payload, dict) or payload.get("type") != "response.completed":
        return []
    response = payload.get("response")
    if not isinstance(response, dict):
        return []
    text = response.get("output_text")
    if not isinstance(text, str) or not text.strip():
        return []
    item = first_message_output_item(response)
    if item is None:
        return []
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        return []
    return [
        {"type": "response.output_text.delta", "item_id": item_id, "delta": text},
        {"type": "response.output_text.done", "item_id": item_id, "text": text},
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "part": {
                "type": "output_text",
                "annotations": [],
                "logprobs": [],
                "text": text,
            },
        },
        {"type": "response.output_item.done", "item": item},
    ]


def first_message_output_item(response):
    for item in response_output_items(response):
        if is_message_output_item(item):
            return item
    return None


def build_proxy_error_payload(message):
    return {
        "error": {
            "message": sanitize_behavior_text(message),
            "type": "gemma_proxy_error",
        }
    }


def request_content_to_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = request_content_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return request_content_to_text(content["text"])
        if "content" in content:
            return request_content_to_text(content["content"])
    return ""


def latest_user_request_text(payload):
    if not isinstance(payload, dict):
        return ""
    input_value = payload.get("input", payload.get("prompt", ""))
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, dict):
        return request_content_to_text(input_value)
    if isinstance(input_value, list):
        fallback = ""
        for item in reversed(input_value):
            if isinstance(item, dict):
                text = request_content_to_text(item.get("content", item))
                if text and not fallback:
                    fallback = text
                if item.get("role") == "user" and text:
                    return text
            elif isinstance(item, str) and item and not fallback:
                fallback = item
        return fallback
    return ""


def tool_choice_selects_tool(tool_choice):
    if isinstance(tool_choice, (dict, list)):
        return bool(tool_choice)
    if tool_choice is None:
        return False
    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() not in {"", "auto", "none"}
    return bool(tool_choice)


def should_short_circuit_simple_greeting(payload):
    if not isinstance(payload, dict):
        return False
    if tool_choice_selects_tool(payload.get("tool_choice")):
        return False
    return SIMPLE_GREETING_RE.match(latest_user_request_text(payload)) is not None


def build_simple_response_payload(text, model=None):
    payload = {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "output_text": text,
        "output": [build_message_output_item(text)],
    }
    if model:
        payload["model"] = model
    return payload


def sse_event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def build_simple_sse_payload(text, model=None):
    response = build_simple_response_payload(text, model=model)
    response_id = response["id"]
    item = response["output"][0]
    item_id = item["id"]
    in_progress = {"id": response_id, "object": "response", "status": "in_progress"}
    events = [
        ("response.created", {"type": "response.created", "response": in_progress}),
        ("response.in_progress", {"type": "response.in_progress", "response": in_progress}),
        (
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "item": {
                    "content": [],
                    "id": item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "type": "message",
                },
            },
        ),
        (
            "response.content_part.added",
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "part": {"type": "output_text", "text": ""},
            },
        ),
        ("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item_id, "delta": text}),
        ("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "text": text}),
        (
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "part": {
                    "type": "output_text",
                    "annotations": [],
                    "logprobs": [],
                    "text": text,
                },
            },
        ),
        ("response.output_item.done", {"type": "response.output_item.done", "item": item}),
        ("response.completed", {"type": "response.completed", "response": response}),
    ]
    return "".join(sse_event(name, payload) for name, payload in events).encode("utf-8")


def simple_greeting_response_for_request(path, body):
    if not path.startswith("/v1/responses") or not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not should_short_circuit_simple_greeting(payload):
        return None
    if payload.get("stream"):
        return 200, build_simple_sse_payload(SIMPLE_GREETING_RESPONSE_TEXT, model=payload.get("model")), "text/event-stream"
    response = build_simple_response_payload(SIMPLE_GREETING_RESPONSE_TEXT, model=payload.get("model"))
    return 200, json.dumps(response).encode("utf-8"), "application/json"


class ProxyHandler(BaseHTTPRequestHandler):
    upstream = UPSTREAM

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        self.forward()

    def do_POST(self):
        self.forward()

    def forward(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or "0"))
        simple_response = simple_greeting_response_for_request(self.path, body)
        if simple_response is not None:
            status, response_body, content_type = simple_response
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            return

        url = self.upstream + self.path
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding"}
        }
        request = Request(url, data=body if body else None, headers=headers, method=self.command)
        try:
            with urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
                status = response.status
                response_body = response.read()
                content_type = response.headers.get("Content-Type", "application/json")
        except TimeoutError:
            status = 504
            response_body = json.dumps(build_proxy_error_payload(UPSTREAM_TIMEOUT_MESSAGE)).encode("utf-8")
            content_type = "application/json"
        except HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            content_type = exc.headers.get("Content-Type", "application/json")
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                status = 504
                response_body = json.dumps(build_proxy_error_payload(UPSTREAM_TIMEOUT_MESSAGE)).encode("utf-8")
            else:
                status = 502
                response_body = json.dumps(build_proxy_error_payload("Gemma upstream request failed.")).encode("utf-8")
            content_type = "application/json"
        except OSError:
            status = 502
            response_body = json.dumps(build_proxy_error_payload("Gemma upstream request failed.")).encode("utf-8")
            content_type = "application/json"

        if self.path.startswith("/v1/responses"):
            if content_type.startswith("application/json"):
                try:
                    payload = json.loads(response_body.decode("utf-8"))
                    response_body = json.dumps(clean_response_payload(payload)).encode("utf-8")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            elif content_type.startswith("text/event-stream"):
                try:
                    response_body = clean_sse_payload(response_body)
                except UnicodeDecodeError:
                    pass

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)


def run_proxy(host=HOST, port=PORT, upstream=UPSTREAM):
    handler = type("ConfiguredProxyHandler", (ProxyHandler,), {"upstream": upstream.rstrip("/")})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Gemma response proxy listening on http://{host}:{port}, upstream {upstream}", flush=True)
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Gemma channel markers from llama.cpp responses.")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--upstream", default=UPSTREAM)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_proxy(args.host, args.port, args.upstream)
