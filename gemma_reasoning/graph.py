from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass
class ReasoningConfig:
    max_revisions: int = 1
    constraints: list[str] = field(default_factory=list)
    use_langgraph: bool = True


class ReasoningState(TypedDict, total=False):
    payload: dict[str, Any]
    task: str
    constraints: list[str]
    plan: str
    draft: str
    verifier: dict[str, Any]
    revisions: int
    final_text: str
    usage: dict[str, Any]
    model_calls: int
    context_length: int | float | None
    cache_hit: bool | None
    cache_hit_rate: int | float | None


def run_reasoning_graph(
    payload: dict[str, Any],
    *,
    client: Any | None = None,
    programs: Any | None = None,
    config: ReasoningConfig | None = None,
) -> dict[str, Any]:
    if client is None:
        raise ValueError("client is required")
    if programs is None:
        from .dspy_programs import DspyPrograms

        programs = DspyPrograms()
    config = config or ReasoningConfig()
    state: ReasoningState = {
        "payload": copy.deepcopy(payload),
        "constraints": _extract_constraints(payload, config),
        "revisions": 0,
    }
    if config.use_langgraph:
        final_state = _run_with_langgraph(state, client, programs, config)
    else:
        final_state = _run_sequential(state, client, programs, config)
    return {
        "final_text": final_state.get("final_text", ""),
        "plan": final_state.get("plan", ""),
        "verifier": final_state.get("verifier", {}),
        "revisions": final_state.get("revisions", 0),
        "usage": final_state.get("usage", {}),
        "model_calls": final_state.get("model_calls", 0),
        "context_length": final_state.get("context_length"),
        "cache_hit": final_state.get("cache_hit"),
        "cache_hit_rate": final_state.get("cache_hit_rate"),
    }


def _run_with_langgraph(
    state: ReasoningState,
    client: Any,
    programs: Any,
    config: ReasoningConfig,
) -> ReasoningState:
    try:
        END, StateGraph = _load_langgraph()
    except ImportError as exc:
        raise RuntimeError(
            "LangGraph is required for the local Gemma reasoning proxy. "
            "Install it with .\\.venv\\Scripts\\python.exe -m pip install -U langgraph."
        ) from exc

    graph = StateGraph(ReasoningState)
    graph.add_node("extract", lambda current: _extract_node(current))
    graph.add_node("plan", lambda current: _plan_node(current, programs))
    graph.add_node("draft", lambda current: _draft_node(current, client))
    graph.add_node("verify", lambda current: _verify_node(current, programs))
    graph.add_node("revise", lambda current: _revise_node(current, client))
    graph.add_node("finalize", lambda current: _finalize_node(current))
    graph.set_entry_point("extract")
    graph.add_edge("extract", "plan")
    graph.add_edge("plan", "draft")
    graph.add_edge("draft", "verify")
    graph.add_conditional_edges(
        "verify",
        lambda current: "revise" if _should_revise(current, config) else "finalize",
        {"revise": "revise", "finalize": "finalize"},
    )
    graph.add_edge("revise", "verify")
    graph.add_edge("finalize", END)
    return graph.compile().invoke(state)


def _load_langgraph():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
            category=UserWarning,
        )
        from langgraph.graph import END, StateGraph
    return END, StateGraph


def _run_sequential(
    state: ReasoningState,
    client: Any,
    programs: Any,
    config: ReasoningConfig,
) -> ReasoningState:
    state.update(_extract_node(state))
    state.update(_plan_node(state, programs))
    state.update(_draft_node(state, client))
    state.update(_verify_node(state, programs))
    while _should_revise(state, config):
        state.update(_revise_node(state, client))
        state.update(_verify_node(state, programs))
    state.update(_finalize_node(state))
    return state


def _extract_node(state: ReasoningState) -> ReasoningState:
    return {"task": extract_latest_user_task(state["payload"])}


def _plan_node(state: ReasoningState, programs: Any) -> ReasoningState:
    return {"plan": programs.plan(state["task"], state["constraints"])}


def _draft_node(state: ReasoningState, client: Any) -> ReasoningState:
    response = client.create_response(_build_draft_payload(state["payload"], state["plan"]))
    return _with_response_metadata(state, response, {"draft": extract_response_text(response)})


def _verify_node(state: ReasoningState, programs: Any) -> ReasoningState:
    return {
        "verifier": programs.verify(
            state["task"],
            state["plan"],
            state["draft"],
            state.get("constraints", []),
        )
    }


def _revise_node(state: ReasoningState, client: Any) -> ReasoningState:
    response = client.create_response(_build_revision_payload(state))
    return _with_response_metadata(state, response, {
        "draft": extract_response_text(response),
        "revisions": state.get("revisions", 0) + 1,
    })


def _finalize_node(state: ReasoningState) -> ReasoningState:
    return {"final_text": state.get("draft", "")}


def _should_revise(state: ReasoningState, config: ReasoningConfig) -> bool:
    verifier = state.get("verifier") or {}
    return (
        not bool(verifier.get("approved"))
        and state.get("revisions", 0) < config.max_revisions
    )


def _extract_constraints(
    payload: dict[str, Any],
    config: ReasoningConfig,
) -> list[str]:
    constraints: list[str] = []
    instructions = payload.get("instructions")
    if instructions:
        constraints.append(f"instructions: {_content_to_text(instructions)}")
    input_value = payload.get("input", "")
    if isinstance(input_value, list):
        for message in input_value:
            if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
                constraints.append(f"{message.get('role')}: {_content_to_text(message.get('content', ''))}")
    raw_constraints = payload.get("constraints", config.constraints)
    if raw_constraints is None:
        return constraints
    if isinstance(raw_constraints, str):
        constraints.append(raw_constraints)
    elif isinstance(raw_constraints, list):
        constraints.extend(str(item) for item in raw_constraints if item is not None)
    else:
        constraints.append(str(raw_constraints))
    return constraints


def extract_latest_user_task(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return _content_to_text(payload)
    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        return input_value
    if isinstance(input_value, list):
        for message in reversed(input_value):
            if isinstance(message, dict) and message.get("role") == "user":
                return _content_to_text(message.get("content", ""))
        if input_value:
            return _content_to_text(input_value[-1])
    return _content_to_text(input_value)


def extract_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return ""
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    output = response.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = item.get("content") if isinstance(item, dict) else item
            text = _content_to_text(content)
            if text:
                texts.append(text)
        return "\n".join(texts)
    return _content_to_text(response.get("content", ""))


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if isinstance(value.get("content"), (str, list, dict)):
            return _content_to_text(value["content"])
        return ""
    if isinstance(value, list):
        parts = [_content_to_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value)


def _build_draft_payload(payload: dict[str, Any], plan: str) -> dict[str, Any]:
    draft_payload = copy.deepcopy(payload)
    _append_input_message(
        draft_payload,
        {
            "role": "user",
            "content": (
                "Private non-authoritative reasoning plan. "
                "Use only when it does not conflict with system, developer, project, or user instructions.\n"
                f"{plan}"
            ),
        },
    )
    return draft_payload


def _build_revision_payload(state: ReasoningState) -> dict[str, Any]:
    payload = copy.deepcopy(state["payload"])
    verifier = state.get("verifier") or {}
    notes = verifier.get("notes") or "Missing or incomplete answer."
    _append_input_message(
        payload,
        {
            "role": "user",
            "content": f"Previous draft for revision:\n{state.get('draft', '')}",
        },
    )
    _append_input_message(
        payload,
        {
            "role": "user",
            "content": (
                "Revise the previous draft while preserving the original task and all higher-priority context.\n\n"
                f"Original task:\n{state.get('task', '')}\n\n"
                f"Private non-authoritative reasoning plan:\n{state.get('plan', '')}\n\n"
                f"Verifier notes:\n{notes}\n\n"
                "Missing or incomplete answer: address the verifier notes directly."
            ),
        },
    )
    return payload


def _append_input_message(payload: dict[str, Any], message: dict[str, str]) -> None:
    input_value = payload.get("input", "")
    if isinstance(input_value, list):
        payload["input"] = [*copy.deepcopy(input_value), message]
    elif isinstance(input_value, str):
        payload["input"] = [{"role": "user", "content": input_value}, message]
    else:
        payload["input"] = [{"role": "user", "content": _content_to_text(input_value)}, message]


def _with_response_metadata(
    state: ReasoningState,
    response: Any,
    update: ReasoningState,
) -> ReasoningState:
    metadata = _response_metadata(response)
    update["model_calls"] = state.get("model_calls", 0) + 1
    if metadata.get("usage"):
        update["usage"] = _merge_usage(state.get("usage", {}), metadata["usage"])
    if "context_length" in metadata:
        update["context_length"] = _max_number(state.get("context_length"), metadata["context_length"])
    if "cache_hit" in metadata:
        update["cache_hit"] = bool(state.get("cache_hit")) or bool(metadata["cache_hit"])
    if "cache_hit_rate" in metadata:
        update["cache_hit_rate"] = metadata["cache_hit_rate"]
    return update


def _response_metadata(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    metadata: dict[str, Any] = {}
    usage = response.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = dict(usage)
    context_length = _first_number(response, "context_length", "context_length_used", "n_ctx", "ctx_size")
    if context_length is not None:
        metadata["context_length"] = context_length
    cache_hit = response.get("cache_hit", response.get("cache_used"))
    if isinstance(cache_hit, bool):
        metadata["cache_hit"] = cache_hit
    cache_hit_rate = _first_number(response, "cache_hit_rate")
    if cache_hit_rate is not None:
        metadata["cache_hit_rate"] = cache_hit_rate
    return metadata


def _merge_usage(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in new.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            merged[key] = value
            continue
        previous = merged.get(key)
        if isinstance(previous, bool) or not isinstance(previous, (int, float)):
            merged[key] = value
        else:
            merged[key] = previous + value
    return merged


def _first_number(payload: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return value
    return None


def _max_number(left: Any, right: Any) -> int | float | None:
    left_is_number = not isinstance(left, bool) and isinstance(left, (int, float))
    right_is_number = not isinstance(right, bool) and isinstance(right, (int, float))
    if left_is_number and right_is_number:
        return max(left, right)
    if right_is_number:
        return right
    if left_is_number:
        return left
    return None
