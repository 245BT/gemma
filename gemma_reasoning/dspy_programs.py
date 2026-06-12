from __future__ import annotations

from typing import Any


DEFAULT_MODEL = "openai/gemma-4-26b-a4b-it-uncensored-q4-k-m"
DEFAULT_API_BASE = "http://127.0.0.1:8081/v1"
DEFAULT_API_KEY = "local-gemma-placeholder"


class LocalHeuristicPrograms:
    def plan(self, _task: str, _constraints: list[str]) -> str:
        return "Answer the task directly."

    def verify(
        self,
        _task: str,
        _plan: str,
        draft: str,
        _constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "approved": bool(str(draft).strip()),
            "notes": "Approved when the draft is non-empty.",
        }


class DspyPrograms:
    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_base: str = DEFAULT_API_BASE,
        api_key: str = DEFAULT_API_KEY,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._dspy: Any | None = None
        self._lm: Any | None = None
        self._planner: Any | None = None
        self._verifier: Any | None = None

    def plan(self, task: str, constraints: list[str]) -> str:
        dspy = self._load_dspy()
        if self._planner is None:
            self._planner = dspy.Predict(self._build_plan_signature(dspy))
        result = self._call_with_lm(
            dspy,
            self._planner,
            task=task,
            constraints="\n".join(constraints),
        )
        return str(getattr(result, "plan", result)).strip()

    def verify(self, task: str, plan: str, draft: str, constraints: list[str] | None = None) -> dict[str, Any]:
        dspy = self._load_dspy()
        if self._verifier is None:
            self._verifier = dspy.Predict(self._build_verify_signature(dspy))
        result = self._call_with_lm(
            dspy,
            self._verifier,
            task=task,
            constraints="\n".join(constraints or []),
            plan=plan,
            draft=draft,
        )
        approved = self._coerce_bool(getattr(result, "approved", False))
        notes = str(getattr(result, "notes", "")).strip()
        return {"approved": approved, "notes": notes}

    def _load_dspy(self) -> Any:
        if self._dspy is not None:
            return self._dspy
        try:
            import dspy
        except ImportError as exc:
            raise RuntimeError(
                "DSPy is required for default reasoning programs. "
                "Install it in the project virtual environment or pass fake programs."
            ) from exc
        self._dspy = dspy
        return dspy

    def _get_lm(self, dspy: Any) -> Any:
        if self._lm is not None:
            return self._lm
        self._lm = dspy.LM(
            self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            model_type="responses",
            max_tokens=self.max_tokens,
        )
        return self._lm

    def _call_with_lm(self, dspy: Any, predictor: Any, **kwargs: Any) -> Any:
        with dspy.context(lm=self._get_lm(dspy)):
            return predictor(**kwargs)

    @staticmethod
    def _build_plan_signature(dspy: Any) -> type:
        class PlanSignature(dspy.Signature):
            """Produce a concise plan for answering the task."""

            task = dspy.InputField()
            constraints = dspy.InputField()
            plan = dspy.OutputField()

        return PlanSignature

    @staticmethod
    def _build_verify_signature(dspy: Any) -> type:
        class VerifySignature(dspy.Signature):
            """Judge whether the draft answers the task and follows constraints."""

            task = dspy.InputField()
            constraints = dspy.InputField()
            plan = dspy.InputField()
            draft = dspy.InputField()
            approved = dspy.OutputField()
            notes = dspy.OutputField()

        return VerifySignature

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "approved"}
