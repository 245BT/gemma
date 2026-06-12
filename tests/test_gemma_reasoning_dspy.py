import sys
import types
import unittest
from unittest.mock import patch

from gemma_reasoning.dspy_programs import DspyPrograms


class FakePrediction:
    plan = "configured plan"
    approved = "true"
    notes = "configured notes"


class FakeDspy(types.SimpleNamespace):
    class Signature:
        pass

    @staticmethod
    def InputField():
        return object()

    @staticmethod
    def OutputField():
        return object()

    def __init__(self):
        super().__init__()
        self.context_lm = None
        self.lm_kwargs = None
        self.predict_kwargs = []

    def LM(self, *args, **kwargs):
        self.lm_kwargs = (args, kwargs)
        return {"args": args, "kwargs": kwargs}

    def context(self, *, lm):
        fake_dspy = self

        class Context:
            def __enter__(self):
                fake_dspy.context_lm = lm

            def __exit__(self, exc_type, exc, tb):
                return False

        return Context()

    def Predict(self, signature):
        def predict(**kwargs):
            self.predict_kwargs.append(kwargs)
            return FakePrediction()

        return predict


class DspyProgramsTests(unittest.TestCase):
    def test_default_programs_configure_dspy_to_local_gemma(self):
        fake_dspy = FakeDspy()
        with patch.dict(sys.modules, {"dspy": fake_dspy}):
            programs = DspyPrograms(api_base="http://127.0.0.1:8081/v1")
            self.assertEqual(programs.plan("task", []), "configured plan")
            verifier = programs.verify("task", "plan", "draft", ["instruction constraint"])

        args, kwargs = fake_dspy.lm_kwargs
        self.assertEqual(args[0], "openai/gemma-4-26b-a4b-it-uncensored-q4-k-m")
        self.assertEqual(kwargs["api_base"], "http://127.0.0.1:8081/v1")
        self.assertEqual(kwargs["api_key"], "local-gemma-placeholder")
        self.assertIsNotNone(fake_dspy.context_lm)
        self.assertEqual(verifier, {"approved": True, "notes": "configured notes"})
        self.assertEqual(fake_dspy.predict_kwargs[-1]["constraints"], "instruction constraint")


if __name__ == "__main__":
    unittest.main()
