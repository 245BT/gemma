# Gemma Output Behavior Sanitizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce Gemma response behavior rules for final text plus tool, error, and proxy strings returned through the Responses API.

**Architecture:** Add compact guidance to the generated Codex model catalog and hard enforcement in `gemma_response_proxy.py`. The proxy sanitizer runs during existing JSON and SSE string traversal after channel/tool marker cleanup.

**Tech Stack:** Python standard library, `unittest`, local Codex model catalog generation.

---

## File Structure

- Modify `tests/test_gemma_response_proxy.py`: response sanitizer regression tests.
- Modify `tests/test_setup_local_codex.py`: base-instruction behavior-rule assertions.
- Modify `gemma_response_proxy.py`: central string sanitizer and recursive response traversal.
- Modify `setup_local_codex.py`: compact base-instruction rule fragments.

## Task 1: Response Proxy Failing Tests

**Files:**
- Modify: `tests/test_gemma_response_proxy.py`
- Test: `tests/test_gemma_response_proxy.py`

- [ ] **Step 1: Write failing sanitizer tests**

Add these methods inside `GemmaResponseProxyTests` after `test_clean_response_payload_removes_internal_tool_call_prefix_before_answer`:

```python
    def test_clean_response_payload_sanitizes_output_behavior_rules(self):
        payload = {
            "output_text": (
                "Ship it \\U0001f680 \\u26a0\\ufe0f It is 100% achieveable and PRODUCTION-READY. "
                "WARNING: HARMFUL EMERGENCY."
            ),
            "output": [
                {
                    "content": [
                        {
                            "type": "output_text",
                            "text": "This is 100% achievable and production-ready \\u2705",
                        }
                    ]
                }
            ],
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)

        combined = json.dumps(cleaned, ensure_ascii=False)
        self.assertNotIn("\\U0001f680", combined)
        self.assertNotIn("\\u26a0", combined)
        self.assertNotIn("\\u2705", combined)
        self.assertNotIn("100% achieveable", combined)
        self.assertNotIn("100% achievable", combined)
        self.assertNotIn("PRODUCTION-READY", combined)
        self.assertNotIn("production-ready", combined)
        self.assertNotIn("WARNING", combined)
        self.assertNotIn("HARMFUL", combined)
        self.assertNotIn("EMERGENCY", combined)
        self.assertIn("likely achievable with verification", combined)
        self.assertIn("ready for review", combined)
        self.assertIn("warning", combined)
        self.assertIn("harmful", combined)
        self.assertIn("emergency", combined)

    def test_clean_response_payload_sanitizes_nested_error_and_tool_strings(self):
        payload = {
            "error": {
                "message": "Proxy says \\u26d4 100% achievable and production-ready",
                "details": {"tool_output": "Tool result \\u2757 WARNING"},
            }
        }

        cleaned = gemma_response_proxy.clean_response_payload(payload)
        combined = json.dumps(cleaned, ensure_ascii=False)

        self.assertNotIn("\\u26d4", combined)
        self.assertNotIn("\\u2757", combined)
        self.assertNotIn("100% achievable", combined)
        self.assertNotIn("production-ready", combined)
        self.assertNotIn("WARNING", combined)
        self.assertIn("likely achievable with verification", combined)
        self.assertIn("ready for review", combined)
        self.assertIn("warning", combined)

    def test_clean_sse_payload_sanitizes_streamed_behavior_rules(self):
        raw = (
            b'data: {"type":"response.output_text.delta","delta":"Deploy \\\\ud83d\\\\udea8 100% achievable"}\\n\\n'
            b'data: {"type":"response.output_text.delta","delta":" and production-ready WARNING"}\\n\\n'
            b"data: [DONE]\\n\\n"
        )

        cleaned = gemma_response_proxy.clean_sse_payload(raw).decode("utf-8")

        self.assertNotIn("\\\\ud83d\\\\udea8", cleaned)
        self.assertNotIn("100% achievable", cleaned)
        self.assertNotIn("production-ready", cleaned)
        self.assertNotIn("WARNING", cleaned)
        self.assertIn("likely achievable with verification", cleaned)
        self.assertIn("ready for review", cleaned)
        self.assertIn("warning", cleaned)
```

- [ ] **Step 2: Run tests and confirm the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy -v
```

Expected: FAIL because the proxy does not yet sanitize emoji/icon symbols, guarantee phrases, `production-ready`, nested error strings, or all-caps emergency terms.

## Task 2: Response Proxy Sanitizer Implementation

**Files:**
- Modify: `gemma_response_proxy.py`
- Test: `tests/test_gemma_response_proxy.py`

- [ ] **Step 1: Add sanitizer imports, constants, and functions**

In `gemma_response_proxy.py`, add `unicodedata` beside the existing imports and add this block after `STREAM_CHANNEL_MARKERS`:

```python
VISUAL_SYMBOL_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
)
VISUAL_SYMBOL_CODEPOINTS = {0x200D, 0x20E3, 0xFE0E, 0xFE0F}
ABSOLUTE_ACHIEVABLE_RE = re.compile(r"\b100\s*%\s+ach(?:ie|ei)vable\b", re.IGNORECASE)
PRODUCTION_READY_RE = re.compile(r"\bproduction[-\s]?ready\b", re.IGNORECASE)
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
```

- [ ] **Step 2: Route cleaned text through the sanitizer**

Update `clean_channel_markers()` in `gemma_response_proxy.py` to sanitize all returned text:

```python
def clean_channel_markers(text):
    final_text = text_after_last_final_channel(text)
    if final_text is not None:
        return sanitize_behavior_text(clean_internal_tool_call_markers(final_text))
    if starts_with_channel_marker(text):
        return ""
    return sanitize_behavior_text(clean_internal_tool_call_markers(text))
```

- [ ] **Step 3: Make non-streaming response cleanup recursive**

Replace `clean_response_payload()` with:

```python
def clean_response_payload(payload):
    cleaned = copy.deepcopy(payload)
    if isinstance(cleaned.get("output_text"), str):
        cleaned["output_text"] = clean_channel_markers(cleaned["output_text"])
    for item in cleaned.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                content["text"] = clean_channel_markers(content["text"])
    return clean_json_strings(cleaned)
```

- [ ] **Step 4: Run focused tests and confirm they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy -v
```

Expected: PASS for `tests.test_gemma_response_proxy`.

## Task 3: Base Instruction Failing Tests

**Files:**
- Modify: `tests/test_setup_local_codex.py`
- Test: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Add base-instruction assertions**

In `test_base_instructions_fit_token_budget_and_preserve_required_behavior`, extend the required fragments list with:

```python
                "No emojis warning signs icons",
                "No 100 percent achievable guarantees",
                "No production-ready claims",
                "No all-caps harmfulness emergency emphasis",
                "read verification-before-completion before final claims",
                "follow Codex CLI project rules",
```

- [ ] **Step 2: Run tests and confirm the expected failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex.LocalCodexSetupTests.test_base_instructions_fit_token_budget_and_preserve_required_behavior -v
```

Expected: FAIL because `setup_local_codex.build_base_instructions()` does not yet include the new compact fragments.

## Task 4: Base Instruction Implementation

**Files:**
- Modify: `setup_local_codex.py`
- Test: `tests/test_setup_local_codex.py`

- [ ] **Step 1: Update compact behavior instruction fragments**

In `setup_local_codex.build_base_instructions()`, replace the final fragment:

```python
        "efficient warmth zero Do not use emojis"
```

with:

```python
        "efficient warmth zero No emojis warning signs icons "
        "No 100 percent achievable guarantees No production-ready claims "
        "No all-caps harmfulness emergency emphasis "
        "read verification-before-completion before final claims "
        "follow Codex CLI project rules"
```

- [ ] **Step 2: Run setup tests and confirm they pass**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_setup_local_codex -v
```

Expected: PASS for `tests.test_setup_local_codex`. If the token-budget test fails by a small amount, reduce wording without removing any required behavior fragment.

## Task 5: Final Verification

**Files:**
- Verify: `tests/test_gemma_response_proxy.py`
- Verify: `tests/test_setup_local_codex.py`
- Verify: full test suite through `package.json`

- [ ] **Step 1: Run focused verification pass one**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy tests.test_setup_local_codex -v
```

Expected: PASS.

- [ ] **Step 2: Run focused verification pass two**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_gemma_response_proxy tests.test_setup_local_codex -v
```

Expected: PASS.

- [ ] **Step 3: Run full suite**

Run:

```powershell
npm test
```

Expected: PASS. If unrelated pre-existing failures appear, record the exact failing tests and rerun the focused verification commands to keep the completed scope clear.

- [ ] **Step 4: Inspect scoped diff**

Run:

```powershell
git diff -- gemma_response_proxy.py setup_local_codex.py tests/test_gemma_response_proxy.py tests/test_setup_local_codex.py docs/superpowers/plans/2026-06-14-gemma-output-behavior-sanitizer.md
```

Expected: diff only includes the sanitizer, prompt fragments, tests, and this implementation plan.
