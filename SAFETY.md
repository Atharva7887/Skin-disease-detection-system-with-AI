# Clinical Screening Assistant — Safety Architecture & Enforcement

This document specifies the safety rules, technical enforcement layers, and corresponding automated tests implemented for the Clinical Screening Assistant.

---

## Safety Rule Matrix

| Rule | Technical Enforcement Layer | Verification Test |
|---|---|---|
| **1. Never diagnose during intake** | • System prompt explicitly forbids diagnosing or condition naming<br>• Pydantic output schema contains no diagnosis field<br>• `validate_intake_response` rejects any message containing HAM10000 class names or synonyms<br>• Deterministic questionnaire fallback requires no model generation | `test_validate_intake_response_catches_disease_names`<br>`test_intake_deterministic_fallback` |
| **2. Never reassure** | • Prompt forbids reassurance words (`safe`, `harmless`, `benign`, `nothing to worry`, etc.)<br>• `BANNED_REASSURANCE_WORDS` scanned on all assistant text<br>• Fixed disclaimers: *"A result that is not flagged is not a reassurance"* | `test_validate_intake_response_catches_reassurance`<br>`test_verifier_catches_reassurance_language` |
| **3. Risk can only escalate (One-Way Gate)** | • Monotonic risk fusion: `final_risk = max(image_risk, symptom_risk)`<br>• Risk ordering: `low (0) < elevated (1) < high (2)`<br>• Pure deterministic Python in `fusion.py` | `test_fusion_max_semantics`<br>`test_property_risk_never_decreases` |
| **4. Red flags are deterministic Python** | • Evaluated as pure Python lambda predicates on `PatientIntake`<br>• The LLM is never involved in red flag identification<br>• Red flag detection triggers fixed urgent clinical referral text | `test_red_flag_rapid_growth`<br>`test_red_flag_bleeding`<br>`test_red_flag_unhealed_sore`<br>`test_red_flag_new_after_50`<br>`test_red_flag_itching_with_change` |
| **5. Prompt-injection defense** | • User inputs wrapped in `<user_input>...</user_input>` delimiters<br>• System prompt instructs model to treat user text strictly as untrusted data<br>• Model constrained to Pydantic structured extraction (`IntakeExtraction`)<br>• No arbitrary tools or shell execution granted | `test_adv_injection_in_symptom` |
| **6. Patient privacy & data retention** | • Ephemeral in-memory sessions with 30-minute automatic TTL eviction; ending or expiring a session also deletes the graph checkpointer state (the patient's answers) and any cached photo<br>• Uploaded images held in volatile memory only during inference and discarded immediately<br>• No PII or session transcript written to server disk or logs<br>• Sessions end immediately when the user restarts or leaves the page (`POST /api/session/<id>/end`); a timer expires them after 10 minutes without activity (30 minutes at most)<br>• Per-IP limits: 20 requests / 60 seconds and 5 open sessions, with distinct error messages | `test_session_manager_ttl_expiry`<br>`test_session_manager_rate_limit`<br>`test_session_manager_max_per_ip`<br>`test_session_manager_on_expire_hook_runs_and_never_breaks_requests`<br>`test_ending_a_session_wipes_graph_state_and_cached_photo` |
| **7. Classifier insulation** | • `Predictor.predict(image, 7)` signature is preserved unchanged<br>• Classifier never receives symptom text or demographic history<br>• Symptom and vision findings fused strictly at the risk-level tier | `test_golden_happy_path`<br>`test_api_existing_predict_and_pages_unaffected` |
| **8. Faithfulness verification** | • Automated summary verified against structured report data<br>• Numeric claims must match report values (no hallucinated probabilities)<br>• Mentioned conditions must exist in `top_classes`<br>• Summaries failing verification are discarded; fallback uses standard template | `test_verifier_catches_hallucinated_number`<br>`test_verifier_catches_unpredicted_class`<br>`test_verifier_passes_faithful_report` |

---

## Deterministic Red Flag Rules

Implemented in `skinlesion/assistant/red_flags.py`:

```python
RED_FLAG_RULES = [
    ("Rapid growth or change reported",
     lambda i: i.evolution in ("growing", "changing")),
    ("Bleeding or oozing reported",
     lambda i: i.symptoms is not None and "bleeding" in i.symptoms),
    ("A sore not healed in 4+ weeks with pain",
     lambda i: i.duration_weeks is not None and i.duration_weeks >= 4
               and i.symptoms is not None and "pain" in i.symptoms),
    ("New pain reported in the lesion",
     lambda i: i.symptoms is not None and "pain" in i.symptoms),
    ("Itching accompanied by change",
     lambda i: i.symptoms is not None and "itching" in i.symptoms
               and i.evolution in ("growing", "changing")),
    ("New lesion appearing after age 50",
     lambda i: i.age_band in ("50_64", "65_plus")
               and i.duration_weeks is not None and i.duration_weeks <= 12),
]
```

---

## Faithfulness Verifier Specifications

Implemented in `skinlesion/assistant/verify.py`:

1. **Number Verification**:
   All numbers and percentages found in generated summaries (`\d+\.?\d*%?`) must exist within the structured `Report` model (calibrated probabilities, duration weeks, concern probability).
2. **Class Name Verification**:
   Condition names and class codes mentioned in the summary must appear in the top predictions. Mentions of other conditions (e.g. unsolicited negative reassurance like *"this is not melanoma"*) trigger a violation.
3. **Word Boundary Checking**:
   Codes and names are verified using strict regex token boundaries to eliminate false positive matches (such as class code `mel` appearing inside the allowed class name `Melanocytic`).
4. **Reassurance Sanitization**:
   Reassurance word scanning isolates recognized diagnostic terms (e.g. `benign keratosis`) to avoid triggering false alarms while strictly intercepting non-clinical reassurance.

---

## Output safety (added after code review)

* **HTML escaping.** Every value interpolated into the printable report (`render.py`) is escaped, and the chat UI
  (`assistant.js`) escapes text before its small markdown conversion. The faithfulness verifier checks numbers,
  class names and reassurance words but not markup, so escaping is the control that stops an LLM or user from
  injecting HTML. Tests: `test_report_html_escapes_untrusted_text`, `test_report_wording_does_not_claim_a_diagnosis_or_outline`.
* **Labels.** The report and UI call class lists "closest image match (not a diagnosis)", the heatmap "model
  attention (not a lesion outline)", and the combined melanoma / basal cell / actinic keratosis probability an
  "uncalibrated model score", never a risk estimate.

## Known limitations

* The verifier only knows the 7 class names; it does not detect other disease names an LLM might write. Its number
  check is strict: any number not in the structured data (even a correct one from elsewhere) triggers the template.
* `vision_node` re-implements the image-risk rule from `skinlesion/triage.py` instead of calling it. Keep them in
  sync (or refactor to share one function) if the thresholds ever change.
* Sessions and the rate limiter are in-process memory: they do not work across multiple server workers, and behind a
  reverse proxy all clients share one IP unless the proxy address is handled.
* Only the fake-chain and questionnaire paths are tested; no live LLM has been run against the assistant, and the
  LLM provider factory in `ROADMAP.md` (1a) is not implemented.
* Symptom-based escalation is a set of conservative rules, not a validated clinical model.
