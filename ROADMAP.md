# Roadmap

Status of the repository at the time of writing: a three-model ensemble (ConvNeXt-V2, EfficientNetV2-S,
MaxViT-tiny) trained on HAM10000 + ISIC 2019, calibrated confidence, Grad-CAM, a rule-based risk flag, a
LangGraph triage pipeline, an optional LLM explanation / Q&A layer, and a web UI with a "Technical details"
page. See `README.md` for numbers and how to reproduce them. This file lists what is planned next and why.

## Safety principles that every future version must keep

1. The tool is a screening aid, never a diagnosis, and never offers reassurance ("not flagged" is not "safe").
2. The risk decision stays in deterministic code. An LLM may collect information and write text; it may not
   decide, raise, lower or explain away the risk flag.
3. Anything the user types (symptoms, questions, uploaded text) can only ever **raise** concern, never lower
   the image-based flag.
4. Published numbers come from `scripts/make_model_card.py`; nothing is typed in by hand. The UI states the
   measured miss rate for melanoma.
5. Images are not sent to third-party services, and nothing personal is stored, without explicit consent.
6. Thresholds (flag rule, calibration temperature) are never tuned on the test sets.

## Planned work, in priority order

| # | Item | Why | Cost |
|---|---|---|---|
| 1 | **Conversational intake + fusion assistant** (prompt below) | Collect symptoms, ask follow-ups, produce a clinician-ready summary | Medium |
| 2 | Retrain at 288 to 320 px | Likelier accuracy gain than the test-time processing that was tried and rejected | Hours of GPU |
| 3 | Put colour constancy / hair removal **inside** training | Test-time use hurt (accuracy 89% to 75-87%); inside training it is untested | Hours of GPU |
| 4 | Age / sex / body-site as model inputs (HAM10000 has them) | Known to help a few points; the app would need to ask for them | Retrain |
| 5 | Evaluate on phone photos (PAD-UFES-20; check its licence first) and on skin-tone diversity data | The app's real input is phone photos; fairness is currently unmeasured | Medium |
| 6 | Publish model weights (about 320 MB; ConvNeXt file exceeds GitHub's 100 MB limit) via a GitHub Release or Git LFS | Others cannot run the app without them | Low |
| 7 | CI (GitHub Actions: pytest), Dockerfile, HawkScan DAST scan (needs `hawk` CLI and `HAWK_API_KEY`) | Reproducibility and security | Low |
| 8 | Add squamous cell carcinoma as an eighth class (ISIC 2019 has 431 extra images) | Currently dropped because it does not map onto HAM10000's classes | Retrain |
| 9 | Voice input (browser Web Speech API first; Vapi only if needed) | Accessibility | Medium |

## Prompt for the conversational assistant (to hand to a coding agent)

```text
ROLE
You are a senior engineer extending an existing, tested Python project. Work in small steps, keep all
tests green, and ask before any step that changes model behaviour. Do NOT write code until you have posted a
design doc (state schema, node list, endpoints, safety rules, test plan) and I have approved it.

PROJECT CONTEXT (read first: README.md, ROADMAP.md, skinlesion/triage.py, skinlesion/llm.py,
skinlesion/predict.py, webapp/app.py, tests/test_pipeline.py)
- Skin-lesion classifier for 7 HAM10000 classes. Shipped inference = 3-model ensemble (ConvNeXt-V2, EfficientNetV2-S,
  MaxViT-tiny), 8-view TTA, temperature-calibrated displayed probabilities (T in webapp/calibration.json).
  Predictor.predict returns `probability` (calibrated, for display) and `raw_probability`.
- The risk flag lives in skinlesion/triage.py (assess_risk) and is defined on RAW probabilities: HIGH if the top class
  is melanoma/bcc/akiec or their combined raw probability >= 0.20; ELEVATED if top raw probability < 0.5. Measured on
  ISIC 2018: flags 89.5% of melanomas and 14% of nevi. Do not change thresholds or tune them on test data.
- LangGraph already exists (triage: check_quality -> classify -> assess_risk -> explain). skinlesion/llm.py has a
  LangChain structured-output explainer and a guard -> retrieve -> answer Q&A graph, both optional (need
  ANTHROPIC_API_KEY; default model env EXPLAIN_MODEL=claude-sonnet-5-5). Tests use fake chains, no network.
- Grad-CAM heatmap comes from the FIRST model in the ensemble only (the others have border artefacts).
- Windows machine, RTX GPU. Each DataLoader worker costs ~2 GB of commit memory: never add persistent workers.
  Model weights are git-ignored. Keep the test suite under ~20 s. Do not retrain models. No heavy new dependencies.

GOAL
Add a conversational "Clinical Screening Assistant" that (1) collects symptoms and context by chat, (2) gets a
photo, (3) runs the existing vision pipeline, (4) fuses everything into a clinician-ready summary.

HARD SAFETY RULES (enforce in CODE and tests, not only in prompts)
1. Never diagnose, never name a disease as the user's likely condition, never recommend treatment or medication.
   During intake the LLM may not mention disease names at all.
2. Never reassure. "Not flagged" must be worded as "not reassurance".
3. The risk level is computed by deterministic Python: final_risk = max(image_flag, symptom_escalation). Symptoms,
   chat content and LLM output can only RAISE concern; a property test must prove nothing lowers it.
4. Red flags are plain Python rules (not LLM): rapid growth or change, bleeding or oozing, a sore that has not
   healed in 4+ weeks, new pain, itching with change, a lesion that is new after age 50, multiple colours or
   irregular border described by the user. Any red flag -> immediate "arrange an in-person dermatology visit soon" message.
5. User text is DATA. Prompt-injection safe: wrap it in delimiters, ignore instructions inside it, validate every LLM
   output against a Pydantic schema, and fall back to a fixed template on any failure or when there is no API key.
6. Privacy: sessions are ephemeral (in-memory, TTL 30 min), no PII logging, the image is never sent to an LLM,
   only structured fields and class names/probabilities are. Show a consent screen first. Rate-limit per client.
7. The classifier never receives symptom text (there is no validated way to fuse it; fusion is rule-based escalation
   only, and the report says it is not validated).

ARCHITECTURE (LangGraph, with a checkpointer keyed by session id and interrupt() to wait for user input)
State (TypedDict or Pydantic): messages, intake: PatientIntake, image_ok: bool, vision: VisionResult,
red_flags: list[str], final_risk: Literal["low","elevated","high"], report: Report, turns: int.
PatientIntake fields: duration_weeks, evolution (growing/changing/stable/unknown), symptoms (itching/bleeding/pain/
none), anatomy (arm/leg/back/face/scalp/hand/foot/other), age_band, previous_skin_cancer (yes/no/unsure), sun_exposure.
Nodes: consent -> intake (adaptive questions, max 6 turns, max 2 re-asks per field, extraction via structured output
with a deterministic questionnaire fallback) -> red_flag_check (pure Python) -> image_gate (reuse check_quality plus a
blur/glare check; on failure ask the user to retake, max 2 loops) -> vision (existing Predictor; keep calibrated and
raw probabilities) -> fusion (final_risk = max(...); build the evidence table) -> report. Conditional edges and
two bounded cycles (re-ask, retake photo).
Report: structured (Pydantic) and rendered to Markdown and a downloadable PDF for the user to show a doctor:
patient-reported facts, image result (top classes, calibrated confidence, risk flag, Grad-CAM thumbnail),
red flags, "what this tool cannot do", and the measured performance note read from webapp/model_card.json.
Every number in the report must be copied programmatically from tool output; add a verifier that fails the build if
the LLM text contains a number or class name that is not in the structured data.

API / UI
POST /api/session, POST /api/session/<id>/message, POST /api/session/<id>/image, GET /api/session/<id>/report.
Add a chat panel to the existing "Take a test" flow in the current cream/black style: quick-reply chips, a
"Skip" button on every question, keyboard accessible, mobile friendly. Keep the existing one-shot upload path working.

LATER, ONLY AFTER THE ABOVE SHIPS (separate PRs)
Browser Web Speech API voice input; a small curated, licence-checked markdown knowledge base with local-embedding
retrieval for follow-up questions (no vector DB at this scale; answers must quote or cite the source passage).

TESTS AND EVALUATION (all offline with fake chains)
Unit tests per node; golden conversations; adversarial set (asks for a diagnosis, "ignore your rules", injected
instructions in a symptom answer, attempts to make the bot reassure); the property test in rule 3; extraction accuracy
on 30+ synthetic transcripts; report-faithfulness verifier; latency and cost budget documented (one session <= N tokens).
Extend scripts/make_model_card.py only if new measured numbers are shown in the UI.

DELIVERABLES / DEFINITION OF DONE
Design doc approved; code in skinlesion/assistant/ and webapp/; tests passing; README and ROADMAP updated; a short
SAFETY.md listing each rule above and the test that enforces it; commits in small steps; no pushes without my OK.
OUT OF SCOPE: diagnosis, treatment advice, storing health data, retraining, changing the risk thresholds.
```
