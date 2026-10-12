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
| 1 | **Conversational intake + fusion assistant**: core done | Shipped in `skinlesion/assistant/` with safety rules in `SAFETY.md` and 39 tests (fake chains only; no live LLM run yet) | Done |
| 1a | **LLM provider factory** (`LLM_PROVIDER` = ollama / nvidia_nim / anthropic, default ollama, offline mode end to end), model benchmark on synthetic transcripts | Planned in the prompt below but **not implemented**: the assistant currently uses the existing Anthropic-only path or the fixed questionnaire | Medium |
| 2 | Retrain at 288 to 320 px | Likelier accuracy gain than the test-time processing that was tried and rejected | Hours of GPU |
| 3 | Put colour constancy / hair removal **inside** training | Test-time use hurt (accuracy 89% to 75-87%); inside training it is untested | Hours of GPU |
| 4 | Age / sex / body-site as model inputs (HAM10000 has them) | Known to help a few points; the app would need to ask for them | Retrain |
| 5 | Evaluate on phone photos (PAD-UFES-20; check its licence first) and on skin-tone diversity data | The app's real input is phone photos; fairness is currently unmeasured | Medium |
| 5a | **Out-of-scope handling** (an "other / not one of these" class or an out-of-distribution detector) | Clinical photo datasets are full of rashes and conditions outside our 7 classes; today the app labels anything it sees as one of the seven | Medium |
| 6 | Publish model weights (about 320 MB; ConvNeXt file exceeds GitHub's 100 MB limit) via a GitHub Release or Git LFS | Others cannot run the app without them | Low |
| 7 | CI (GitHub Actions: pytest), Dockerfile, HawkScan DAST scan (needs `hawk` CLI and `HAWK_API_KEY`) | Reproducibility and security | Low |
| 8 | Add squamous cell carcinoma as an eighth class (ISIC 2019 has 431 extra images) | Currently dropped because it does not map onto HAM10000's classes | Retrain |
| 9 | Voice input (browser Web Speech API first; Vapi only if needed) | Accessibility | Medium |

## Dataset findings: what to add, what not to

Verified means checked against the ISIC challenge data page or our own accounting in this repository.
Everything else comes from secondary sources (including an LLM-written summary) and must be checked
against the dataset's own page before use. Licences vary and several are non-commercial or request-based.

### ISIC challenge years (verified)

| Year | Images | Labels | Licence | Usable for the 7-class model? |
|---|---|---|---|---|
| 2016 Part 3 | 900 | benign / malignant only | CC-0 | No: cannot map to 7 classes |
| 2017 | 2,000 | melanoma / nevus / seborrheic keratosis | CC-0 | Partly: maps to mel, nv, bkl only |
| 2018 Task 3 | 10,015 | 7 classes | CC-BY-NC | Already used (this is HAM10000) |
| 2019 | 25,331 | 8 classes | CC-BY-NC | Already used: 14,885 extra images (SCC and UNK dropped) |
| 2020 | 33,126 | diagnosis CSV | CC-BY-NC | Partly, see below |
| 2024 SLICE-3D | 401,059 | malignancy only | CC-BY-NC | No: different image type, binary labels |

* ISIC 2019 already contains all 10,015 HAM10000 images (verified: exact overlap) and none of the ISIC 2018
  test images. Its 15,316 other images are BCN20000 and MSK; about 12.4k of those are BCN20000, so most of
  what is publicly labelled in BCN20000 is already in our training data. A quoted figure of 24,453 BCN20000
  images looks wrong.
* **ISIC 2020** is the only yearly set worth a trial, mainly for melanoma. Unverified recollection: most of its
  "benign" images have an unknown diagnosis (unusable) and it has only around 600 melanomas, so the gain would
  be modest (roughly 13% more melanoma training data). Check the diagnosis column before committing.
* ISIC 2024 SLICE-3D are 2D crops from total-body photographs with binary labels; the claim that it holds 3D
  volumetric data appears to be inaccurate.

### Other datasets (unverified; check licence and label taxonomy first)

| Dataset | What it is | Use here |
|---|---|---|
| PAD-UFES-20 | 2,298 smartphone images, 6 classes, patient features (licence not confirmed) | Test-only first: phone-photo gap |
| DDI (Stanford) | Biopsy-proven images across light, medium and dark skin tones; registration / data-use terms | Test-only: skin-tone fairness |
| Fitzpatrick 17k | 16,577 clinical images with Fitzpatrick skin-type labels; 100+ conditions; labels reported noisy | Skin-tone analysis; partial class mapping |
| SCIN (Google) | 10k+ crowdsourced phone images with symptoms and skin-type fields; many non-lesion conditions | Out-of-scope examples; check licence |
| Derm7pt | About 1,000 cases with paired dermoscopic and clinical photos | Academic terms; small |
| BCN20000 | Hospital Clinic Barcelona dermoscopy | Mostly already in ISIC 2019 |
| MIDAS | Paired dermoscopic and clinical images, biopsy-proven | Not verified |

### Recommended order

1. **Measure before collecting.** Evaluate the current ensemble on PAD-UFES-20 and DDI (test only, no
   training). The phone-photo and skin-tone gaps are the real risk and have never been measured here.
2. **Handle out-of-scope images** (item 5a above) before adding clinical data that contains conditions
   outside the 7 classes.
3. **Then** consider training additions: ISIC 2020 for melanoma (map nevus, melanoma and
   seborrheic-keratosis-type diagnoses to nv, mel, bkl; drop unknown), and phone-photo data for the classes it
   covers.
4. Report accuracy by Fitzpatrick skin type once any such data is available.

### Rules for adding any dataset

Check the licence; de-duplicate by image ID **and** perceptual hash against HAM10000, ISIC 2019 and the ISIC 2018
test set; group splits by patient or lesion; never tune thresholds on test data; keep the existing evaluation
sets untouched; compare against the current model with the same protocol (change one thing at a time) and
record the result in the model card.

## Compute and model-hosting decisions

* **RunPod: not needed now.** All current training (three models plus baselines) ran on the local RTX 5070
  Laptop GPU in about 2.5 GPU-hours. Revisit only if retraining at 288 to 320 px proves too slow locally
  (untested estimate: a few hours per model). Use rented GPUs with public datasets only, never patient data.
* **LLM for the assistant: local and offline by default (Ollama).** Free, private, and consistent with the rule
  that nothing health-related goes to a third party without consent. The GPU has about 8 GB: a 4-bit quantised
  3 to 8 billion parameter instruct model should fit, but an 8B model plus the vision ensemble on the same GPU
  is tight (an estimate, not tested). Check `nvidia-smi`; if too tight, use a 3 to 4B model or run the vision
  models on CPU.
* **Small local models follow instructions and structured output less reliably.** That is acceptable because
  every safety-critical decision is deterministic code. It does affect conversation quality and symptom
  extraction, so choose the model by measurement (extraction accuracy and safety-rule compliance on the
  synthetic transcripts), using schema-constrained JSON output, Pydantic validation and the fixed
  questionnaire fallback.
* **NVIDIA NIM:** the hosted API catalogue is acceptable for prototyping with **synthetic data only** (not
  offline; requests go to NVIDIA; free tier is rate-limited). Self-hosted NIM containers are probably
  impractical on an 8 GB laptop GPU and production use may need a licence. Verify NVIDIA's current terms.
  Other free cloud tiers (Groq, OpenRouter, Google AI Studio) follow the same rule: synthetic data only.
* **One provider factory** selected by `LLM_PROVIDER` (ollama | nvidia_nim | anthropic; default ollama), so the
  fake-chain tests do not change and offline mode is always testable.

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
LLM PROVIDER
Implement a provider factory (env LLM_PROVIDER = ollama | nvidia_nim | anthropic; default ollama, fully offline)
via LangChain chat models (langchain-ollama, langchain-nvidia-ai-endpoints, langchain-anthropic). Target a
local 3-8B instruct model on an 8 GB GPU (the vision ensemble shares it). Use schema-constrained JSON output for
extraction and validate with Pydantic. Offline mode (no network, no API key) must work end to end, falling back
to a fixed questionnaire and templated report. Benchmark 2-3 candidate local models on the synthetic
transcripts and document extraction accuracy and safety-rule compliance before choosing a default. Hosted
providers may only receive synthetic data unless the user has explicitly consented.

OUT OF SCOPE: diagnosis, treatment advice, storing health data, retraining, changing the risk thresholds.
```
