# Skin Disease Detection System with AI

![Home Page Screenshot](screenshots/ss1.png)

A skin-lesion classifier for the seven diagnostic categories of the
[HAM10000](https://doi.org/10.7910/DVN/DBW86T) dataset, with a Flask web UI/API and a
[LangGraph](https://langchain-ai.github.io/langgraph/) triage pipeline.

> **Not a medical device.** This is a research/educational prototype. It cannot diagnose
> skin cancer. See a qualified dermatologist for any skin concern.

## What's in v2

| Area | Before | Now |
|---|---|---|
| Model | Decision Tree on flattened 64×64 pixels; a separate Keras CNN whose `.h5` was not in the repo | ImageNet-pretrained EfficientNet / MobileNetV3 / ResNet50, or any modern [timm](https://github.com/huggingface/pytorch-image-models) backbone (ConvNeXt-V2, EVA02, …) |
| Splitting | Random split (images of the same lesion leak between train and test) | Stratified split **grouped by `lesion_id`**, with a leakage assertion |
| Imbalance | None (nevus is ~67% of the data) | Class-weighted loss, augmentation, macro-F1 early stopping |
| Metrics | Accuracy | Accuracy, balanced accuracy, macro-F1, per-class report, melanoma recall, confusion matrix |
| Serving | Raw `.save(file.filename)` upload, missing templates | In-memory image handling, format/size limits, tests, `/health` |
| Agentic layer | – | LangGraph: quality check → classify → risk routing → explanation (Claude optional) |

## Setup

```sh
python -m venv .venv && .venv\Scripts\activate
# RTX 50-series (Blackwell) needs CUDA 12.8 wheels; use plain `pip install torch torchvision` for CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Train

Download HAM10000 (Harvard Dataverse or Kaggle `kmader/skin-cancer-mnist-ham10000`) and unzip it
so that `HAM10000_metadata.csv` and the images are somewhere under one folder.

```sh
python train.py --data-dir D:/datasets/HAM10000 --model efficientnet_b0 --epochs 15
# newer backbone:
python train.py --data-dir D:/datasets/HAM10000 --model timm/convnextv2_tiny.fcmae_ft_in22k_in1k
```

Outputs `models/best.pt` (weights + class list + metrics) and `models/metrics.json`.
The test split is evaluated once, after training, on the best validation checkpoint.

## Run the app

```sh
python -m webapp.app            # http://127.0.0.1:5000   (MODEL_PATH defaults to models/best.pt)
curl -F image=@lesion.jpg http://127.0.0.1:5000/api/predict
```

Set `ANTHROPIC_API_KEY` to have Claude write the plain-language explanation (model via
`EXPLAIN_MODEL`, default `claude-sonnet-5-5`). Without a key a deterministic template is used.
The LLM only receives class probabilities, never the image, and is instructed not to diagnose.

## Triage graph

```
check_quality ──► classify ──► assess_risk ──► explain
      └─(blank / tiny image)──► END
```

`assess_risk` marks a case **high** if the top class (or ≥20% combined probability) is
melanoma, basal cell carcinoma or actinic keratosis, **elevated** if the top probability is under 50%,
otherwise **low**.

## Tests

```sh
python -m pytest -q
```

## Repo layout

```
skinlesion/   config, data, model, evaluate, predict, triage (LangGraph)
webapp/       Flask app + template
train.py      training CLI
tests/        unit + API tests
legacy/       original Decision Tree script and Flask backend (kept for reference)
DT/, screenshots/, Skin Disease Detection Report.pdf   original project material
```

## Limitations

HAM10000 is dermoscopic imagery from two sites and skews toward lighter skin tones; the model
will not generalise to phone photos or darker skin without additional data. Probabilities are not
calibrated clinical risk.
