import io

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from skinlesion.config import CLASS_CODES
from skinlesion.data import class_weights, split_by_lesion
from skinlesion.evaluate import compute_metrics
from skinlesion.model import build_model, load_checkpoint, save_checkpoint
from skinlesion.triage import build_graph, run_triage
from webapp.app import create_app


def fake_predict(top_code, top_p):
    def predict(image, top_k=3):
        rest = (1 - top_p) / (len(CLASS_CODES) - 1)
        rows = [{"code": c, "name": c, "probability": top_p if c == top_code else rest} for c in CLASS_CODES]
        return sorted(rows, key=lambda r: -r["probability"])[:top_k]
    return predict


def noisy_image(size=128):
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_split_has_no_lesion_leakage():
    rng = np.random.default_rng(1)
    n = 600
    df = pd.DataFrame({"lesion_id": [f"L{i // 3}" for i in range(n)],
                       "label": rng.integers(0, 7, n // 3).repeat(3)})
    s = split_by_lesion(df)
    ids = {k: set(v["lesion_id"]) for k, v in s.items()}
    assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
    assert sum(len(v) for v in s.values()) == n


def test_class_weights_favour_rare_classes():
    w = class_weights(np.array([0] * 90 + [1] * 10), num_classes=2)
    assert w[1] > w[0]


def test_metrics_melanoma_recall():
    mel = CLASS_CODES.index("mel")
    probs = np.eye(7)[[mel, 0, mel]]
    targets = np.array([mel, mel, mel])
    assert compute_metrics(probs, targets)["melanoma_recall"] == pytest.approx(2 / 3)


def test_checkpoint_roundtrip(tmp_path):
    model = build_model("mobilenet_v3_large", pretrained=False)
    save_checkpoint(tmp_path / "m.pt", model, "mobilenet_v3_large", 224, {})
    loaded, ckpt = load_checkpoint(tmp_path / "m.pt")
    x = torch.zeros(1, 3, 224, 224)
    assert torch.allclose(model.eval()(x), loaded(x), atol=1e-5)
    assert ckpt["classes"] == CLASS_CODES


def test_timm_backbone_builds():
    pytest.importorskip("timm")
    model = build_model("timm/resnet18", pretrained=False)
    assert model(torch.zeros(1, 3, 224, 224)).shape == (1, 7)


def test_triage_flags_melanoma_high_risk():
    out = run_triage(build_graph(fake_predict("mel", 0.8)), noisy_image())
    assert out["risk"] == "high" and out["needs_review"]
    assert "dermatologist" in out["explanation"].lower()


def test_triage_low_risk_benign():
    out = run_triage(build_graph(fake_predict("nv", 0.9)), noisy_image())
    assert out["risk"] == "low" and not out["needs_review"]


def test_triage_rejects_blank_image():
    out = run_triage(build_graph(fake_predict("nv", 0.9)), Image.new("RGB", (128, 128), "white"))
    assert out["quality_ok"] is False and "predictions" not in out


@pytest.fixture
def client():
    return create_app(predict=fake_predict("nv", 0.9)).test_client()


def test_api_happy_path(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(png_bytes(noisy_image())), "x.png")})
    assert r.status_code == 200
    body = r.get_json()
    assert body["predictions"][0]["code"] == "nv" and "disclaimer" in body


def test_api_rejects_non_image(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "x.png")})
    assert r.status_code == 400


def test_api_rejects_missing_file(client):
    assert client.post("/api/predict").status_code == 400


def test_api_503_without_model(tmp_path):
    c = create_app(model_path=str(tmp_path / "missing.pt")).test_client()
    assert c.post("/api/predict").status_code == 503
    assert c.get("/health").get_json()["status"] == "model_unavailable"
