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


@pytest.mark.parametrize("path", ["/", "/test", "/learn", "/contact"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200 and b"not a medical device" in r.data.lower()


def test_api_includes_class_info(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(png_bytes(noisy_image())), "x.png")})
    assert r.get_json()["info"]["title"] == "Melanocytic Nevi"


def test_soft_class_weights_are_milder():
    labels = np.array([0] * 90 + [1] * 10)
    full = class_weights(labels, 2, power=1.0)
    soft = class_weights(labels, 2, power=0.5)
    assert soft[1] / soft[0] < full[1] / full[0]
    assert torch.allclose(class_weights(labels, 2, power=0.0), torch.ones(2))


def test_predictor_ensembles_checkpoints(tmp_path):
    from skinlesion.predict import Predictor
    for i in range(2):
        m = build_model("mobilenet_v3_large", pretrained=False)
        save_checkpoint(tmp_path / f"m{i}.pt", m, "mobilenet_v3_large", 64, {})
    single = Predictor(str(tmp_path / "m0.pt"), device="cpu").predict(noisy_image(), top_k=7)
    both = Predictor(f"{tmp_path / 'm0.pt'},{tmp_path / 'm1.pt'}", device="cpu").predict(noisy_image(), top_k=7)
    assert len(both) == 7 and abs(sum(r["probability"] for r in both) - 1) < 0.01
    assert [r["code"] for r in single] != [] and len(Predictor(str(tmp_path / "m0.pt"), device="cpu").members) == 1


# ---------- LangChain / LangGraph LLM features (with fake chains: no network, no API key) ----------
from langchain_core.runnables import RunnableLambda

from skinlesion.llm import Explanation, build_chat_graph, known_codes, make_explainer


def test_structured_explainer_feeds_triage():
    chain = RunnableLambda(lambda x: Explanation(summary="Short summary.", next_steps=["See a dermatologist."]))
    graph = build_graph(fake_predict("mel", 0.8), explainer=make_explainer(chain=chain))
    out = run_triage(graph, noisy_image())
    assert out["explanation"] == "Short summary. 1. See a dermatologist."


def test_explainer_failure_falls_back_to_template():
    def boom(_):
        raise RuntimeError("quota")
    graph = build_graph(fake_predict("mel", 0.8), explainer=make_explainer(chain=RunnableLambda(boom)))
    assert "dermatologist" in run_triage(graph, noisy_image())["explanation"].lower()


def test_chat_graph_grounds_and_guards():
    seen = {}
    def fake(x):
        seen.update(x)
        return " grounded answer "
    g = build_chat_graph(chain=RunnableLambda(fake))
    ok = g.invoke({"question": "What is this?", "codes": ["mel", "<script>"]})
    assert ok["answer"] == "grounded answer" and "Melanoma" in seen["context"] and "<script>" not in seen["context"]
    assert "limited" in g.invoke({"question": "x" * 501, "codes": []})["error"]
    assert "enter" in g.invoke({"question": "  ", "codes": []})["error"]
    assert build_chat_graph() is None and known_codes(["nv", "zzz", 5]) == ["nv"]


def test_chat_endpoint_and_rate_limit():
    chat = build_chat_graph(chain=RunnableLambda(lambda x: "fine"))
    c = create_app(predict=fake_predict("nv", 0.9), chat_graph=chat).test_client()
    assert c.post("/api/chat", json={"question": "hi", "codes": ["nv"]}).get_json()["answer"] == "fine"
    assert c.post("/api/chat", json={"question": ""}).status_code == 400
    codes = [c.post("/api/chat", json={"question": "hi"}).status_code for _ in range(12)]
    assert 429 in codes


def test_chat_disabled_without_llm(client):
    assert client.post("/api/chat", json={"question": "hi"}).status_code == 503


def test_normalization_resolved_per_backbone_and_saved(tmp_path):
    pytest.importorskip("timm")
    from skinlesion.data import eval_transform_for
    from skinlesion.model import normalization_for
    m = build_model("timm/tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
    mean, std = normalization_for(m, "timm/tf_efficientnetv2_s.in21k_ft_in1k")
    assert mean == (0.5, 0.5, 0.5) and std == (0.5, 0.5, 0.5)
    save_checkpoint(tmp_path / "e.pt", m, "timm/tf_efficientnetv2_s.in21k_ft_in1k", 64, {}, mean, std)
    _, ckpt = load_checkpoint(tmp_path / "e.pt")
    assert tuple(ckpt["mean"]) == (0.5, 0.5, 0.5)
    t = eval_transform_for(ckpt)
    assert abs(float(t(Image.new("RGB", (80, 80), (255, 255, 255))).max()) - 1.0) < 1e-5  # white -> +1 with 0.5/0.5


def test_old_checkpoint_without_stats_defaults_to_imagenet(tmp_path):
    from skinlesion.data import eval_transform_for
    m = build_model("mobilenet_v3_large", pretrained=False)
    save_checkpoint(tmp_path / "o.pt", m, "mobilenet_v3_large", 64, {})
    ck = torch.load(tmp_path / "o.pt", weights_only=True)
    ck.pop("mean"); ck.pop("std")  # emulate a checkpoint saved before this change
    assert eval_transform_for(ck)(Image.new("RGB", (80, 80))).shape == (3, 64, 64)


# ---------- Grad-CAM ----------
from skinlesion.gradcam import grad_cam, overlay, target_layer, to_data_url


@pytest.mark.parametrize("arch", [
    "mobilenet_v3_large",
    "timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
    "timm/tf_efficientnetv2_s.in21k_ft_in1k",
    "timm/maxvit_tiny_tf_224.in1k",
])
def test_gradcam_works_for_every_shipped_architecture(arch):
    pytest.importorskip("timm")
    model = build_model(arch, pretrained=False)
    cam = grad_cam(model, arch, torch.rand(1, 3, 224, 224), class_idx=4)
    assert cam.shape == (224, 224) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
    assert np.isfinite(cam).all()


def test_gradcam_responds_to_the_class_and_overlay_roundtrips():
    model = build_model("mobilenet_v3_large", pretrained=False)
    x = torch.rand(1, 3, 224, 224)
    a, b = grad_cam(model, "mobilenet_v3_large", x, 0), grad_cam(model, "mobilenet_v3_large", x, 5)
    assert not np.allclose(a, b)
    img = overlay(noisy_image(200), a)
    assert img.size[0] == 384 and to_data_url(img).startswith("data:image/jpeg;base64,")


def test_gradcam_unknown_architecture_is_a_clear_error():
    with pytest.raises(ValueError):
        target_layer(torch.nn.Linear(2, 2), "mystery")


def test_api_includes_heatmap_and_survives_its_failure():
    ok = create_app(predict=fake_predict("nv", 0.9), heatmap_fn=lambda img, code: "data:image/jpeg;base64,AAA")
    r = ok.test_client().post("/api/predict", data={"image": (io.BytesIO(png_bytes(noisy_image())), "x.png")})
    assert r.get_json()["heatmap"].startswith("data:image/jpeg")
    def boom(*a): raise RuntimeError("cam failed")
    bad = create_app(predict=fake_predict("nv", 0.9), heatmap_fn=boom)
    r = bad.test_client().post("/api/predict", data={"image": (io.BytesIO(png_bytes(noisy_image())), "x.png")})
    assert r.status_code == 200 and "heatmap" not in r.get_json()


def test_predictor_heatmap_uses_only_the_chosen_member(tmp_path):
    from skinlesion.predict import Predictor
    for i in range(2):
        save_checkpoint(tmp_path / f"m{i}.pt", build_model("mobilenet_v3_large", pretrained=False),
                        "mobilenet_v3_large", 96, {})
    pred = Predictor(f"{tmp_path / 'm0.pt'},{tmp_path / 'm1.pt'}", device="cpu", cam_member=1)
    assert pred.heatmap(noisy_image(120), "mel").startswith("data:image/jpeg;base64,")
    assert pred.cam_member == 1 and len(pred.members) == 2


def test_low_risk_text_never_reassures():
    for code, p in (("nv", 0.95), ("df", 0.9)):
        out = run_triage(build_graph(fake_predict(code, p)), noisy_image())
        text = out["explanation"].lower()
        assert out["risk"] == "low"
        assert "benign" not in text and "harmless" not in text and "safe" not in text
        assert "not reassurance" in text


def test_pages_state_measured_miss_rate(client):
    from skinlesion.config import PERFORMANCE_NOTE
    assert "14%" in PERFORMANCE_NOTE
    assert b"did not flag about 14%" in client.get("/test").data
