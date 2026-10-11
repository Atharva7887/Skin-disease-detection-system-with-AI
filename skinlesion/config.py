"""Shared constants for training and inference."""

# Alphabetical, matching LabelEncoder / sorted() order so indices are stable.
CLASS_CODES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DISCLAIMER = (
    "This tool is a research prototype and is NOT a medical device. "
    "It cannot diagnose skin cancer. Consult a qualified dermatologist."
)

# Measured on the independent ISIC 2018 Task 3 test set (1,511 images; 171 melanomas, 908 nevi)
# with the shipped 3-model ensemble (8-view TTA) and the triage rule in triage.py. The app reads the live
# numbers from webapp/model_card.json; this is only the fallback. Update if models change.
PERFORMANCE_NOTE = (
    "In testing on 1,511 dermoscopy images, this tool did not flag about 10% of melanomas, and flagged "
    "about 14% of ordinary moles. A result that is not flagged is not a reassurance."
)
