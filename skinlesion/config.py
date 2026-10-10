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
