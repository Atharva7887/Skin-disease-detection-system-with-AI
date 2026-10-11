"""Patient-education text shown next to a prediction (general information, not advice)."""

CLASS_INFO = {
    "akiec": {
        "title": "Actinic Keratoses",
        "summary": "Actinic keratosis, also known as solar keratosis, is a precancerous growth that develops "
                   "on sun-damaged skin. It appears as rough, scaly patches and can progress into squamous "
                   "cell carcinoma if left untreated.",
    },
    "bcc": {
        "title": "Basal Cell Carcinoma",
        "summary": "Basal cell carcinoma is the most common skin cancer. It usually appears as a pearly or "
                   "waxy bump or a flat, scar-like patch on sun-exposed skin. It grows slowly and rarely "
                   "spreads, but needs treatment to prevent local tissue damage.",
    },
    "bkl": {
        "title": "Benign Keratosis-like Lesions",
        "summary": "This group includes seborrheic keratoses, solar lentigines and lichen-planus-like "
                   "keratoses. They are non-cancerous, often waxy or 'stuck-on' looking growths, though "
                   "they can resemble melanoma and are sometimes biopsied to be sure.",
    },
    "df": {
        "title": "Dermatofibroma",
        "summary": "A dermatofibroma is a common, harmless firm nodule in the skin, often on the legs, "
                   "that dimples when pinched. It usually needs no treatment unless it is symptomatic "
                   "or its appearance changes.",
    },
    "mel": {
        "title": "Melanoma",
        "summary": "Melanoma is the most serious type of skin cancer. It develops in pigment-producing "
                   "cells and can spread to other organs. Early detection greatly improves outcomes, so "
                   "any new, changing, asymmetric or multi-coloured mole should be examined promptly.",
    },
    "nv": {
        "title": "Melanocytic Nevi",
        "summary": "Melanocytic nevi are common moles. Most are harmless, but it is worth watching for "
                   "changes in size, shape, colour or symptoms such as itching or bleeding.",
    },
    "vasc": {
        "title": "Vascular Lesions",
        "summary": "Vascular lesions such as cherry angiomas, angiokeratomas and pyogenic granulomas are "
                   "growths made of blood vessels. They are usually benign, though bleeding or rapidly "
                   "growing lesions should be checked.",
    },
}

PRECAUTIONS = [
    ("Regular Skin Checks", "Keep an eye on any moles or skin abnormalities. Report changes to your doctor."),
    ("Sun Protection", "Wear sunscreen, protective clothing, and seek shade to prevent further damage."),
    ("Avoid Tanning Beds", "UV radiation from tanning beds can increase your risk of skin cancer."),
    ("Healthy Lifestyle", "Eat a balanced diet, exercise regularly, and avoid smoking."),
    ("Medical Advice", "Follow your dermatologist's recommendations for treatment and follow-up appointments."),
    ("Stay Informed", "Educate yourself about your condition and treatment options."),
]
