# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    return text.lower().replace("-", "_").replace(" ", "_")


# ==============================
# FORMAT NAME (DISPLAY)
# ==============================
def format_name(name):
    return name.replace("__", " - ").replace("_", " ")


# ==============================
# SEVERITY LOGIC
# ==============================
def get_severity(conf):
    if conf > 85:
        return "Severe 🔴"
    elif conf > 60:
        return "Moderate 🟠"
    else:
        return "Mild 🟢"


# ==============================
# NORMALIZE MODEL OUTPUT
# ==============================
def normalize_model_output(text):
    return text.lower()\
        .replace(" - ", "__")\
        .replace("-", "_")\
        .replace(" ", "_")\
        .replace("___", "__")


# ==============================
# MAIN FUNCTION
# ==============================
def get_cure(disease, confidence=None):

    disease_clean = normalize_model_output(disease)

    cures = {
        # 🌾 GENERAL
        "anthracnose": {
            "pesticide": "Fungicide",
            "chemical": "Carbendazim",
            "dosage": "1g/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Use disease-free seeds"
        },

        "bacterialblight": {
            "pesticide": "Bactericide",
            "chemical": "Copper Oxychloride",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Garlic extract",
            "cure": "Avoid water stagnation"
        },

        "brownspot": {
            "pesticide": "Fungicide",
            "chemical": "Mancozeb",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Compost spray",
            "cure": "Improve soil nutrition"
        },

        "leaf_crinckle": {
            "pesticide": "None",
            "chemical": "-",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Remove infected plants (viral)"
        },

        "powdery_mildew": {
            "pesticide": "Fungicide",
            "chemical": "Sulfur",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Milk spray",
            "cure": "Reduce humidity"
        },

        "yellow_mosaic": {
            "pesticide": "Insecticide",
            "chemical": "Imidacloprid",
            "dosage": "0.5ml/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Control whiteflies"
        },

        "healthy": {
            "pesticide": "None",
            "chemical": "-",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Plant is healthy 🌱"
        },

        # 🌶️ PEPPER
        "pepper__bell__bacterial_spot": {
            "pesticide": "Bactericide",
            "chemical": "Copper Oxychloride",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Avoid overhead watering"
        },

        "pepper__bell__healthy": {
            "pesticide": "None",
            "chemical": "-",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Healthy plant 🌱"
        },

        # 🥔 POTATO
        "potato__early_blight": {
            "pesticide": "Fungicide",
            "chemical": "Mancozeb",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Remove infected leaves"
        },

        "potato__late_blight": {
            "pesticide": "Fungicide",
            "chemical": "Chlorothalonil",
            "dosage": "2.5g/L",
            "schedule": "Every 5 days",
            "organic": "Baking soda",
            "cure": "Improve drainage"
        },

        "potato__healthy": {
            "pesticide": "None",
            "chemical": "-",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Healthy plant 🌱"
        },

        # 🍅 TOMATO
        "tomato_bacterial_spot": {
            "pesticide": "Bactericide",
            "chemical": "Copper Hydroxide",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Garlic spray",
            "cure": "Avoid overhead irrigation"
        },

        "tomato_early_blight": {
            "pesticide": "Fungicide",
            "chemical": "Mancozeb",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Remove infected leaves"
        },

        "tomato_late_blight": {
            "pesticide": "Fungicide",
            "chemical": "Chlorothalonil",
            "dosage": "2.5g/L",
            "schedule": "Every 5 days",
            "organic": "Baking soda",
            "cure": "Improve airflow"
        },

        "tomato_leaf_mold": {
            "pesticide": "Fungicide",
            "chemical": "Copper fungicide",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Milk spray",
            "cure": "Reduce humidity"
        },

        "tomato_septoria_leaf_spot": {
            "pesticide": "Fungicide",
            "chemical": "Mancozeb",
            "dosage": "2g/L",
            "schedule": "Weekly",
            "organic": "Neem oil",
            "cure": "Remove infected leaves"
        },

        "tomato_spider_mites_two_spotted_spider_mite": {
            "pesticide": "Insecticide",
            "chemical": "Abamectin",
            "dosage": "1ml/L",
            "schedule": "Every 5 days",
            "organic": "Neem oil",
            "cure": "Spray underside"
        },

        "tomato_target_spot": {
            "pesticide": "Fungicide",
            "chemical": "Azoxystrobin",
            "dosage": "1g/L",
            "schedule": "Weekly",
            "organic": "Compost extract",
            "cure": "Improve airflow"
        },

        "tomato_healthy": {
            "pesticide": "None",
            "chemical": "-",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Healthy plant 🌱"
        }
    }

    print("MODEL:", disease)
    print("CLEAN:", disease_clean)

    # ✅ EXACT MATCH (FIXED)
    result = cures.get(disease_clean)

    # ✅ PARTIAL MATCH
    if result is None:
        for key in cures:
            if key in disease_clean:
                result = cures[key]
                break

    # ✅ FINAL FALLBACK
    if result is None:
        result = {
            "pesticide": "General",
            "chemical": "Consult expert",
            "dosage": "-",
            "schedule": "-",
            "organic": "-",
            "cure": "Consult agricultural expert"
        }

    conf_percent = float(confidence ) if confidence else 0.0

    return {
        "name": format_name(disease),
        "severity": get_severity(conf_percent),
        "pesticide": result["pesticide"],
        "chemical": result["chemical"],
        "dosage": result["dosage"],
        "schedule": result["schedule"],
        "organic": result["organic"],
        "cure": result["cure"],
        "confidence": conf_percent
    }
