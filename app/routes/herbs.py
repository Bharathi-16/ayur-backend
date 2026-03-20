"""
Herbs Routes — Herb lookup and formulation search
"""
from flask import Blueprint, request, jsonify

herbs_bp = Blueprint('herbs', __name__)

# ── Built-in Herb Database ──
HERBS = [
    {"name": "Ashwagandha", "sanskrit": "Ashwagandha (अश्वगन्धा)", "latin": "Withania somnifera",
     "properties": "Rasayana, Balya, Vajikarana", "dosha": "Vata-Kapha shamaka",
     "uses": "Stress, anxiety, fatigue, immunity, strength", "part": "Root",
     "caution": "Avoid in hyperthyroidism, pregnancy"},
    {"name": "Tulsi", "sanskrit": "Tulasi (तुलसी)", "latin": "Ocimum sanctum",
     "properties": "Deepana, Pachana, Krimighna", "dosha": "Kapha-Vata shamaka",
     "uses": "Respiratory disorders, fever, immunity, stress", "part": "Leaves, seeds",
     "caution": "May affect blood clotting; use cautiously with anticoagulants"},
    {"name": "Turmeric", "sanskrit": "Haridra (हरिद्रा)", "latin": "Curcuma longa",
     "properties": "Krimighna, Varnya, Vishaghna", "dosha": "Tridosha shamaka",
     "uses": "Inflammation, wounds, skin disorders, liver support", "part": "Rhizome",
     "caution": "High doses may aggravate Pitta"},
    {"name": "Brahmi", "sanskrit": "Brahmi (ब्राह्मी)", "latin": "Bacopa monnieri",
     "properties": "Medhya, Rasayana, Hridya", "dosha": "Tridosha shamaka",
     "uses": "Memory, concentration, anxiety, epilepsy", "part": "Whole plant",
     "caution": "May cause GI upset in some; start with low dose"},
    {"name": "Guduchi", "sanskrit": "Guduchi (गुडूची)", "latin": "Tinospora cordifolia",
     "properties": "Rasayana, Deepana, Tridoshahara", "dosha": "Tridosha shamaka",
     "uses": "Immunity, fever, diabetes, liver disorders", "part": "Stem",
     "caution": "May lower blood sugar; monitor with diabetes medication"},
    {"name": "Shatavari", "sanskrit": "Shatavari (शतावरी)", "latin": "Asparagus racemosus",
     "properties": "Rasayana, Balya, Stanyajanana", "dosha": "Vata-Pitta shamaka",
     "uses": "Female reproductive health, lactation, digestion, immunity", "part": "Root",
     "caution": "Avoid in estrogen-sensitive conditions"},
    {"name": "Triphala", "sanskrit": "Triphala (त्रिफला)", "latin": "Haritaki + Bibhitaki + Amalaki",
     "properties": "Rasayana, Rechana, Chakshushya", "dosha": "Tridosha shamaka",
     "uses": "Digestion, constipation, detox, eye health", "part": "Fruit",
     "caution": "Avoid during pregnancy; reduce dose if loose stools"},
    {"name": "Neem", "sanskrit": "Nimba (निम्ब)", "latin": "Azadirachta indica",
     "properties": "Krimighna, Kushthaghna, Raktashodhaka", "dosha": "Kapha-Pitta shamaka",
     "uses": "Skin disorders, blood purification, diabetes, fever", "part": "Leaf, bark, seed",
     "caution": "Excess may aggravate Vata; avoid in debility"},
    {"name": "Amalaki", "sanskrit": "Amalaki (आमलकी)", "latin": "Emblica officinalis",
     "properties": "Rasayana, Chakshushya, Vrishya", "dosha": "Tridosha shamaka",
     "uses": "Vitamin C source, immunity, hair, skin, digestion", "part": "Fruit",
     "caution": "Generally safe; may cause acidity in excess"},
    {"name": "Guggulu", "sanskrit": "Guggulu (गुग्गुलु)", "latin": "Commiphora mukul",
     "properties": "Lekhana, Medohara, Vedanasthapana", "dosha": "Kapha-Vata shamaka",
     "uses": "Cholesterol, obesity, arthritis, inflammation", "part": "Resin",
     "caution": "Avoid in pregnancy, thyroid disorders without supervision"},
]


@herbs_bp.route("/herbs", methods=["GET"])
def herbs_list():
    query = request.args.get("q", "").lower()
    if query:
        results = [h for h in HERBS if query in h["name"].lower()
                   or query in h["sanskrit"].lower()
                   or query in h["uses"].lower()
                   or query in h["dosha"].lower()]
    else:
        results = HERBS
    return jsonify(results)


@herbs_bp.route("/herbs/<name>", methods=["GET"])
def herb_detail(name):
    herb = next((h for h in HERBS if h["name"].lower() == name.lower()), None)
    if not herb:
        return jsonify({"error": "Herb not found"}), 404
    return jsonify(herb)
