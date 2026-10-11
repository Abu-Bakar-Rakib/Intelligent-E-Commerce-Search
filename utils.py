import re
import json
import numpy as np
from collections import Counter
from difflib import get_close_matches

# ===== TEXT UTILITIES =====
_STOP = set("a an the for of and or in on with to is it my me i that who this at by from under over below above give show find get want need please can you".split())

def _toks(s):
    s = s.lower().replace("h&m", "hm")
    return [w for w in re.findall(r"[^\W_]+", s) if w not in _STOP]

def _key(query):
    return " ".join(query.lower().split())

def _alnum(s):
    return re.sub(r"[\W_]", "", str(s).lower())

def _word_in(value, text):
    v = str(value).strip().lower()
    return bool(v) and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", text) is not None

def _cat(s):
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[^\W_]+", str(s).lower()))

def _minmax(x):
    x = np.asarray(x, dtype="float64")
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

# ===== MATERIAL EQUIVALENCE & MATCHING =====
MATERIAL_SYNONYMS = {
    "metal": {"metal", "steel", "aluminium", "aluminum", "iron", "brass", "metallic"},
    "steel": {"steel", "metal", "stainless steel", "iron"},
    "aluminum": {"aluminum", "aluminium", "metal"},
    "aluminium": {"aluminium", "aluminum", "metal"},
    "leather": {"leather", "genuine leather", "faux leather", "synthetic leather"},
    "cotton": {"cotton", "cotton blend", "tant cotton"},
    "silk": {"silk", "half silk", "katan silk", "silk blend"},
    "plastic": {"plastic", "pvc", "silicone", "resin", "polycarbonate"},
}

def is_material_match(query_mat, doc_mat):
    qm = _alnum(query_mat)
    dm = _alnum(doc_mat)
    if not qm or not dm:
        return False
    if qm in dm or dm in qm:
        return True
    equiv = MATERIAL_SYNONYMS.get(qm, set())
    return any(e in dm or dm in e for e in equiv)


# ===== CONVERSATIONAL PHRASE CLEANING =====
CONVERSATIONAL_PHRASES = [
    r"\bgive\s+me\s+(?:a\s+|an\s+)?",
    r"\bshow\s+me\s+(?:a\s+|an\s+)?",
    r"\bfind\s+me\s+(?:a\s+|an\s+)?",
    r"\bget\s+me\s+(?:a\s+|an\s+)?",
    r"\bi\s+(?:want|need|am\s+looking\s+for)\s+(?:a\s+|an\s+)?",
    r"\bcan\s+you\s+(?:give|show|find|get)\s+(?:me\s+)?(?:a\s+|an\s+)?",
    r"\bamake\s+(?:dao|dekhao)\s*",
    r"\bamake\b\s*",
    r"\bdekhao\s*",
    r"\bkhujchi\s*",
    r"\bdorkar\s*",
    r"\bchai\s*",
    r"\blagbe\s*",
    r"\bplease\s*",
]

def strip_conversational_phrases(query):
    """Strip conversational filler like 'give me a', 'show me', 'amake dekhao'."""
    q = query
    for pat in CONVERSATIONAL_PHRASES:
        q = re.sub(pat, " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()

# ===== PRICE EXTRACTION =====
_NUM = r"(?:[$₹৳£€]\s*)?(\d+(?:\.\d+)?)\s*(k)?"
_CUR = r"(?:\s*(?:taka|tk|bdt|takar?|৳|rs|inr|rupees?|usd|dollars?|eur|euros?|gbp|pounds?))?"
_LOW = r"(?:under|below|less than|upto|up to|within|maximum|max|cheaper than|not more than|<)"
_HIGH = r"(?:over|above|more than|at least|minimum|min|starting from|>)"
_LOW_POST = r"(?:takar\s+moddhe|taka\s+er\s+niche|taka\s+er\s+kom|tk\s+er\s+niche|tk\s+er\s+kom|takar\s+kom|moddhe|niche|kom)"
_HIGH_POST = r"(?:takar\s+beshi|taka\s+er\s+upore|takar\s+upore|beshi|upore)"

def _val(num, k):
    v = float(num) * (1000 if k else 1)
    return int(v) if v.is_integer() else v

def extract_price(query):
    q = query.lower().replace(",", "")
    
    # 1. Price Range: between X and Y
    m = re.search(rf"(?:between|from)\s+{_NUM}{_CUR}\s*(?:and|to|-)\s*{_NUM}{_CUR}", q)
    if m:
        a, b = _val(m.group(1), m.group(2)), _val(m.group(3), m.group(4))
        return min(a, b), max(a, b)
    
    # 2. Prefix max price: under 1000 taka / below 1000 / under tk 1000
    m = re.search(rf"{_LOW}\s*(?:taka|tk|bdt|৳)?\s*{_NUM}{_CUR}", q)
    if m:
        return None, _val(m.group(1), m.group(2))
    
    # 3. Postfix max price: 1000 takar moddhe / 1000 tk er niche
    m = re.search(rf"{_NUM}{_CUR}\s*(?:er\s+)?{_LOW_POST}", q)
    if m:
        return None, _val(m.group(1), m.group(2))
    
    # 4. Prefix min price: over 1000 taka / above 1000
    m = re.search(rf"{_HIGH}\s*(?:taka|tk|bdt|৳)?\s*{_NUM}{_CUR}", q)
    if m:
        return _val(m.group(1), m.group(2)), None
        
    # 5. Postfix min price: 1000 takar beshi
    m = re.search(rf"{_NUM}{_CUR}\s*(?:er\s+)?{_HIGH_POST}", q)
    if m:
        return _val(m.group(1), m.group(2)), None
        
    return None, None

def strip_price_phrase(query):
    q = query.lower().replace(",", "")
    q = re.sub(rf"(?:between|from)\s+{_NUM}{_CUR}\s*(?:and|to|-)\s*{_NUM}{_CUR}", " ", q, flags=re.I)
    q = re.sub(rf"{_LOW}\s*(?:taka|tk|bdt|৳)?\s*{_NUM}{_CUR}", " ", q, flags=re.I)
    q = re.sub(rf"{_NUM}{_CUR}\s*(?:er\s+)?{_LOW_POST}", " ", q, flags=re.I)
    q = re.sub(rf"{_HIGH}\s*(?:taka|tk|bdt|৳)?\s*{_NUM}{_CUR}", " ", q, flags=re.I)
    q = re.sub(rf"{_NUM}{_CUR}\s*(?:er\s+)?{_HIGH_POST}", " ", q, flags=re.I)
    # Also strip stray currency mentions
    q = re.sub(r"\b(?:taka|tk|bdt|৳)\b", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()

# ===== SIZE HANDLING =====
STD_SIZES = {"xs", "s", "m", "l", "xl", "xxl", "xxxl"}
_SZ = r"(xxxl|xxl|xl|xs|s|m|l|\d{1,2})"

def _find_size(q):
    q = q.lower()
    for pat in (rf"\bsize\s*[:\-]?\s*{_SZ}\b", rf"\b{_SZ}\s+size\b"):
        m = re.search(pat, q)
        if m:
            return m.group(1).upper()
    for t in q.split():
        t = t.strip(".,;:!?()")
        if t in STD_SIZES:
            return t.upper()
    return None

def strip_size_phrase(query):
    q = query.lower()
    q = re.sub(rf"\bsize\s*[:\-]?\s*{_SZ}\b", " ", q)
    q = re.sub(rf"\b{_SZ}\s+size\b", " ", q)
    q = " ".join(t for t in q.split() if t.strip(".,;:!?()") not in STD_SIZES)
    return re.sub(r"\s+", " ", q).strip()

# ===== JSON PARSING =====
def parse_json(text):
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    return data

# ===== GENDER & RECIPIENT DETECTION =====
GENDER_CUES = {
    "men": [
        "men", "mens", "men's", "man", "male", "gents", "chele", "cheleder", "chheleder", "purush",
        "father", "fathers", "father's", "dad", "daddy", "baba", "babar", "abba", "abbar",
        "husband", "bor", "brother", "bhai", "bhaiya", "grandfather", "dada", "nana"
    ],
    "women": [
        "women", "womens", "women's", "woman", "female", "ladies", "lady", "meye", "meyeder", "mohila",
        "mother", "mothers", "mother's", "mom", "mommy", "ma", "maa", "mayer", "amma", "ammar",
        "wife", "bou", "sister", "bon", "apu", "grandmother", "dadi", "nani"
    ],
    "boys": ["boy", "boys", "balok", "baccha chele", "chotto chele", "kid boy", "son", "cheler"],
    "girls": ["girl", "girls", "balika", "baccha meye", "chotto meye", "kid girl", "daughter", "meyetir"],
    "unisex": ["unisex", "sobar", "shobar"]
}

FATHER_CUES = ["father", "fathers", "father's", "dad", "daddy", "baba", "babar", "abba", "abbar", "grandfather", "dada", "nana"]
MOTHER_CUES = ["mother", "mothers", "mother's", "mom", "mommy", "ma", "maa", "mayer", "amma", "ammar", "grandmother", "dadi", "nani"]
GIFT_CUES = ["gift", "gifts", "present", "presents", "upohar", "item for"]

def _find_gender(q):
    for g in ("women", "men", "girls", "boys", "unisex"):
        if any(_word_in(c, q) for c in GENDER_CUES[g]):
            return g
    return None

def is_father_gift_query(q):
    q_lower = q.lower()
    has_father = any(_word_in(c, q_lower) for c in FATHER_CUES)
    has_gift = any(_word_in(g, q_lower) for g in GIFT_CUES) or "for" in q_lower or "jonno" in q_lower
    return has_father and has_gift

# ===== BANGLA/BANGLISH NORMALIZATION =====
ALIASES = {
    "juta": "shoes", "jutha": "shoes", "joota": "shoes", "jutta": "shoes", "jooto": "shoes", "juto": "shoes",
    "chappal": "sandals", "chapal": "sandals", "chappals": "sandals", "slipper": "sandals", "slippers": "sandals",
    "sandal": "sandals", "keds": "sneakers", "moja": "socks",
    "sari": "saree", "shari": "saree", "sarees": "saree", "punjabi": "panjabi",
    "pant": "trousers", "pants": "trousers", "tshirt": "t-shirt",
    "tupi": "cap", "topi": "cap", "ghori": "watch", "ghadi": "watch",
    "chosma": "sunglasses", "chashma": "sunglasses", "chasma": "sunglasses", "shades": "sunglasses",
    "baag": "bag", "beg": "bag", "byag": "bag", "thola": "bag",
    "chata": "umbrella", "chhata": "umbrella", "chaata": "umbrella", "chhaata": "umbrella",
    "mobile": "smartphone", "mobail": "smartphone", "pankha": "fan",
    "kolom": "pen", "kalam": "pen", "khata": "notebook", "khatha": "notebook", "boi": "book",
    "lal": "red", "nil": "blue", "sada": "white", "shada": "white", "kalo": "black", "kala": "black",
    "shobuj": "green", "sabuj": "green", "sobuj": "green", "holud": "yellow", "golapi": "pink",
    "chele": "boys", "cheler": "boys", "meye": "girls", "meyer": "girls", "meyeder": "girls",
    "dam": "", "daam": "",
    "botol": "bottle", "botal": "bottle",
}

PHRASES = [
    ("t shirt", "t-shirt"),
    ("mobile cover", "phone case"),
    ("phone cover", "phone case"),
    ("mobile case", "phone case"),
    ("panir botol", "water bottle"),
    ("pani botol", "water bottle"),
]

# ===== FUZZY SPELL CORRECTION =====
# Comprehensive vocabulary built from the entire product catalog
SPELL_VOCAB = [
    # ── Gender / Person ──
    "women", "men", "boys", "girls", "unisex", "female", "male",
    "ladies", "woman", "man", "kids", "children", "baby", "adult",

    # ── Clothing & Apparel ──
    "shirt", "t-shirt", "tshirt", "polo", "top", "blouse", "tunic",
    "dress", "gown", "frock", "skirt", "lehenga", "kurti", "kameez",
    "salwar", "churidar", "anarkali", "saree", "shari", "jamdani",
    "panjabi", "punjabi", "fatua", "lungi", "gamcha", "jacket",
    "blazer", "coat", "hoodie", "sweater", "cardigan", "sweatshirt",
    "trousers", "pants", "jeans", "chinos", "shorts", "leggings",
    "tracksuit", "jogger", "uniform", "jersey", "overalls",
    "underwear", "innerwear", "bra", "lingerie", "nightwear", "pajama",
    "socks", "stockings", "tights",

    # ── Footwear ──
    "shoes", "sneakers", "boots", "sandals", "heels", "slippers",
    "loafers", "oxfords", "moccasins", "flats", "wedges", "platforms",
    "flip-flops", "clogs", "espadrilles", "stilettos",

    # ── Accessories ──
    "bag", "handbag", "purse", "clutch", "tote", "backpack",
    "satchel", "messenger", "duffel", "wallet", "cardholder",
    "belt", "suspenders", "scarf", "shawl", "stole", "dupatta",
    "cap", "hat", "beanie", "turban", "headband",
    "sunglasses", "glasses", "goggles", "spectacles",
    "watch", "bracelet", "necklace", "earrings", "ring", "anklet",
    "brooch", "hairpin", "hairband", "bangle", "pendant", "chain",
    "umbrella", "gloves", "mittens", "tie", "bowtie",

    # ── Electronics ──
    "laptop", "notebook", "computer", "desktop", "pc", "imac",
    "smartphone", "phone", "mobile", "tablet", "ipad",
    "headphones", "earphones", "earbuds", "airpods",
    "speaker", "soundbar", "amplifier",
    "television", "tv", "monitor", "display", "projector",
    "camera", "webcam", "dslr", "mirrorless", "lens",
    "keyboard", "mouse", "mousepad", "joystick", "gamepad",
    "charger", "adapter", "cable", "usb", "powerbank",
    "router", "modem", "smartwatch", "fitness", "band",
    "smartplug", "bulb", "fan", "cooler", "heater",
    "printer", "scanner", "microphone",
    "ssd", "harddisk", "pendrive", "ram", "processor",

    # ── Food & Grocery ──
    "rice", "flour", "sugar", "salt", "oil", "butter", "ghee",
    "milk", "yogurt", "cheese", "paneer", "egg", "eggs",
    "chicken", "beef", "mutton", "fish", "prawn", "shrimp",
    "vegetable", "fruit", "onion", "garlic", "ginger", "tomato",
    "potato", "carrot", "spinach", "cucumber", "pepper",
    "chilli", "turmeric", "cumin", "coriander", "spice",
    "lentil", "dal", "bean", "chickpea", "noodle", "pasta",
    "bread", "biscuit", "cookie", "cake", "pastry",
    "tea", "coffee", "juice", "water", "soda", "drink",
    "honey", "jam", "sauce", "pickle", "vinegar",
    "chocolate", "candy", "sweets", "icecream", "dessert",
    "snack", "chips", "popcorn",

    # ── Beauty & Personal Care ──
    "shampoo", "conditioner", "serum", "moisturizer", "lotion",
    "sunscreen", "facewash", "toner", "scrub", "mask",
    "lipstick", "lipbalm", "eyeliner", "mascara", "eyeshadow",
    "foundation", "concealer", "blush", "highlighter", "kajal",
    "perfume", "attar", "deodorant", "cologne",
    "soap", "bodywash", "toothbrush", "toothpaste", "mouthwash",
    "razor", "shaver", "trimmer", "hairdryer", "straightener",
    "mehendi", "henna", "nail", "polish", "remover",

    # ── Home & Kitchen ──
    "bed", "mattress", "pillow", "blanket", "bedsheet", "comforter",
    "curtain", "cushion", "sofa", "chair", "table", "desk", "shelf",
    "lamp", "light", "bulb", "candle",
    "pan", "pot", "cooker", "kettle", "blender", "mixer",
    "plate", "bowl", "cup", "mug", "glass", "spoon", "fork",
    "knife", "chopper", "grater", "strainer", "colander",
    "refrigerator", "fridge", "microwave", "oven", "toaster",
    "washing", "machine", "iron", "vacuum",
    "bucket", "mop", "broom", "dustbin", "cleaner",
    "mosquito", "repellent", "coil", "spray",

    # ── Sports & Fitness ──
    "gym", "fitness", "yoga", "running", "cycling", "swimming",
    "football", "cricket", "badminton", "tennis", "basketball",
    "dumbbell", "barbell", "treadmill", "mat", "gloves",
    "helmet", "jersey", "trackpants",

    # ── Books & Stationery ──
    "book", "novel", "textbook", "dictionary", "notebook",
    "pen", "pencil", "marker", "highlighter", "eraser",
    "ruler", "stapler", "scissors", "tape", "glue", "folder",

    # ── Travel & Outdoor ──
    "luggage", "suitcase", "trolley", "travel", "camping",
    "trekking", "hiking", "tent", "sleeping", "flask",
    "bottle", "thermos", "pillow", "organizer",

    # ── Brands (Common) ──
    "nike", "adidas", "puma", "reebok", "vans", "converse", "fila",
    "levi", "levis", "zara", "hm", "uniqlo", "gucci", "prada",
    "samsung", "apple", "sony", "lg", "xiaomi", "oneplus", "oppo",
    "realme", "asus", "dell", "lenovo", "hp", "acer",
    "logitech", "philips", "anker", "jbl", "bose", "sennheiser",
    "fossil", "casio", "titan", "rolex", "seiko",
    "aarong", "yellow", "bata", "apex", "pran", "radhuni",
    "maybelline", "loreal", "garnier", "neutrogena", "nivea",
    "vaseline", "dove", "lux", "lifebuoy",

    # ── Colors ──
    "red", "blue", "black", "white", "green", "yellow", "pink",
    "purple", "orange", "brown", "grey", "gray", "golden", "silver",
    "maroon", "beige", "cream", "navy", "cyan", "violet", "indigo",
    "multicolor", "transparent", "clear", "floral", "printed",

    # ── Materials ──
    "cotton", "polyester", "silk", "wool", "linen", "denim",
    "leather", "suede", "canvas", "nylon", "spandex", "velvet",
    "chiffon", "georgette", "satin", "knit", "fleece",
    "plastic", "metal", "steel", "aluminium", "glass", "rubber",
    "wood", "bamboo", "ceramic", "foam", "memory",

    # ── Price / Modifier words ──
    "cheap", "affordable", "budget", "expensive", "premium",
    "best", "good", "quality", "original", "genuine", "branded",
    "comfortable", "lightweight", "waterproof", "wireless",
    "formal", "casual", "party", "office", "outdoor", "sport",
    "summer", "winter", "latest", "new", "classic", "modern",
    "small", "medium", "large", "extra", "mini", "portable",
    "under", "below", "above", "upto", "between",

    # ── Gifts & Persona words ──
    "gift", "gifts", "present", "presents", "item", "items",
    "father", "mother", "dad", "mom", "brother", "sister",
    "husband", "wife", "son", "daughter", "mature", "adult",
    "baba", "babar", "abba", "abbar", "dada", "nana",
]

COMMON_WORDS = {
    "for", "the", "and", "with", "from", "into", "onto", "over", "your", "mine", "them", "this", "that", "these", "those",
    "give", "show", "find", "need", "want", "have", "some", "good", "best", "like",
    "baba", "babar", "abba", "abbar", "mama", "kaka", "chacha", "fupu", "khala", "nana", "dada", "dadi", "nani",
    "bhai", "bhaiya", "bon", "apu", "bou", "bor", "shami", "meye", "chele", "baccha",
    "jonno", "moddhe", "niche", "beshi", "kom", "er", "te", "amake", "dekhao", "dao", "koto", "dam",
    "taka", "takar"
}

from functools import lru_cache
SPELL_VOCAB_SET = set(SPELL_VOCAB) | COMMON_WORDS

@lru_cache(maxsize=2048)
def fuzzy_correct(query: str) -> str:
    """Correct misspelled English words in query using fast fuzzy matching."""
    words = query.split()
    corrected = []
    for word in words:
        w_lower = word.lower()
        # Skip short words, numbers, Bangla characters
        if len(w_lower) <= 2 or not w_lower.isascii() or w_lower.isdigit():
            corrected.append(word)
            continue
        # If the word is already in vocab, common words, or aliases, keep it as-is (O(1) set lookup)
        if w_lower in SPELL_VOCAB_SET or w_lower in ALIASES or w_lower in COMMON_WORDS:
            corrected.append(word)
            continue
        # Try to find a close match (cutoff=0.75 = 75% similarity required)
        matches = get_close_matches(w_lower, SPELL_VOCAB, n=1, cutoff=0.75)
        if matches:
            corrected.append(matches[0])
        else:
            corrected.append(word)
    return " ".join(corrected)

@lru_cache(maxsize=2048)
def _alias_pass(query):
    # Step 1: Fuzzy-correct the raw query before any other processing
    query = fuzzy_correct(query)
    q = " " + query.lower() + " "
    for a, b in PHRASES:
        q = q.replace(" " + a + " ", " " + b + " ")
    return " ".join(w for w in (ALIASES.get(t, t) for t in q.split()) if w)

# ===== BUILD BM25 INDEX =====
def build_bm25_index(texts):
    """Build BM25 components from catalog texts."""
    docs = [_toks(t) for t in texts]
    df = Counter(w for d in docs for w in set(d))
    vocab_words = sorted(df)
    vocab = {w: i for i, w in enumerate(vocab_words)}
    
    n_docs = len(docs)
    tf = np.zeros((n_docs, len(vocab)), dtype="float32")
    for i, d in enumerate(docs):
        for w, c in Counter(d).items():
            tf[i, vocab[w]] = c
    
    dl = np.array([len(d) for d in docs], dtype="float32")
    dfv = np.array([df[w] for w in vocab_words], dtype="float32")
    idf = np.log(1 + (n_docs - dfv + 0.5) / (dfv + 0.5))
    
    return {"vocab": vocab, "tf": tf, "dl": dl, "idf": idf, "avg_len": float(np.mean(dl))}

def bm25_scores(query, bm25_index, k1=1.5, b=0.75):
    """Compute BM25 scores for a query."""
    vocab = bm25_index["vocab"]
    tf = bm25_index["tf"]
    dl = bm25_index["dl"]
    idf = bm25_index["idf"]
    avg_len = bm25_index["avg_len"]
    
    idx = [vocab[w] for w in _toks(query) if w in vocab]
    if not idx:
        return np.zeros(len(dl))
    
    tf_q = tf[:, idx]
    norm = tf_q * (k1 + 1) / (tf_q + k1 * (1 - b + b * dl / avg_len)[:, None])
    return (norm * idf[idx]).sum(1)