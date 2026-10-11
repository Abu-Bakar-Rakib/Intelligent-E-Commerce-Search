import numpy as np

# ===== MODEL CONFIGURATION =====
LLM_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
EMB_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CE_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ===== RETRIEVAL SETTINGS =====
F_ALPHA = 0.6           # weight of semantic score
F_BS = 1.0              # scale of attribute boosts
G_POOL = 10             # candidates sent to cross-encoder
CE_RERANK_MAX = 12      # maximum candidate pairs sent to cross-encoder for fast sub-second latency
G_W = 0.5               # cross-encoder weight in final score
GROUP_TIER = 1.0        # bonus for same-product-group items
COS_MIN = 0.30          # confidence gate: min raw cosine
CE_MIN = -8.0           # confidence gate: min cross-encoder score
PARSER_MODE = "rules+llm"
CE_FP16 = True          # half-precision reranker on GPU

# ===== FILTER SETTINGS =====
BOOST = {"category": 0.15, "brand": 0.20, "color": 0.10, "material": 0.10}
MIN_RESULTS = 3
RELAX_ORDER = ["size", "gender", "price"]
ALLOW_FALLBACK = True

# ===== ATTRIBUTE WEIGHTS =====
ATTR_W = {"color": 0.6, "brand": 0.6, "material": 0.4, "category": 0.3}

# ===== UI SETTINGS =====
PAGE_TITLE = "🛍️ E-Commerce Intelligent Search"
PAGE_ICON = "🔍"
LAYOUT = "wide"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20