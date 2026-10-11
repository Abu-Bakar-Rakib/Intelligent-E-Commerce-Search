import os
import re
import json
import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from config import (
    F_ALPHA, F_BS, G_POOL, G_W, GROUP_TIER, COS_MIN, CE_MIN, CE_RERANK_MAX,
    BOOST, MIN_RESULTS, RELAX_ORDER, ALLOW_FALLBACK, ATTR_W, PARSER_MODE
)
from utils import (
    _toks, _key, _alnum, _word_in, _cat, _minmax, extract_price,
    strip_price_phrase, strip_size_phrase, _find_size, _find_gender,
    parse_json, _alias_pass, build_bm25_index, bm25_scores, GENDER_CUES,
    strip_conversational_phrases, is_father_gift_query, is_material_match
)

MATURE_MEN_CATEGORIES = {
    "wallet", "watch", "formal shoes", "shirt", "grooming", "fragrance",
    "panjabi", "sunglasses", "backpack", "sandals"
}

# Cache for parsed intents, prepared data, cross-encoder scores, and searches
INTENT_CACHE = {}
PREP_CACHE = {}
CE_CACHE = {}
SEARCH_CACHE = {}

class SearchEngine:
    def __init__(self, catalog, models):
        self.catalog = catalog.copy()
        self.models = models
        self.device = models.device
        
        # Build catalog metadata
        self._build_catalog_metadata()
        self._build_embeddings()
        self._build_bm25_index()
        self._build_groups()
        self._warmup()
    
    def _warmup(self):
        """Warm up PyTorch JIT kernels so even the first query is sub-second"""
        try:
            embedder = self.models.get_embedder()
            ce = self.models.get_reranker()
            with torch.inference_mode():
                embedder.encode(["warmup"], show_progress_bar=False)
                ce.predict([("warmup", "warmup")], show_progress_bar=False)
        except Exception:
            pass
    
    def _build_catalog_metadata(self):
        """Prepare catalog arrays for fast filtering"""
        self.n_docs = len(self.catalog)
        self.price = self.catalog["price"].to_numpy(dtype="float64")
        self.gender = self.catalog["gender"].to_numpy()
        self.size_sets = self.catalog["size_set"].tolist()
        self.cat_norm = [_cat(c) for c in self.catalog["category"]]
        self.cat_tok = [set(c.split()) for c in self.cat_norm]
        self.brand_n = np.array([_alnum(b) for b in self.catalog["brand"]])
        self.color_n = np.array([_alnum(c) for c in self.catalog["color"]])
        self.mat_n = [_alnum(m) for m in self.catalog["material"]]
    
    def _build_embeddings(self):
        """Generate or load cached embeddings for all products"""
        emb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog_embeddings.npy")
        texts = self.catalog["text"].tolist()
        
        # Fast disk cache check
        if os.path.exists(emb_file):
            try:
                cached = np.load(emb_file)
                if cached.shape[0] == len(texts):
                    self.embeddings = cached.astype("float32")
                    return
            except Exception:
                pass
        
        embedder = self.models.get_embedder()
        with torch.inference_mode():
            self.embeddings = embedder.encode(
                texts, batch_size=64, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False
            ).astype("float32")
        
        try:
            np.save(emb_file, self.embeddings)
        except Exception:
            pass
    
    def _build_bm25_index(self):
        """Build BM25 index"""
        texts = self.catalog["text"].tolist()
        self.bm25_index = build_bm25_index(texts)
    
    def _build_groups(self):
        """Build product groups for intent matching"""
        self.groups = {
            "footwear": (
                {"shoe", "shoes", "footwear", "sneaker", "sneakers", "sandal", "sandals", "boot", "boots",
                 "clog", "clogs", "loafer", "loafers", "heel", "heels", "flats"},
                ["running shoes", "sneakers", "formal shoes", "sandals", "walking shoes", "boots",
                 "clogs", "training shoes", "school shoes", "loafers", "heels", "flats"]
            ),
            "bags": (
                {"bag", "bags", "backpack", "backpacks", "rucksack", "handbag", "suitcase", "luggage",
                 "trolley", "duffel", "tote"},
                ["backpack", "duffel bag", "travel bag", "handbag", "gym bag", "sling bag", "suitcase"]
            ),
            "watches": ({"watch", "watches", "smartwatch", "wristwatch"}, ["watch", "smartwatch", "fitness band"]),
            "eyewear": ({"sunglasses", "glasses", "goggles"}, ["sunglasses", "goggles"]),
            "ethnic": ({"panjabi", "saree", "kurti", "salwar", "kameez"},
                      ["panjabi", "saree", "kurti", "salwar kameez"]),
            "tops": ({"shirt", "shirts", "tee", "tees", "polo", "hoodie", "hoodies", "sweater", "cardigan"},
                    ["t-shirt", "shirt", "hoodie", "sweater"]),
            "bottoms": ({"jeans", "trousers", "shorts", "leggings", "chinos"},
                       ["jeans", "trousers", "shorts", "leggings", "track pants"]),
            "outerwear": ({"jacket", "jackets", "raincoat", "poncho", "coat"}, ["jacket", "raincoat"]),
            "audio": ({"headphone", "headphones", "earbuds", "earphone", "earphones", "speaker", "speakers"},
                     ["headphones", "earbuds", "speaker"]),
            "bottles": (
                {"bottle", "bottles", "flask", "flasks", "mug", "mugs", "thermos", "waterbottle", "water-bottle"},
                ["water bottle", "travel mug", "thermos", "mug"]
            ),
            "wallets": ({"wallet", "wallets", "purse", "clutch", "cardholder"}, ["wallet"]),
            "phone_acc": ({"case", "cover", "protector"}, ["phone case", "screen protector"]),
            "appliances": (
                {"blender", "kettle", "toaster", "iron", "oven", "cooker", "fryer", "appliance"},
                ["blender", "kettle", "air fryer", "rice cooker", "coffee maker", "microwave oven", "toaster", "appliance"]
            ),
            "toys": ({"toy", "toys", "puzzle", "puzzles", "lego", "blocks"}, ["toy", "board game"]),
        }
        
        self.group_masks = {}
        for g, (_, cats) in self.groups.items():
            self.group_masks[g] = self.catalog["category"].isin(cats).to_numpy()
    
    def detect_groups(self, q):
        """Detect product groups from query"""
        toks = set(re.findall(r"[a-z0-9]+", q.lower()))
        names = [g for g, (trig, _) in self.groups.items() if toks & trig]
        if not names:
            return None, []
        mask = np.zeros(self.n_docs, dtype=bool)
        for g in names:
            mask |= self.group_masks[g]
        return mask, names
    
    def parse_query(self, query):
        """Parse user query into structured intent"""
        key = _key(query)
        if key in INTENT_CACHE:
            return INTENT_CACHE[key]
        
        q = query.lower()
        brands = sorted({b.lower() for b in self.catalog["brand"]}, key=len, reverse=True)
        colors = sorted({c.lower() for c in self.catalog["color"]}, key=len, reverse=True)
        mats = sorted({m.lower() for m in self.catalog["material"]}, key=len, reverse=True)
        cats = sorted({str(c).lower() for c in self.catalog["category"]}, key=lambda c: len(_cat(c)), reverse=True)
        
        intent = {
            "category": None,
            "brand": None,
            "size": None,
            "color": None,
            "material": None,
            "gender": None,
            "min_price": None,
            "max_price": None,
            "requirement": None
        }
        
        # Find category
        qn = " " + _cat(query) + " "
        for c in cats:
            if " " + _cat(c) + " " in qn:
                intent["category"] = c
                break
        
        # Find gender
        intent["gender"] = _find_gender(q)
        
        # Find size
        intent["size"] = _find_size(q)
        
        # Find color
        intent["color"] = next((c for c in colors if _word_in(c, q)), None)
        
        # Find material
        intent["material"] = next((m for m in mats if _word_in(m, q)), None)
        
        # Find brand
        qtoks = {_alnum(w) for w in q.split()}
        intent["brand"] = next((b for b in brands if _word_in(b, q) or (" " not in b and _alnum(b) in qtoks)), None)
        
        # Avoid color/brand confusion
        if intent["brand"] and intent["color"] and _alnum(intent["brand"]) == _alnum(intent["color"]):
            intent["brand"] = None
        
        # Extract prices
        min_p, max_p = extract_price(query)
        intent["min_price"] = min_p
        intent["max_price"] = max_p
        
        # Recipient/Father gift detection
        intent["is_father_gift"] = is_father_gift_query(query)
        if intent["is_father_gift"] and not intent["gender"]:
            intent["gender"] = "men"
        
        INTENT_CACHE[key] = intent
        return intent
    
    def retrieval_query(self, q, intent=None):
        """Clean query for retrieval: no price, no size, no conversational fillers"""
        a = strip_price_phrase(q) or q
        a = strip_size_phrase(a) or a
        cleaned = strip_conversational_phrases(a) or a
        if intent and intent.get("is_father_gift") and not intent.get("category"):
            return "gift for father mature men wallet watch formal shirt electric shaver attar leather shoes"
        return cleaned
    
    def prepare_retrieval(self, query, intent):
        """Prepare BM25 & cosine scores with RRF-based hybrid fusion"""
        key = _key(query)
        if key in PREP_CACHE:
            return PREP_CACHE[key]
        
        q_clean = self.retrieval_query(query, intent)
        embedder = self.models.get_embedder()
        
        # --- Semantic (cosine) scores ---
        with torch.inference_mode():
            qv = embedder.encode([q_clean], normalize_embeddings=True, convert_to_numpy=True,
                                show_progress_bar=False).astype("float32")[0]
        cos = self.embeddings @ qv
        
        # --- Lexical (BM25) scores ---
        bm = bm25_scores(q_clean, self.bm25_index)
        
        # --- Reciprocal Rank Fusion (RRF) ---
        # RRF is provably better than linear blending because it's
        # immune to score scale differences between BM25 and cosine.
        # Formula: rrf(d) = Σ 1 / (k + rank(d))  where k=60 is standard
        K = 60
        cos_rank = np.argsort(np.argsort(-cos))   # rank 0=best
        bm_rank  = np.argsort(np.argsort(-bm))    # rank 0=best
        rrf = (1.0 / (K + cos_rank + 1)) + (1.0 / (K + bm_rank + 1))
        
        # --- Attribute boosts (on top of RRF) ---
        boost = self._make_boost(intent)
        
        # Apply filters with relaxation
        active = list(RELAX_ORDER)
        relaxed = []
        while True:
            mask = self._make_mask(intent, active)
            if mask.sum() >= MIN_RESULTS or not active:
                break
            relaxed.append(active.pop(0))
        
        prep = {
            "cos": _minmax(cos),
            "bm": _minmax(bm),
            "rrf": rrf,                    # ← new: RRF hybrid score
            "raw_cos": cos,
            "raw_bm": bm,
            "boost": boost,
            "mask": mask,
            "relaxed": relaxed,
            "q_clean": q_clean
        }
        PREP_CACHE[key] = prep
        return prep
    
    def _make_mask(self, intent, active):
        """Create filter mask"""
        m = np.ones(self.n_docs, dtype=bool)
        
        if "size" in active and intent.get("size"):
            sz = str(intent["size"]).upper()
            m &= np.array([(not s) or (sz in s) for s in self.size_sets])
        
        if "gender" in active and intent.get("gender"):
            m &= (self.gender == intent["gender"]) | (self.gender == "unisex")
        
        if "price" in active:
            if intent.get("max_price") is not None:
                m &= self.price <= intent["max_price"]
            if intent.get("min_price") is not None:
                m &= self.price >= intent["min_price"]
        
        # When looking for gifts for father/mature adult men, strictly exclude juvenile/kids items
        if intent.get("is_father_gift"):
            juvenile = self.catalog["text"].str.contains(r"\b(?:kid|kids|baby|child|children|school shoes|toys?)\b", case=False, regex=True).to_numpy()
            m &= ~juvenile
        
        return m
    
    def _make_boost(self, intent):
        """Create attribute boost vector"""
        b = np.zeros(self.n_docs)
        
        if intent.get("is_father_gift"):
            is_mature = self.catalog["category"].isin(MATURE_MEN_CATEGORIES).to_numpy()
            b += BOOST["category"] * 1.5 * is_mature
        
        if intent.get("category"):
            pc = _cat(intent["category"])
            ptok = set(pc.split())
            for i, (rc, rt) in enumerate(zip(self.cat_norm, self.cat_tok)):
                if pc == rc:
                    b[i] += BOOST["category"]
                elif ptok & rt:
                    b[i] += BOOST["category"] / 2
        
        if intent.get("brand"):
            b += BOOST["brand"] * (self.brand_n == _alnum(intent["brand"]))
        
        if intent.get("color"):
            b += BOOST["color"] * (self.color_n == _alnum(intent["color"]))
        
        if intent.get("material"):
            b += BOOST["material"] * np.array([is_material_match(intent["material"], m) for m in self.catalog["material"]])
        
        return b
    
    def _attr_bonus(self, intent):
        """Compute attribute bonus and flags"""
        flags = {}
        
        if intent.get("color"):
            flags["color"] = self.color_n == _alnum(intent["color"])
        if intent.get("brand"):
            flags["brand"] = self.brand_n == _alnum(intent["brand"])
        if intent.get("material"):
            flags["material"] = np.array([is_material_match(intent["material"], m) for m in self.catalog["material"]])
        if intent.get("category"):
            pc = _cat(intent["category"])
            flags["category"] = np.array([pc == rc for rc in self.cat_norm])
        
        bonus = np.zeros(self.n_docs)
        for name, f in flags.items():
            bonus += ATTR_W[name] * f
        
        return bonus, flags
    
    def _compute_ce_scores(self, q_clean, cand):
        """Compute cross-encoder scores with caching, batching, and inference mode"""
        ce = self.models.get_reranker()
        need = [i for i in cand if (q_clean, i) not in CE_CACHE]
        
        if need:
            # Use concise representation (first 150 chars) for lightning-fast cross-encoder scoring
            pairs = [(q_clean, str(self.catalog["text"].iloc[i])[:150]) for i in need]
            with torch.inference_mode():
                preds = ce.predict(
                    pairs,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
            for i, s in zip(need, preds):
                CE_CACHE[(q_clean, i)] = float(s)
        
        return np.array([CE_CACHE[(q_clean, i)] for i in cand])
    
    def search(self, query, k=5):
        """Main search function with high-speed hybrid retrieval and capped reranking"""
        # Fast query-result cache
        cache_key = f"{query.strip().lower()}_{k}"
        if cache_key in SEARCH_CACHE:
            return SEARCH_CACHE[cache_key]
        
        # Normalize query (Bangla support)
        query_norm = _alias_pass(query)
        if not query_norm.strip():
            query_norm = "product"
        
        # Detect product groups
        grp_mask, grp_names = self.detect_groups(query_norm)
        
        # Parse intent
        intent = self.parse_query(query_norm)
        
        # Prepare retrieval
        prep = self.prepare_retrieval(query_norm, intent)
        
        # --- RRF-based candidate selection ---
        # Apply attribute boost on top of RRF
        rrf_boosted = prep["rrf"] + F_BS * prep["boost"]
        
        # Compute attribute bonus
        bonus, flags = self._attr_bonus(intent)
        
        # Group bonus
        grp_bonus = GROUP_TIER * grp_mask if grp_mask is not None else np.zeros(self.n_docs)
        
        # Combined selection score
        sel = rrf_boosted + grp_bonus + bonus
        
        # Hard constraint filtering (price, gender, size)
        mask = prep.get("mask")
        if mask is not None and mask.any():
            sel[~mask] = -9999.0
        
        # Select candidates: pool size scales with k
        pool_size = max(G_POOL, k)
        cand = [int(i) for i in np.argsort(-sel)[:pool_size]]
        
        if not cand:
            cand = [int(i) for i in np.argsort(-prep["raw_cos"])[:pool_size]]
        
        # Keep also the base hybrid score
        base = F_ALPHA * prep["cos"] + (1 - F_ALPHA) * prep["bm"] + F_BS * prep["boost"]
        
        # Fast Neural Reranking: Only rerank top candidates (up to CE_RERANK_MAX)
        rerank_n = min(len(cand), CE_RERANK_MAX)
        rerank_cand = cand[:rerank_n]
        remaining_cand = cand[rerank_n:]
        
        if rerank_n > 1:
            ce_top = self._compute_ce_scores(prep["q_clean"], rerank_cand)
            min_ce = float(ce_top.min()) if len(ce_top) > 0 else CE_MIN
            # Remaining candidates outside top rerank pool receive baseline CE score
            ce_rem = np.full(len(remaining_cand), min_ce - 2.0)
            ce_raw = np.concatenate([ce_top, ce_rem])
        else:
            ce_raw = np.zeros(len(cand))
        
        # Determine confidence
        confident = (grp_mask is not None) or bool(flags) or \
                   (prep["raw_bm"][cand].max() > 0) or \
                   (prep["raw_cos"][cand].max() >= COS_MIN) or \
                   (ce_raw.max() >= CE_MIN)
        
        if not confident and not ALLOW_FALLBACK:
            return pd.DataFrame(), intent, prep["relaxed"], "No confident match found"
        
        # Final scoring
        if confident:
            blend = (1 - G_W) * _minmax(base[cand]) + G_W * _minmax(ce_raw) if len(cand) > 1 else np.ones(1)
            final = blend + grp_bonus[cand] + bonus[cand]
            match_type = ["intent" if grp_mask[i] else "related" for i in cand] if grp_mask is not None else [""] * len(cand)
        else:
            final = _minmax(base[cand]) if len(cand) > 1 else np.ones(1)
            match_type = ["closest"] * len(cand)
        
        # Get top-k
        order = np.argsort(-final)[:k]
        top = [cand[j] for j in order]
        
        # Build result dataframe
        df = self.catalog.iloc[top][["id", "title", "brand", "color", "price", "description", "category", "sizes", "gender", "material"]].copy()
        df.insert(0, "rank", range(1, len(df) + 1))
        df["score"] = np.round(final[order] * 100, 1)
        df["cosine"] = np.round(prep["raw_cos"][top], 3)
        df["bm25"] = np.round(prep["raw_bm"][top], 3)
        df["rerank"] = np.round(ce_raw[order], 2)
        df["match"] = [match_type[j] for j in order]
        df["matched"] = ["+".join(n for n, f in flags.items() if f[i]) for i in top]
        
        note = ""
        if intent.get("is_father_gift"):
            note += "Recipient: Father (Mature Men's Collection) | "
        if intent.get("max_price") is not None and intent.get("min_price") is not None:
            note += f"Price: ৳{intent['min_price']} - ৳{intent['max_price']} | "
        elif intent.get("max_price") is not None:
            note += f"Price: <= ৳{intent['max_price']} | "
        elif intent.get("min_price") is not None:
            note += f"Price: >= ৳{intent['min_price']} | "
        if intent.get("size"):
            note += f"Size: {intent['size']} | "
        if grp_names:
            note += f"Category: {', '.join(grp_names)}"
        
        res = (df.reset_index(drop=True), intent, prep["relaxed"], note.rstrip(" | "))
        SEARCH_CACHE[cache_key] = res
        return res