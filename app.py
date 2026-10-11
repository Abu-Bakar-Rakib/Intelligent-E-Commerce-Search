import streamlit as st
import pandas as pd
from data_products import RAW_PRODUCTS
from models import load_models
from search_engine import SearchEngine
from streamlit_mic_recorder import speech_to_text

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="🛍️ Intelligent E-Commerce Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    /* Clean up overall padding */
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }
    
    /* Header Styling */
    h1 {
        text-align: center;
        font-weight: 800 !important;
        background: -webkit-linear-gradient(45deg, #4f46e5, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem !important;
    }
    
    /* Modern Search Bar */
    div[data-testid="stTextInput"] > div[data-baseweb="input"] {
        border-radius: 50px !important;
        border: 2px solid #e2e8f0 !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01) !important;
        background-color: white !important;
        padding: 8px 20px !important;
        transition: all 0.3s ease;
    }
    div[data-testid="stTextInput"] > div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15) !important;
    }
    div[data-testid="stTextInput"] input {
        font-size: 1.15rem !important;
        color: #1e293b !important;
        font-weight: 500 !important;
    }
    
    /* Product Card Styling */
    .product-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.025);
        border: 1px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -4px rgba(0, 0, 0, 0.04);
        border-color: #e2e8f0;
    }
    .product-info {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .product-rank {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        height: 50px;
        width: 50px;
        min-width: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.2);
    }
    .product-details h4 {
        margin: 0 0 6px 0 !important;
        font-size: 1.15rem !important;
        color: #0f172a !important;
        font-weight: 700 !important;
    }
    .product-details p {
        margin: 0 !important;
        color: #64748b !important;
        font-size: 0.95rem !important;
    }
    .product-details .tag {
        display: inline-block;
        background: #f8fafc;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        color: #475569;
        margin-right: 8px;
        margin-top: 10px;
        font-weight: 600;
        border: 1px solid #e2e8f0;
    }
    .product-price h3 {
        margin: 0 !important;
        font-size: 1.6rem !important;
        color: #10b981 !important;
        font-weight: 800 !important;
    }
    
    /* Headers */
    h3 {
        font-weight: 700 !important;
        color: #334155 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to render modern product cards
def render_product_card(row):
    tags = f"<span class='tag'>{str(row['category']).title()}</span>"
    if pd.notna(row['brand']) and str(row['brand']).lower() != 'local':
        tags += f"<span class='tag'>{str(row['brand']).title()}</span>"
    if pd.notna(row['color']) and row['color']:
        tags += f"<span class='tag'>{str(row['color']).title()}</span>"
        
    html = f'''
    <div class="product-card">
        <div class="product-info">
            <div class="product-rank">#{row['rank']}</div>
            <div class="product-details">
                <h4>{row['title']}</h4>
                <p>{row['description']}</p>
                <div>{tags}</div>
            </div>
        </div>
        <div class="product-price">
            <h3>₹{row['price']:,}</h3>
        </div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# ===== SESSION STATE =====
if "search_engine" not in st.session_state:
    st.session_state.search_engine = None
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "search_count" not in st.session_state:
    st.session_state.search_count = 0

# ===== HEADER =====
st.title("🛍️ Intelligent E-Commerce Search")

# ===== SIDEBAR =====
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Initialize models once
    if st.session_state.search_engine is None:
        st.info("Loading models for the first time (may take 1-2 minutes)...")
        models = load_models()
        
        # Build catalog
        catalog_data = {
            "id": [f"P{i+1:03d}" for i in range(len(RAW_PRODUCTS))],
            "title": [p[0] for p in RAW_PRODUCTS],
            "category": [p[1] for p in RAW_PRODUCTS],
            "brand": [p[2] for p in RAW_PRODUCTS],
            "color": [p[3] for p in RAW_PRODUCTS],
            "material": [p[4] for p in RAW_PRODUCTS],
            "gender": [p[5] for p in RAW_PRODUCTS],
            "sizes": [p[6] for p in RAW_PRODUCTS],
            "price": [int(p[7]) for p in RAW_PRODUCTS],
            "description": [p[8] for p in RAW_PRODUCTS],
        }
        
        import pandas as pd
        catalog = pd.DataFrame(catalog_data)
        catalog["size_set"] = catalog["sizes"].apply(lambda s: {x.strip().upper() for x in s.split(",") if x.strip()})
        catalog["text"] = (
            catalog["title"] + ". Category: " + catalog["category"] +
            ". Brand: " + catalog["brand"] + ". Color: " + catalog["color"] +
            ". Material: " + catalog["material"] + ". For " + catalog["gender"] +
            ". " + catalog["description"]
        )
        
        st.session_state.search_engine = SearchEngine(catalog, models)
        st.success("✅ Search engine ready!")
    
    top_k = st.slider("Results to show", min_value=1, max_value=20, value=5, step=1)
    
    st.divider()
    st.markdown("### 🎙️ Voice Search Language")
    voice_lang = st.selectbox("Language", ["English", "Bengali"], label_visibility="collapsed")
    st.session_state['voice_lang_code'] = "bn-BD" if voice_lang == "Bengali" else "en-US"
    
    st.divider()
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Products", len(st.session_state.search_engine.catalog))
    with col2:
        st.metric("Searches", st.session_state.search_count)
        
    st.divider()
    st.markdown("### 🎛️ Filter Results")
    all_categories = sorted(list(set(p[1] for p in RAW_PRODUCTS)))
    category_filter = st.selectbox("Category", ["All"] + all_categories)
    price_filter = st.slider("Max Price (₹)", 0, 150000, 150000, step=500)

# ===== MAIN SEARCH =====
col_search, col_mic = st.columns([10, 1], gap="small")
with col_search:
    text_query = st.text_input(
        "🔍 Search for products",
        placeholder="e.g., blue nike shoes under 5000, comfortable t-shirt for summer...",
        label_visibility="collapsed"
    )
with col_mic:
    voice_query = speech_to_text(
        start_prompt="🎤",
        stop_prompt="⏹️",
        language=st.session_state.get('voice_lang_code', 'en-US'),
        use_container_width=True,
        just_once=True,
        key='STT'
    )

# Use voice query if recorded, otherwise text query
query = voice_query if voice_query else text_query

# ===== EXECUTE SEARCH =====
import time

if query.strip():
    current_search_key = f"{query.strip()}_{top_k}"
    if current_search_key != st.session_state.get("last_search_key") or st.session_state.last_results is None:
        st.session_state.search_count += 1
        st.session_state["last_search_key"] = current_search_key
        
        t0 = time.time()
        with st.spinner("🔎 Searching..."):
            fetch_k = max(25, top_k * 3)
            results, intent, relaxed, note = st.session_state.search_engine.search(query, k=fetch_k)
        st.session_state["last_latency"] = time.time() - t0
        st.session_state.last_results = (results, intent, relaxed, note)

# ===== DISPLAY RESULTS =====
if st.session_state.last_results:
    results, intent, relaxed, note = st.session_state.last_results
    
    # Display speed latency indicator
    if "last_latency" in st.session_state:
        st.markdown(
            f"<div style='margin-bottom: 12px; font-size: 0.88rem; color: #64748b; font-weight: 500;'>"
            f"⚡ Search completed in <span style='color: #10b981; font-weight: 700;'>{st.session_state['last_latency']:.2f}s</span> "
            f"· <span style='color: #6366f1;'>Hybrid RRF + Accelerated Neural Reranker</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Apply Sidebar Filters
    if len(results) > 0:
        if category_filter != "All":
            results = results[results["category"].str.lower() == category_filter.lower()]
        results = results[results["price"] <= price_filter]
        
    if len(results) > 0:
        # "Match" threshold: Score >= 50 is considered an exact/strong match
        main_results = results[results['score'] >= 50.0]
        related_results = results[results['score'] < 50.0]
        
        if main_results.empty:
            related_results = results.head(top_k + 5)
        else:
            related_results = related_results.head(top_k)
            
        # Display main results
        if not main_results.empty:
            st.markdown("### 🎯 Top Matches")
            for idx, row in main_results.iterrows():
                render_product_card(row)
                
        # Display related results
        if not related_results.empty:
            if main_results.empty:
                st.markdown("### 💡 Related Products")
                st.info("We couldn't find an exact match for your search, but you might like these related items!")
            else:
                st.markdown("### 💡 Other Related Items")
            
            for idx, row in related_results.iterrows():
                render_product_card(row)
        
        # Results table
        st.markdown("### 📊 Results Table")
        display_cols = ["rank", "id", "title", "category", "brand", "color", "material", "gender", "sizes", "price", "description"]
        st.dataframe(
            results[display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "price": st.column_config.NumberColumn("Price", format="₹%d")
            }
        )
        
        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name="search_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("❌ No products found. Try a different search query.")


# ===== FOOTER =====
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px;">
    🚀 E-Commerce Intelligent Search System | Powered by Qwen 2.5, Sentence Transformers & Cross-Encoder
    </div>
""", unsafe_allow_html=True)