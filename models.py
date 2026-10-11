import torch
import streamlit as st
from transformers import StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer, CrossEncoder
from config import EMB_NAME, CE_NAME, CE_FP16

class ModelManager:
    """Singleton for model management - loads once, reuses"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Skip loading LLM as it requires GPU and is unused
        
        with st.spinner("⏳ Loading embedder..."):
            self.embedder = SentenceTransformer(EMB_NAME, device=self.device)
        
        with st.spinner("⏳ Loading reranker..."):
            self.ce = CrossEncoder(CE_NAME, device=self.device)
            if self.device == "cuda" and CE_FP16:
                try:
                    self.ce.model.half()
                except Exception as e:
                    st.warning(f"FP16 skipped: {e}")
        
        st.success(f"✅ All models loaded on {self.device.upper()}")
        self._initialized = True
    

    def get_embedder(self):
        return self.embedder
    
    def get_reranker(self):
        return self.ce

@st.cache_resource
def load_models():
    """Streamlit cache wrapper for singleton"""
    return ModelManager()