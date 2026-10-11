# 🏗️ Intelligent E-Commerce Search - Architecture Overview

This document illustrates the complete data flow and architectural design of the E-commerce Search System.

## High-Level Architecture Diagram

```mermaid
flowchart TD
    %% User Inputs
    subgraph UI["🖥️ Frontend (Streamlit)"]
        A[User Input] -->|Text Query| C(app.py)
        B[🎤 Voice Input] -->|streamlit-mic-recorder| C
        C -->|Raw Query| D[Search Engine Module]
        Filters[🎛️ Sidebar Filters: Category & Price] -.-> C
    end

    %% Processing & Search Engine
    subgraph Core["⚙️ Core Search Logic (search_engine.py & utils.py)"]
        D --> E{Pre-processing}
        E -->|normalize_banglish| F[Banglish & Alias Translation]
        F --> G[Extract Intent: Brand, Gender, Color]
        
        G --> H{Two-Pronged Retrieval}
        
        %% BM25
        H -->|Lexical Exact Match| I[BM25 Index]
        
        %% Semantic
        H -->|Semantic Match| J[Sentence Transformer]
        
        I --> K[Candidate Pool Generation]
        J --> K
    end

    %% AI Models
    subgraph AI["🧠 AI Model Layer (models.py)"]
        J -.->|paraphrase-multilingual-MiniLM| L(Dense Embeddings)
        L -.-> J
        
        K --> M[Cross-Encoder Re-ranker]
        M -.->|mmarco-mMiniLMv2| N(Semantic Scoring)
        N -.-> M
    end

    %% Data Source
    subgraph Data["💾 Data Layer (data_products.py)"]
        DB[(Product Catalog)] -.->|Pandas DataFrame| I
        DB -.->|Pre-computed Embeddings| J
    end

    %% Output
    M --> O[Final Sorted Results]
    O -->|Top Matches vs Related| P{Apply Filters}
    Filters -.-> P
    P --> Q[Render HTML Product Cards]
    Q --> UI
```

## How It Works (Step-by-Step)

1. **User Input (`app.py`)**: 
   - A user types a search term or uses the **Web Speech API** via the microphone to speak a query (in English or Bengali).
2. **Pre-processing (`utils.py`)**:
   - The system intercepts the query and cleans it. If the user used "Banglish" (e.g. typing *kacha morich* or *shari*), the custom normalization dictionary translates or maps these terms into searchable English/semantic equivalents.
3. **Intent Extraction**:
   - It runs Regex rules to detect intent. If the user searches "blue nike shoes", it automatically detects "blue" (color) and "nike" (brand), ensuring these filters are strictly applied.
4. **Retrieval (`search_engine.py`)**:
   - **Lexical Search**: Uses the **BM25 algorithm** to find exact keyword matches across product titles and descriptions.
   - **Semantic Search**: Uses a **Multilingual Sentence Transformer** to understand the *meaning* of the text, matching products even if the exact words weren't used.
5. **Re-Ranking (`models.py`)**:
   - The candidates from the retrieval phase are pooled together and sent to a **Cross-Encoder Model**. This heavy AI model deeply compares the original query against each product description to assign a highly accurate confidence score.
6. **Filtering & Display**:
   - The results are split into "🎯 Top Matches" (high confidence) and "💡 Related Products" (lower confidence). 
   - Any active Sidebar Filters (like Max Price or Category) are applied locally via Pandas.
   - The final results are rendered as beautiful, hoverable HTML cards in the UI.
