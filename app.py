import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time

# Page configuration and theme
st.set_page_config(
    page_title="Gemini RAG Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: 700;
        color: #4f8bf9;
        margin-bottom: 10px;
        text-align: center;
    }
    .sub-header {
        font-size: 20px;
        font-weight: 400;
        color: #555;
        margin-bottom: 30px;
        text-align: center;
    }
    .stTextInput > label {
        font-size: 18px;
        font-weight: 500;
        color: #333;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .context-header {
        font-size: 20px;
        font-weight: 500;
        color: #555;
        margin-top: 10px;
    }
    .answer-container {
        background-color: #e6f2ff;
        padding: 20px;
        border-radius: 8px;
        margin-top: 10px;
    }
    .answer-header {
        font-size: 22px;
        font-weight: 500;
        color: #333;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        color: #888;
        font-size: 14px;
    }
    .metrics-container {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .badge {
        display: inline-block;
        padding: 5px 10px;
        background-color: #4f8bf9;
        color: white;
        border-radius: 15px;
        font-weight: 500;
        margin-right: 10px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with app info
with st.sidebar:
    st.image("https://developers.generativeai.google/static/site-assets/images/marketing/gemini-api-logo-rgb.svg", width=200)
    st.markdown("### About This App")
    st.markdown("""
    This application uses RAG (Retrieval-Augmented Generation) to answer questions based on your knowledge base.
    
    **Technologies used:**
    - 🤖 Google Gemini 1.5 Flash
    - 🔍 FAISS for vector search
    - 🔢 Sentence Transformers
    - 📊 Streamlit
    """)
    
    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. Your question is converted to an embedding
    2. FAISS finds similar content from your documents
    3. Gemini generates an answer based on the retrieved context
    """)
    
# Load Gemini API key from secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load FAISS and Data
@st.cache_resource
def load_faiss_and_data():
    with st.spinner("Loading knowledge base..."):
        # Load the CSV file
        df = pd.read_csv("chunks_with_embeddings.csv")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['chunks'].tolist())
        embeddings = np.vstack(embeddings)
        df['embedding'] = embeddings.tolist()
        index = faiss.read_index("faiss_index.index")
        return df, index, len(df)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Load data and models
df, index, total_chunks = load_faiss_and_data()
model = load_model()

# Main UI
st.markdown('<div class="main-header">🧠 Gemini RAG Knowledge Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about your documents and get AI-powered answers</div>', unsafe_allow_html=True)

# Info metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="metrics-container"><span class="badge">Model</span> Gemini 1.5 Flash</div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metrics-container"><span class="badge">Database</span> {total_chunks} Document Chunks</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metrics-container"><span class="badge">Embedding</span> all-MiniLM-L6-v2</div>', unsafe_allow_html=True)

# Query input with larger font and better styling
query = st.text_input("💭 What would you like to know?", key="query_input", placeholder="Enter your question here...")

# Number of chunks slider
num_chunks = st.slider("Number of context chunks to retrieve", min_value=1, max_value=10, value=5)

# Run button for better UX
run_query = st.button("🔍 Get Answer", type="primary", use_container_width=True)

if run_query and query:
    # Create tabs for different views
    main_tab, context_tab, debug_tab = st.tabs(["Answer", "Retrieved Context", "Debug Info"])
    
    with st.spinner("Processing your question..."):
        # Start timer
        start_time = time.time()
        
        # Embed query
        query_embedding = model.encode([query])
        start_retrieval = time.time()
        _, indices = index.search(np.array(query_embedding).astype("float32"), k=num_chunks)
        retrieval_time = time.time() - start_retrieval

        # Fetch top chunks
        context_chunks = df.iloc[indices[0]]['chunks'].tolist()
        context = "\n\n".join(context_chunks)
        
        # Generate response with Gemini
        prompt = f"""Answer the following question using the context below. If the context doesn't contain relevant information, say so.

Context:
{context}

Question:
{query}

Answer:"""

        start_generation = time.time()
        model = genai.GenerativeModel("gemini-2.0-flash-001")
        response = model.generate_content(prompt)
        generation_time = time.time() - start_generation
        total_time = time.time() - start_time
        
    # Show answer in the main tab
    with main_tab:
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-header">📝 Answer</div>', unsafe_allow_html=True)
        st.write(response.text)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Time", f"{retrieval_time:.2f}s")
        col2.metric("Generation Time", f"{generation_time:.2f}s")
        col3.metric("Total Time", f"{total_time:.2f}s")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show context in the context tab
    with context_tab:
        st.markdown('<div class="context-header">📄 Retrieved Context</div>', unsafe_allow_html=True)
        for i, chunk in enumerate(context_chunks):
            st.markdown(f"**Chunk {i+1}**")
            st.markdown(f"<div class='result-container'>{chunk}</div>", unsafe_allow_html=True)
    
    # Show debug info in the debug tab
    with debug_tab:
        st.markdown("### Query Embedding")
        st.write(f"Shape: {query_embedding.shape}")
        
        st.markdown("### Retrieved Indices")
        st.write(indices[0])
        
        st.markdown("### Full Prompt Sent to Gemini")
        st.code(prompt, language="text")

# Add a footer with LinkedIn-friendly info
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, FAISS, Sentence Transformers and Google Gemini 1.5 | 
    <a href="https://www.linkedin.com/in/your-profile/" target="_blank">Connect on LinkedIn</a>
</div>
""", unsafe_allow_html=True)