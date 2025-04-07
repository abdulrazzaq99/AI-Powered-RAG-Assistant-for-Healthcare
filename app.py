import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Load Gemini API key from secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Load FAISS and Data
@st.cache_resource
def load_faiss_and_data():
    df = pd.read_csv("chunks_with_embedding.csv")
    embeddings = np.vstack(df['embedding'].apply(eval).to_numpy())
    index = faiss.read_index("faiss_index.index")
    return df, index

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")  # use the same model used for indexing

df, index = load_faiss_and_data()
model = load_model()

# UI
st.title("💡 Gemini RAG Assistant")
query = st.text_input("Ask a question:")

if query:
    # Embed query
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding).astype("float32"), k=5)

    # Fetch top chunks
    context_chunks = df.iloc[indices[0]]['text'].tolist()
    context = "\n\n".join(context_chunks)

    st.subheader("📄 Retrieved Context")
    st.write(context)

    # Generate response with Gemini
    prompt = f"""Answer the following question using the context below.

Context:
{context}

Question:
{query}

Answer:"""

    model = genai.GenerativeModel("gemini-1.5-flash")
    with st.spinner("Generating response..."):
        response = model.generate_content(prompt)
        st.subheader("💬 Gemini's Answer")
        st.write(response.text)
