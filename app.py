import streamlit as st
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Set up the web page title and icon
st.set_page_config(page_title="Employee Policy Assistant", page_icon="📋", layout="centered")

# ==========================================
# 1. CORE DATA & RESOURCE CACHING
# ==========================================

@st.cache_resource
def load_rag_system():
    """Loads and compiles models and vector index once into memory."""
    # Step A: The Source Knowledge Corpus (Your exact policy chunks)
    policy_chunks = [
        "Employees receive 12 casual leaves annually.",
        "Employees can work from home twice per week.",
        "Travel expenses are reimbursed within 30 days.",
        "Medical insurance is provided to all employees."
    ]
    
    # Step B: Initialize the Embedding Model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Step C: Populate FAISS Vector Database Index
    embeddings = embed_model.encode(policy_chunks).astype('float32')
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    
    # Step D: Initialize the Text Generation Pipeline
    llm_pipeline = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    return embed_model, faiss_index, policy_chunks, llm_pipeline

# Spin up the backend assets
with st.spinner("Initializing AI Models and Knowledge Base... Please wait."):
    model, index, chunks, qa_pipeline = load_rag_system()

# ==========================================
# 2. STREAMLIT USER INTERFACE (UI)
# ==========================================

st.title("📋 Employee Policy Assistant")
st.markdown("Ask any question regarding leave, remote work, reimbursements, or insurance.")

# Initialize chat history state so conversation flows naturally
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw existing chat history on screen refresh
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input question
if user_question := st.chat_input("e.g., How many casual leaves do I get?"):
    
    # Display the user's question in the UI
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    # Generate response block
    with st.chat_message("assistant"):
        with st.spinner("Searching internal policies..."):
            
            # Step 1: Vectorize question and run FAISS retrieval
            query_vector = model.encode([user_question]).astype('float32')
            distances, indices = index.search(query_vector, k=2)
            
            retrieved_contexts = []
            for idx in indices[0]:
                if idx < len(chunks) and idx != -1:
                    retrieved_contexts.append(chunks[idx].strip())
            
            context_str = "\n".join(retrieved_contexts)
            
            # Step 2: Structure the system/user instruction prompt for TinyLlama
            prompt = f"""<|system|>
You are a helpful HR Assistant. Answer the employee's question accurately using ONLY the provided context. If the answer cannot be found in the context, say "I do not have that information in my current policy documents."
Context:
{context_str}</s>
<|user|>
{user_question}</s>
<|assistant|>
"""
            
            # Step 3: Run pipeline inference
            outputs = qa_pipeline(
                prompt,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False
            )
            
            # Step 4: Parse final answer text cleanly
            generated_raw = outputs[0]["generated_text"]
            answer = generated_raw.split("<|assistant|>")[-1].strip()
            
            # Render answer and source context
            st.markdown(answer)
            with st.expander("🔍 View Retrieved Policy Context"):
                st.caption(f"**Matches found:**\n{context_str}")
                
        # Commit response to history state
        st.session_state.messages.append({"role": "assistant", "content": answer})
