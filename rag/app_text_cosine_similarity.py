import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_model(model_name="all-MiniLM-L6-v2"):
    """Load the embedding model"""
    with st.spinner(f"Loading model: {model_name}..."):
        return SentenceTransformer(model_name)

def get_embedding(model, text):
    """Generate embeddings for input text"""
    return model.encode(text)

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    # Reshape embeddings for cosine_similarity function
    emb1 = embedding1.reshape(1, -1)
    emb2 = embedding2.reshape(1, -1)
    
    similarity = cosine_similarity(emb1, emb2)[0][0]
    return similarity

def main():
    st.set_page_config(
        page_title="Text Embedding & Similarity",
        page_icon="🔤",
    )
    
    st.title("Text Similarity Comparison")
    st.write("Compare the semantic meaning of different texts using embeddings.")
    
    # Initialize session state for texts if not already present
    if 'texts' not in st.session_state:
        st.session_state.texts = ["", ""]  # Start with two empty text fields
    
    # Model selection
    model_name = st.selectbox(
        "Select embedding model",
        ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0
    )
    
    # Load model (cached with st.cache)
    @st.cache_resource
    def get_model(model_name):
        return load_model(model_name)
    
    model = get_model(model_name)
    
    # Text input area
    st.subheader("Enter texts to compare")
    
    # Create dynamic text areas based on session state
    updated_texts = []
    for i, text in enumerate(st.session_state.texts):
        text_input = st.text_area(f"Text {i+1}", text, height=100, key=f"text_{i}")
        updated_texts.append(text_input)
    
    # Update session state
    st.session_state.texts = updated_texts
    
    # Add/remove text areas
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Add another text") and len(st.session_state.texts) < 5:
            st.session_state.texts.append("")
            st.rerun()
    
    with col2:
        if st.button("➖ Remove last text") and len(st.session_state.texts) > 2:
            st.session_state.texts.pop()
            st.rerun()
    
    # Compare button
    if st.button("Compare Texts"):
        # Filter out empty texts
        valid_texts = [text for text in st.session_state.texts if text.strip()]
        
        if len(valid_texts) < 2:
            st.error("Please enter at least 2 texts to compare.")
        else:
            with st.spinner("Generating embeddings..."):
                embeddings = [get_embedding(model, text) for text in valid_texts]
            
            st.subheader("Similarity Results")
            
            # Create a matrix to show all comparisons
            if len(valid_texts) > 2:
                st.write("Similarity matrix:")
                # Create similarity matrix
                sim_matrix = np.zeros((len(valid_texts), len(valid_texts)))
                for i in range(len(valid_texts)):
                    for j in range(len(valid_texts)):
                        sim_matrix[i, j] = calculate_similarity(embeddings[i], embeddings[j])
                
                # Display as dataframe
                import pandas as pd
                df = pd.DataFrame(sim_matrix, 
                                  index=[f"Text {i+1}" for i in range(len(valid_texts))],
                                  columns=[f"Text {i+1}" for i in range(len(valid_texts))])
                st.dataframe(df.style.format("{:.4f}").background_gradient(cmap="viridis"))
            
            # Show detailed comparison of all pairs
            st.write("Detailed comparisons:")
            for i in range(len(valid_texts)):
                for j in range(i+1, len(valid_texts)):
                    similarity = calculate_similarity(embeddings[i], embeddings[j])
                    
                    # Create expander for each pair
                    with st.expander(f"Text {i+1} & Text {j+1} - Similarity: {similarity:.4f}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area("", valid_texts[i], height=100, disabled=True,
                                        label_visibility="collapsed", key=f"compare_text_{i}_{j}_1")
                        with col2:
                            st.text_area("", valid_texts[j], height=100, disabled=True,
                                        label_visibility="collapsed", key=f"compare_text_{i}_{j}_2")
    
    st.info("ℹ️ The similarity score ranges from -1 to 1, where 1 indicates identical meaning, " +
           "0 indicates unrelated texts, and negative values suggest opposite meanings.")

if __name__ == "__main__":
    main()
