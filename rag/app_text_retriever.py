import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class TextRetriever:
    def __init__(self, texts, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the text retriever with a list of texts and a model for creating embeddings.
        
        Args:
            texts (list): List of texts to search through
            model_name (str): Name of the sentence-transformer model to use
        """
        self.texts = texts
        self.model = SentenceTransformer(model_name)
        self.update_embeddings()
        
    def update_embeddings(self):
        """Update embeddings after texts have been modified"""
        self.embeddings = self.model.encode(self.texts)
    
    def add_text(self, new_text):
        """Add a new text to the collection and update embeddings"""
        self.texts.append(new_text)
        self.update_embeddings()
    
    def find_most_similar_texts(self, query_text, top_k=5):
        """
        Find the top k most similar texts to the query_text.
        
        Args:
            query_text (str): The query text to compare against
            top_k (int): Number of most similar texts to return
            
        Returns:
            list: List of tuples (text, similarity_score) for the top k most similar texts
        """
        # Encode the query text
        query_embedding = self.model.encode([query_text])
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Find indices of top k most similar texts
        # Get the number of texts to return (either top_k or the total number of texts, whichever is smaller)
        num_to_return = min(top_k, len(self.texts))
        top_indices = np.argsort(similarities)[::-1][:num_to_return]
        
        # Return list of (text, score) tuples
        results = [(self.texts[idx], similarities[idx]) for idx in top_indices]
        
        return results

# Initialize session state for texts if not already set
def initialize_session():
    # Clear existing session state to ensure we're using the latest class definition
    if 'retriever' in st.session_state:
        del st.session_state.retriever
    
    if 'texts' not in st.session_state:
        st.session_state.texts = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn fox leaps above the sleepy canine",
            "The sky is blue and the grass is green",
            "Machine learning is a subfield of artificial intelligence",
            "Python is a versatile programming language"
        ]
    
    # Always reinitialize the retriever to ensure we're using the latest class definition
    st.session_state.retriever = TextRetriever(st.session_state.texts)

def main():
    st.title("Text Similarity Search")
    st.write("Find the top 5 most similar texts from a collection using cosine similarity")
    
    # Initialize app state
    initialize_session()
    
    # Add new text section
    st.subheader("Add New Text")
    new_text = st.text_area("Enter new text to add to the collection:", height=100)
    
    if st.button("Add Text"):
        if new_text and new_text not in st.session_state.texts:
            st.session_state.retriever.add_text(new_text)
            st.success(f"Added: '{new_text}'")
        elif new_text in st.session_state.texts:
            st.warning("This text is already in the collection!")
        else:
            st.warning("Please enter some text!")
    
    # Show current texts
    st.subheader("Current Texts Collection")
    for i, text in enumerate(st.session_state.texts):
        st.write(f"{i+1}. {text}")
    
    # Query section
    st.subheader("Find Similar Texts")
    query_text = st.text_input("Enter your query:")
    
    if st.button("Search"):
        if query_text:
            with st.spinner("Finding most similar texts..."):
                results = st.session_state.retriever.find_most_similar_texts(query_text)
                
            st.success("Search complete!")
            st.subheader("Results")
            
            for i, (text, score) in enumerate(results):
                with st.container():
                    st.write(f"**Rank {i+1}** (Similarity score: {score:.4f})")
                    st.info(text)
                    st.markdown("---")
        else:
            st.warning("Please enter a query!")

if __name__ == "__main__":
    main()
