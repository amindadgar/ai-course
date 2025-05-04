import os
import streamlit as st
from together import Together
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import uuid
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size (int): Size of document chunks
            chunk_overlap (int): Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text, metadata=None):
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Text to split
            metadata (dict): Optional metadata for the document
            
        Returns:
            list: List of dictionaries with text chunks and metadata
        """
        if not text.strip():
            return []
            
        # Simple chunking by character count with overlap
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_text = text[i:i + self.chunk_size]
            if not chunk_text.strip():
                continue
                
            # Create chunk with unique ID and metadata
            chunk = {
                "id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": metadata or {}
            }
            chunks.append(chunk)
            
        return chunks

class VectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the vector store with an embedding model.
        
        Args:
            model_name (str): Name of the sentence-transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = np.array([])
    
    def add_documents(self, documents):
        """
        Add documents to the vector store and compute embeddings.
        
        Args:
            documents (list): List of document dictionaries with 'id', 'text', and 'metadata'
        """
        if not documents:
            return
            
        # Add documents to store
        self.documents.extend(documents)
        
        # Compute embeddings for new documents
        texts = [doc["text"] for doc in documents]
        new_embeddings = self.model.encode(texts)
        
        # Append to existing embeddings
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, new_embeddings))
    
    def search(self, query, top_k=5):
        """
        Search for most similar documents to the query.
        
        Args:
            query (str): Query text
            top_k (int): Number of documents to return
            
        Returns:
            list: Top k most similar documents
        """
        if not self.documents:
            return []
            
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k results
        num_to_return = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:num_to_return]
        
        # Return documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            doc["score"] = float(similarities[idx])
            results.append(doc)
            
        return results

class LLMInterface:
    def __init__(self, model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Initialize the LLM interface.
        
        Args:
            model (str): LLM model to use
        """
        self.model = model
        self.api_key_available = api_key is not None and api_key.strip() != ""
    
    def generate(self, prompt, max_tokens=500):
        """
        Generate text using the LLM.
        
        Args:
            prompt (str): Prompt for the LLM
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated text
        """
        if self.api_key_available:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                    n=1,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                st.error(f"Error generating response with Together AI: {str(e)}")
                return self._local_generate(prompt)
        else:
            st.warning("Together AI API key not found. Using local text extraction mode.")
            return self._local_generate(prompt)
    
    def _local_generate(self, prompt):
        """
        Simple local text generation when API is not available.
        This extracts key information from the retrieved documents.
        
        Args:
            prompt (str): The prompt with context and question
            
        Returns:
            str: Generated response based on context
        """
        # Parse the prompt to extract context and question
        try:
            # Extract the user question
            match = re.search(r"User question: (.*?)(\n|$)", prompt)
            question = match.group(1) if match else "the question"
            
            # Extract relevant snippets from the context
            context_parts = prompt.split("Context:")[1].split("User question:")[0].strip()
            documents = re.findall(r"Document \d+:\n(.*?)(?=Document \d+:|$)", context_parts, re.DOTALL)
            
            # Simple text extraction (without actual NLP understanding)
            if documents:
                # Find sentences in the documents that might contain answers
                # This is a very basic approach - just looking for sentences containing keywords from the question
                question_words = set(question.lower().split())
                question_words = {w for w in question_words if len(w) > 3}  # Filter out short words
                
                relevant_sentences = []
                for doc in documents:
                    sentences = re.split(r'[.!?]+', doc)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        sentence_words = set(sentence.lower().split())
                        # Check if sentence shares keywords with question
                        if question_words.intersection(sentence_words):
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    return f"Based on the provided documents, I found the following information about {question}:\n\n" + \
                           "\n".join([f"- {sentence}." for sentence in relevant_sentences[:3]])
                else:
                    # If no relevant sentences, return the first few sentences from top documents
                    first_doc = documents[0] if documents else ""
                    first_sentences = re.split(r'[.!?]+', first_doc)[:3]
                    return f"I found the following information that might help answer your question:\n\n" + \
                           "\n".join([f"- {s.strip()}." for s in first_sentences if s.strip()])
            else:
                return "I couldn't find specific information to answer your question in the provided documents."
        except Exception as e:
            return f"I couldn't generate a response based on the documents. Please try a different question or add more documents. Error: {str(e)}"

class RAGPipeline:
    def __init__(self, chunk_size=1000, chunk_overlap=200, model_name='all-MiniLM-L6-v2', llm_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
        """
        Initialize the RAG pipeline.
        
        Args:
            chunk_size (int): Size of document chunks
            chunk_overlap (int): Overlap between chunks
            model_name (str): Name of the sentence-transformer model
            llm_model (str): LLM model to use
        """
        self.processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(model_name)
        self.llm = LLMInterface(llm_model)
        
    def add_document(self, text, metadata=None):
        """
        Process and add a document to the RAG pipeline.
        
        Args:
            text (str): Document text
            metadata (dict): Optional metadata
        """
        chunks = self.processor.chunk_text(text, metadata)
        self.vector_store.add_documents(chunks)
        return len(chunks)
        
    def query(self, query_text, top_k=5):
        """
        Perform a RAG query: retrieve documents and generate a response.
        
        Args:
            query_text (str): User query
            top_k (int): Number of documents to retrieve
            
        Returns:
            dict: Results including retrieved documents and generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.search(query_text, top_k)
        
        if not retrieved_docs:
            return {
                "query": query_text,
                "retrieved_documents": [],
                "response": "No relevant documents found to answer your query."
            }
        
        # Construct prompt with retrieved context
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""Use the following information to answer the user's question.
        
Context:
{context}

User question: {query_text}

Answer:"""
        
        # Generate response
        response = self.llm.generate(prompt)
        
        return {
            "query": query_text,
            "retrieved_documents": retrieved_docs,
            "response": response
        }

def initialize_rag_pipeline():
    """Initialize or get the RAG pipeline from session state"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    return st.session_state.rag_pipeline

def main():
    st.title("RAG Pipeline Demo")
    st.write("Upload documents, ask questions, and get AI-generated answers based on your documents.")
    
    # Initialize RAG pipeline
    rag = initialize_rag_pipeline()
    
    # Display API status
    if not rag.llm.api_key_available:
        st.warning("Together AI API key not found. Using local text extraction mode. Set your TOGETHER_API_KEY in a .env file for better results.")
    
    # Sidebar for document upload and management
    with st.sidebar:
        st.header("Document Management")
        
        # Text input
        st.subheader("Add Text Document")
        doc_text = st.text_area("Enter document text:", height=200)
        doc_name = st.text_input("Document name (optional):")
        
        if st.button("Add Text Document"):
            if doc_text:
                metadata = {"name": doc_name or "Unnamed Document", "source": "text_input"}
                num_chunks = rag.add_document(doc_text, metadata)
                st.success(f"Added document: {metadata['name']} ({num_chunks} chunks)")
            else:
                st.warning("Please enter some text.")
        
        # File upload
        st.subheader("Upload Document File")
        uploaded_file = st.file_uploader("Choose a text file", type=["txt", "md", "csv"])
        
        if uploaded_file is not None:
            try:
                # For simplicity, just read as text
                content = uploaded_file.read().decode("utf-8")
                metadata = {"name": uploaded_file.name, "source": "file_upload"}
                
                if st.button("Process Uploaded File"):
                    num_chunks = rag.add_document(content, metadata)
                    st.success(f"Processed file: {uploaded_file.name} ({num_chunks} chunks)")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # Main area for RAG interaction
    st.header("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if st.button("Submit Question"):
        if query:
            with st.spinner("Processing your query..."):
                result = rag.query(query)
            
            st.subheader("Answer")
            st.write(result["response"])
            
            # Show retrieved documents
            with st.expander("View Retrieved Documents"):
                for i, doc in enumerate(result["retrieved_documents"]):
                    st.markdown(f"**Document {i+1}** (Score: {doc['score']:.4f})")
                    st.text(doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"])
                    st.markdown("---")
        else:
            st.warning("Please enter a question.")
    
    # Document statistics
    st.sidebar.subheader("Statistics")
    st.sidebar.write(f"Total documents: {len(rag.vector_store.documents)}")

if __name__ == "__main__":
    main()
