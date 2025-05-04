# ISF-AI: Intelligent Systems Framework

This repository contains AI components for natural language processing, focusing on two main areas:

1. **RAG (Retrieval-Augmented Generation)**: Enhancing LLM responses with external knowledge retrieval
2. **Chat Completion**: Examples and utilities for working with chat models

## Project Structure

```
.
├── chat_completion/    # Chat completion examples and utilities
│   ├── main.ipynb           # Jupyter notebook with examples
│   ├── langchain_examples.py # LangChain integration examples
│   └── requirements.txt     # Dependencies for chat completion
└── rag/               # Retrieval-Augmented Generation components
    ├── app_rag_pipeline.py        # Main RAG pipeline implementation
    ├── app_text_retriever.py      # Text retrieval utilities
    ├── app_text_cosine_similarity.py # Text-based similarity search
    ├── app_vector_cosine_similarity.py # Vector-based similarity search
    └── requirements.txt     # Dependencies for RAG components
```

## Features

### RAG (Retrieval-Augmented Generation)
- Document processing and chunking
- Vector embeddings for semantic search
- Multiple similarity search approaches (text-based and vector-based)
- Integration with LLMs via Together AI
- Streamlit web interface for interactive use

### Chat Completion
- Examples of using various LLM APIs
- LangChain integration examples
- Jupyter notebook with interactive examples

## Getting Started

### Prerequisites
- Python 3.8+
- API keys for Together AI and/or OpenAI (as needed)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/isf-ai.git
cd isf-ai
```

2. Set up the RAG components:
```bash
cd rag
pip install -r requirements.txt
```

3. Set up the Chat Completion components:
```bash
cd ../chat_completion
pip install -r requirements.txt
```

4. Create a `.env` file in each directory with your API keys:
```
TOGETHER_API_KEY=your_together_api_key
OPENAI_API_KEY=your_openai_api_key
```

### Running the RAG Application

```bash
cd rag
streamlit run app_rag_pipeline.py
```

### Using the Chat Completion Examples

Open and run the Jupyter notebook:
```bash
cd chat_completion
jupyter notebook main.ipynb
```

## Dependencies

### RAG Components
- sentence-transformers
- scikit-learn
- numpy
- huggingface-hub
- streamlit
- pandas
- together
- python-dotenv
- uuid

### Chat Completion
- llama-index
- ipykernel
- python-dotenv
- openai
- langchain
- together

## License

[MIT License]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 