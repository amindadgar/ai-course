# RAG FastAPI Starter

## Setup

1. Create a Python 3.9+ virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   .\\venv\\Scripts\\activate  # Windows
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your TogetherAI API key
   # TOGETHER_API_KEY=your_api_key
   ```

## Running

Start the FastAPI server:

```bash
fastapi run main.py
```

- **/retrieve** endpoint: POST to `http://localhost:8000/retrieve` with JSON `{"query": "Your question", "top_k": 5}`
- **/generate** endpoint: POST to `http://localhost:8000/generate` with JSON `{"query": "Your question", "top_k": 5}`
