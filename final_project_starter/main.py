from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(title="RAG Pipeline API")


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/retrieve")
def retrieve(request: QueryRequest):
    try:
        # TODO: Instantiate and run the retriever
        chunks = ["chunk 1", "chunk 2", "chunk 3", "chunk 4", "chunk 5"]
        return {"query": request.query, "results": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate(request: QueryRequest):
    try:
        # TODO: Instantiate and run the generator
        response = "This is a test response"
        return {"query": request.query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
