from fastapi import FastAPI, HTTPException
from scraper import scrape_wikipedia_page
from embedding import embed_text, load_embeddings_to_milvus, search_embeddings_to_milvus
from models import LoadRequest, QueryRequest
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from typing import List, Dict
import openai

app = FastAPI()

@app.post("/load")
def load_data(load_request: LoadRequest):
    try:
        content = scrape_wikipedia_page(load_request.url)
        sentences, embeddings = embed_text(content)
        load_embeddings_to_milvus(sentences, embeddings)
        return {"message": "Data loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/query")
def query_data(query_request: QueryRequest) -> Dict:
    try:
        question_embedding = model.encode(query_request.question).tolist()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        results = search_embeddings_to_milvus(question_embedding,"embedding",search_params,5,None)
 
        search_results = []
        for hits in results:
            for hit in hits:
                search_results.append(hit.entity)

        context = " ".join(search_results)
        openai.api_key = 'your-api-key-here'
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Answer the question based on the following context: {context}\nQuestion: {query_request.question}",
            max_tokens=100
        )

        return {"message": "Query completed successfully", "answer": response.choices[0].text.strip()}

    except Exception as e:
        return {"message": f"Query failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=19530)
