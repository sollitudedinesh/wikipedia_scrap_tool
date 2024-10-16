from pydantic import BaseModel

class LoadRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str
