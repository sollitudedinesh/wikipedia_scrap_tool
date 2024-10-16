from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusClient
)

def embed_text(content):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = content.split('.')
    embeddings = model.encode(sentences)
    return sentences, embeddings

def create_milvus_collection():
    connections.connect("default", host="localhost", port="19530")
    fields = [
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=500, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, description="Wikipedia Sentences")
    collection = Collection("wikipedia_collection", schema)
    collection.create()
    return collection

def load_embeddings_to_milvus(sentences, embeddings):
    collection = create_milvus_collection()
    data = [
        sentences,
        embeddings.tolist()
    ]
    collection.insert(data)

def search_embeddings_to_milvus(question_embedding,field,search_params,limit,expr):
    collection = create_milvus_collection()
    results = collection.search(
        data=[question_embedding],         
        anns_field=field,
        param=search_params,
        limit=limit,
        expr=expr
    )
    return results
