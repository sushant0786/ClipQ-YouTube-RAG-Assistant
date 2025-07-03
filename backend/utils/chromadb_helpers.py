import chromadb
from backend.utils.embedding import GeminiEmbeddingFunction



def create_chroma_db(documents,name):
  chroma_client = chromadb.Client()
  db=chroma_client.create_collection(
    name=name,
    embedding_function=GeminiEmbeddingFunction()
  )

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db


def topk_relevent_chunks(query,db,topk):
  return db.query(query_texts=[query],n_results=topk)["documents"][0]
