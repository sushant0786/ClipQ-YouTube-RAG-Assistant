from functools import lru_cache
from backend.utils.helper import chunk,video_id
from backend.utils.chromadb_helpers import create_chroma_db,topk_relevent_chunks
from backend.services.transcript_service import get_transcript
from typing import List

@lru_cache(maxsize=32)
def get_collection(url:str):
  transcript=get_transcript(url)
  chunks:List[str]=chunk(transcript)
  return create_chroma_db(chunks,name=video_id(url))

def retrieve_chunks(question:str,collection,top_k:int=4)->List[str]:
  return topk_relevent_chunks(question, collection, top_k)
