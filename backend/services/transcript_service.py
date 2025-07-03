from functools import lru_cache
from backend.utils.helper import fetch_transcript


@lru_cache(maxsize=32)
def get_transcript(url:str)->str:
  return fetch_transcript(url)