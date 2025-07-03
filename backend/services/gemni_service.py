from functools import lru_cache
from backend.utils.genai_clent import get_client   # your existing file

@lru_cache(maxsize=1)
def gemini_client():
    return get_client()
