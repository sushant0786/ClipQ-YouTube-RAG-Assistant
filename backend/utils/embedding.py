
from backend.utils.genai_clent import get_client
from google.genai import types
from chromadb import EmbeddingFunction, Documents, Embeddings

EMBEDDING_MODEL_ID = "models/embedding-001"

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        client = get_client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL_ID,
            contents=input,
            config=types.EmbedContentConfig(
                task_type="retrieval_document",
                title="Custom Query"
            )
        )
        return response.embeddings[0].values
