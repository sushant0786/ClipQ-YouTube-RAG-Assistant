
# ───────────────────────────────────────── embeddings
def embed_batch(texts: List[str]) -> List[List[float]]:
    """Vectorise a list of strings via embed_contents (batched call)."""
    resp = genai_client.models.embed_contents(
        model=EMBED_MODEL,
        requests=[{"content": t} for t in texts],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    # resp.embeddings is a list[EmbedContentResponse]; pick .values
    return [e.embedding for e in resp.embeddings]





class GoogEmbedder:                    # LangChain wrapper
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        single = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return single.embedding



# ───────────────────────────────────────── vector store
CHROMA_PATH = "chroma"                 # persisted on disk
vectordb = Chroma(
    collection_name="yt",
    persist_directory=CHROMA_PATH,
    embedding_function=GoogEmbedder(),  # our wrapper
)


def ingest(url: str):
    """
    Pull captions, chunk, embed with text-embedding-004,
    and upsert into Chroma.
    """
    vid   = _video_id(url)
    text  = fetch_transcript(url)
    docs  = chunk(text)
    vecs  = embed_batch([d.page_content for d in docs])

    ids   = [f"{vid}_{i}" for i in range(len(docs))]
    metas = [{"video_id": vid}] * len(docs)

    # native Chroma add (LangChain wrapper transparently calls .add)
    vectordb.add_documents(docs, ids=ids, metadatas=metas, embeddings=vecs)
    vectordb.persist()
    print(f"Ingested {len(docs)} chunks for video {vid}")


# ───────────────────────────────────────── RAG answer
def answer(url: str, question: str, k: int = 4) -> str:
    """
    Retrieve top-k transcript chunks related to the question,
    feed them to Gemini-1.5-Flash and return the answer.
    """
    vid = _video_id(url)
    rel_docs = vectordb.similarity_search(
        query=question,
        k=k,
        filter={"video_id": vid},
    )

    
    context_block = "\n".join(d.page_content for d in rel_docs)
    prompt = f"""
    You are an expert assistant. Use ONLY the context below to answer.
    If the context is insufficient, say you don't know.

    --- BEGIN CONTEXT ---
    {context_block}
    --- END CONTEXT ---

    Question: {question}
    """

    resp = genai_client.models.generate_content(
        model=GEN_MODEL,
        contents=[prompt],
        config={"max_output_tokens": 256, "temperature": 0.3},
    )
    return resp.text.strip()


def quick_summary(url: str) -> str:
    """
    Condense an entire video into a 5-bullet-point summary.
    """
    transcript = fetch_transcript(url)
    prompt = f"Summarize the following YouTube transcript in 5 concise bullet points:\n{transcript}"

    resp = genai_client.models.generate_content(
        model="gemini-pro",                       # higher-quality model
        contents=[prompt],
        config={"max_output_tokens": 128, "temperature": 0.3},
    )
    return resp.text.strip()


# ───────────────────────────────────────── manual test
if __name__ == "__main__":
    demo_url = "https://www.youtube.com/watch?v=4RixMPF4xis"

    # 1. index once
    ingest(demo_url)

    # 2. lightweight summary
    print("\n--- SUMMARY ---")
    print(quick_summary(demo_url))

    # 3. ask a question
    print("\n--- Q&A ---")
    print(answer(demo_url, "Who is singing in this video?"))






# --------


# # ───────────────────────────────────────── client
# genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# EMBEDDING_MODEL_ID = "models/embedding-001"              
# GEN_MODEL   = "gemini-2.0-flash-lite"                  


# ───────────────────────────────────────── misc helpers
