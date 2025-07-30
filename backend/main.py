from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.routes.youtube_router import router


app=FastAPI(
  title="ClipQ YouTube RAG API",
  version="0.1.0",
  description=(
    "Paste a YouTube URL, embed its captions into Chroma, "
    "and ask questions that are answered with Gemini."
  ),
  docs_url="/"
)
 

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["GET","POST","OPTIONS"],
  allow_headers=["*"]
)

@app.get("/ping")
def ping():
  return {
    "status":"ok"
  }


app.include_router(router,prefix="/api")

if __name__ == "__main__":          
  uvicorn.run("app.main:app",host="0.0.0.0",port=8000,reload=True)
