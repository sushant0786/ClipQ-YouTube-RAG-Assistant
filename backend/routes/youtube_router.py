from fastapi import APIRouter,HTTPException
from backend.model.request_models import UrlBody,AskBody
from backend.services.transcript_service import get_transcript
from backend.services.gemni_service import gemini_client
from backend.services.chroma_service import get_collection, retrieve_chunks
from typing import List

router=APIRouter()

MODEL_ID = "gemini-2.0-flash-lite"

@router.post("/summarise")
async def summarise(body:UrlBody):
  transcript=get_transcript(body.url)

  prompt = (
      "Summarise the following transcript into concise, high-level bullet points. "
      "Focus on key facts and actionable insights.\n\n"
      f"{transcript}"
  )


  try:
    summary=gemini_client().models.generate_content(
      model=MODEL_ID,
      contents=prompt
    ).text
  except Exception as e:
    raise HTTPException(status_code=500,detail=f"Gemini error: {e}")
  
  return {"summary": summary}



@router.post("/ask")
async def ask(body:AskBody):
  collection=get_collection(body.url)

  try:
    top_k_chunks:List[str] = retrieve_chunks(body.question,collection,body.top_k)
  except Exception as e:
        raise HTTPException(status_code=500,detail=f"Chroma retrieval error: {e}")
  
  context_block = "\n\n".join(top_k_chunks)
  prompt = (
    "Use the context below to answer the question. "
    "If the answer cannot be found, say so.\n\n"
    f"Context:\n{context_block}\n\n"
    f"Question: {body.question}\n\nAnswer:"
  )
  
  try:
    answer = gemini_client().models.generate_content(
       model=MODEL_ID,
       contents=prompt
    ).text
  except Exception as e:
    raise HTTPException(status_code=500,detail=f"Gemini error: {e}")

  return {"answer": answer, "context": top_k_chunks}