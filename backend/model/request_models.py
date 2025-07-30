from pydantic import BaseModel

class UrlBody(BaseModel):
  url:str



class AskBody(BaseModel):
  url:str
  question:str
  top_k:int=2
