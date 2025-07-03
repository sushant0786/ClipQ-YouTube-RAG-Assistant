import os, re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def video_id(url: str) -> str:
    m = re.search(r"v=([a-zA-Z0-9_-]{11})", url)
    if not m: raise ValueError("Invalid YouTube URL")
    return m.group(1)

def fetch_transcript(url: str) -> str:
    vid = video_id(url)
    tx = YouTubeTranscriptApi.get_transcript(vid, languages=['en'])      # [5]
    return " ".join([s['text'] for s in tx])

def chunk(text: str, size=800, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                              chunk_overlap=overlap)
    return splitter.split_text(text)


