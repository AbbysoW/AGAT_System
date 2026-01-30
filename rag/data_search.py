from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],   
    allow_headers=["*"],
)

class RAGReqsest(BaseModel):
    text:  str

@app.post("/rag/get_info")
async def get_info(reqest: RAGReqsest):
    # Some logic for rag

    return {"info": [
        "Article1",
        "Article2",
        "Article3"
        ]}