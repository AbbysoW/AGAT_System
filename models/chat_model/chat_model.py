from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from typing import List


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],   
    allow_headers=["*"],
)

class ModelReqsest(BaseModel):
    text: List[int]
    rag_info: List[int]
    context: List[int]

@app.post("/model/get_responce")
async def get_responce(reqest: ModelReqsest):
    # Some logic for model

    return {"response": "response"}