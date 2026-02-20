import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
from pydantic import BaseModel

# from data_processing import preprocess, seq_to_ids, ids_to_seq, make_padding_mask
# from vocabulary import vocabulary

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],   
    allow_headers=["*"],
)

context = "context"  # Placeholder for context data

# RAG
async def get_info(text: str): # Searching for information in RAG
    # Placeholder for RAG model interaction
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8001/rag/get_info",
                json={"text": text}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="RAG model error")

            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
# DISPLAY

#Some APIs to  display model responce


async def req_to_model(text: str, context: str):
    text_with_role = "<USE>" + text + "<EOS>"
    text_tokens = preprocess(text_with_role)
    text_ids = seq_to_ids(text_tokens, vocabulary)

    rag_info_list = (await get_info(text))["info"]
    rag_info = "\n".join(rag_info_list)
    rag_with_role = "<SYS>" + rag_info + "<EOS>"
    rag_info_tokens = preprocess(rag_with_role)
    rag_info_ids = seq_to_ids(rag_info_tokens, vocabulary)

    context_tokens = preprocess(context)
    context_ids = seq_to_ids(context_tokens, vocabulary)

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8002/model/get_responce", # Placeholder URL
                json={
                    "text": text_ids, 
                    "rag_info": rag_info_ids,
                    "context": context_ids
                    }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Model error")
            
            context += " " + text_with_role + " " + response.json()["response"]

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))

class STTReqsest(BaseModel):
    text: str

@app.post("/chat/request")
async def reqest_from_stt(reqest: STTReqsest):
    asyncio.create_task(req_to_model(reqest.text, context))

    return {"status": "ok"}
