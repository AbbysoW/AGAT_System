import logging
import asyncio

from fastapi import FastAPI, HTTPException
import httpx

from speech_pipeline import Pipeline


app = FastAPI()

pipline = Pipeline()

logger = logging.getLogger(__name__)

CHAT_URL = "http://127.0.0.1:8000"

# Block of API
async def send_to_chat(text: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                CHAT_URL + "/chat/request",
                json={"text": text}
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/stt/listen")
async def start_to_listen():
    text = await asyncio.create_task(pipline.listen())

    await send_to_chat(text)

    return {'status': 'ok'}
        


@app.on_event("startup")
async def startup_event():
    await start_to_listen()

    


