import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio

CHAT_URL = "http://127.0.0.1:8000"

logger = logging.getLogger(__name__)

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


# STT model

async def get_text(audio):
    while(True):
        try:
            stt_result = "Text_placeholder" # Placeholder while no model
            await send_to_chat(stt_result)
        
        except: # Runtime error
            pass

if __name__ == "__main__":

    async def main():
        responce = await send_to_chat("Text_placeholder")

        print(responce.status_code)
        print(responce.json())


    asyncio.run(main())
