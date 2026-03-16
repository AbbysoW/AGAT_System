import logging

from fastapi import HTTPException
import httpx



logger = logging.getLogger(__name__)

CHAT_URL = "http://127.0.0.1:8000"

# Block of API
async def send_to_chat(text: str):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                CHAT_URL + "", # new adress
                json={
                    "text": text
                    }
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
        

    


