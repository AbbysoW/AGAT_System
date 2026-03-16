from  fastapi import FastAPI, HTTPException
import httpx


# OUTPUT

# Dispatcher

# side modules
async def send_math(text: str): # send request to math module
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
async def send_rag(text: str): # send request to RAG module
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))

async def send_console(text: str): # send request to console module
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))
        
# main modules
async def send_request(text: str): # send request to chat modele
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )

            return response
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=str(e))


