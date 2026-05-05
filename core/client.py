from  fastapi import FastAPI, HTTPException
import httpx

CHAT_URL = "http://127.0.0.1:8001"

# OUTPUT
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=2.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    trust_env=False
)

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
async def send_request(context: list[dict]): # send request to chat modele
    try:
        response = await http_client.post(
            f"{CHAT_URL}/chat/generate",  
            json={
                "context": context
            }
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        print("Chat Service не ответил за 10 секунд (Таймаут)")
    except httpx.HTTPStatusError as e:
        print(f"Chat Service вернул ошибку {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Ошибка сети при запросе к Chat: {e}")
    except Exception as e:
        print("Непредвиденная ошибка при отправке в Chat")
        
    return None


