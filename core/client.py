from  fastapi import FastAPI, HTTPException
import httpx
import logging

logger = logging.getLogger(__name__)

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
    logger.debug(f"Math request | text_len={len(text)}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"Math response | status={response.status_code}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"Math error: {type(e).__name__}")
            raise HTTPException(status_code=500, detail=str(e))
        
async def send_rag(text: str): # send request to RAG module
    logger.debug(f"RAG request | text_len={len(text)}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"RAG response | status={response.status_code}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"RAG error: {type(e).__name__}")
            raise HTTPException(status_code=500, detail=str(e))

async def send_console(text: str): # send request to console module
    logger.debug(f"Console request | text_len={len(text)}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"Console response | status={response.status_code}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"Console error: {type(e).__name__}")
            raise HTTPException(status_code=500, detail=str(e))
        
# main modules
timeout = httpx.Timeout(300.0, connect=10.0)
async def send_request(context: list[dict]): # send request to chat modele
    logger.info(f"Chat request | context_len={len(context)} | timeout={timeout.read}s")
    try:
        response = await http_client.post(
            f"{CHAT_URL}/chat/generate",  
            json={
                "context": context
            },
            timeout=timeout
        )
        response.raise_for_status()
        logger.info(f"Chat response received | status={response.status_code}")
        return response.json()

    except httpx.TimeoutException:
        logger.error(f"Chat timeout (>{timeout.read}s)")
    except httpx.HTTPStatusError as e:
        logger.error(f"Chat HTTP error | code={e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Chat connection error: {type(e).__name__}")
    except Exception as e:
        logger.error(f"Chat error: {type(e).__name__}: {e}", exc_info=True)
        
    return None


