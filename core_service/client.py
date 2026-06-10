import logging

from  fastapi import FastAPI, HTTPException
import httpx

logger = logging.getLogger(__name__)

CHAT_URL = "http://chat_service:8001"

logger.debug(f"Chat service URL configured: {CHAT_URL}")

# OUTPUT
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=2.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    trust_env=False
)
logger.debug("HTTP client initialized | timeout=10s | connect_timeout=2s | max_connections=10")

# Dispatcher

# side modules
async def send_math(text: str): # send request to math module
    logger.info(f"📤 Math module request | text_len={len(text)}")
    logger.debug(f"Math request text: '{text[:100]}...' if len(text) > 100 else '{text}'")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"✓ Math response received | status={response.status_code}")
            logger.debug(f"Math response details: {response.json()}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"✗ Math module error: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        
async def send_rag(text: str): # send request to RAG module
    logger.info(f"📤 RAG module request | text_len={len(text)}")
    logger.debug(f"RAG request text: '{text[:100]}...' if len(text) > 100 else '{text}'")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"✓ RAG response received | status={response.status_code}")
            logger.debug(f"RAG response details: {response.json()}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"✗ RAG module error: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

async def send_console(text: str): # send request to console module
    logger.info(f"📤 Console module request | text_len={len(text)}")
    logger.debug(f"Console request text: '{text[:100]}...' if len(text) > 100 else '{text}'")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "", # new adress
                json={
                    "requets": text
                    }
            )
            logger.info(f"✓ Console response received | status={response.status_code}")
            logger.debug(f"Console response details: {response.json()}")
            return response
        except httpx.HTTPError as e:
            logger.error(f"✗ Console module error: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        
# main modules
timeout = httpx.Timeout(300.0, connect=10.0)
logger.debug(f"Chat service timeout configured | read_timeout={timeout.read}s | connect_timeout={timeout.connect}s")

async def send_request(context: list[dict]): # send request to chat modele
    logger.info(f"📤 Chat service request | context_len={len(context)} | timeout={timeout.read}s")
    logger.debug(f"Context structure | elements: {[list(el.keys()) for el in context]}")
    
    try:
        logger.debug(f"Connecting to Chat service at {CHAT_URL}/chat/generate")
        
        response = await http_client.post(
            f"{CHAT_URL}/chat/generate",  
            json={
                "context": context
            },
            timeout=timeout
        )
        response.raise_for_status()
        logger.info(f"✓ Chat response received | status={response.status_code}")
        
        result = response.json()
        logger.debug(f"Chat response parsed | response_type={type(result)} | response_len={len(str(result))}")
        
        return result

    except httpx.TimeoutException:
        logger.error(f"✗ Chat service timeout (>{timeout.read}s) - Service may be unresponsive")
    except httpx.HTTPStatusError as e:
        logger.error(f"✗ Chat service HTTP error | code={e.response.status_code} | reason={e.response.reason_phrase}")
    except httpx.RequestError as e:
        logger.error(f"✗ Chat service connection error: {type(e).__name__} - Cannot reach {CHAT_URL}")
    except Exception as e:
        logger.error(f"✗ Chat service error: {type(e).__name__}: {e}", exc_info=True)
        
    logger.warning("⚠ Returning None due to chat service error")
    return None


