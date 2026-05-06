import logging

from fastapi import FastAPI
import httpx


logger = logging.getLogger(__name__)


app = FastAPI()

CORE_URL = "http://127.0.0.1:8000"

# OUTPUT
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=2.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    trust_env=False
)

async def send_to_core(speach_info: dict):
    logger.debug(f"Sending speech to core | text_len={len(speach_info['text'])}")
    try:
        response = await http_client.post(
            f"{CORE_URL}/core/stt",  
            json={
                "text": speach_info['text'],
                "language": speach_info['language'],
                "speaker": speach_info['speaker']
            }
        )
        response.raise_for_status()
        logger.info(f"Speech sent to core | status={response.status_code}")
        return response.json()

    except httpx.TimeoutException:
        logger.error(f"Core timeout (>10s)")
    except httpx.HTTPStatusError as e:
        logger.error(f"Core HTTP error | code={e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(f"Core connection error: {type(e).__name__}")
    except Exception as e:
        logger.error(f"Core send error: {type(e).__name__}: {e}", exc_info=True)
    
    return None




