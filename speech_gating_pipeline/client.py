import logging

from fastapi import FastAPI
import httpx


logger = logging.getLogger(__name__)

logger.debug("STT client module initializing...")

app = FastAPI()

CORE_URL = "http://127.0.0.1:8000"
logger.debug(f"Core service URL configured: {CORE_URL}")

# OUTPUT
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=2.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    trust_env=False
)
logger.debug("HTTP client initialized | timeout=10s | connect_timeout=2s | max_connections=10")

async def send_to_core(speach_info: dict):
    logger.info(f"📤 Sending speech to core service | lang={speach_info['language']} | text_len={len(speach_info['text'])}")
    logger.debug(f"Speech info: {speach_info}")
    
    try:
        logger.debug(f"Connecting to Core service at {CORE_URL}/core/stt")
        response = await http_client.post(
            f"{CORE_URL}/core/stt",  
            json={
                "text": speach_info['text'],
                "language": speach_info['language'],
                "speaker": speach_info['speaker']
            }
        )
        response.raise_for_status()
        logger.info(f"✓ Speech sent to core | status={response.status_code}")
        logger.debug(f"Core response: {response.json()}")
        return response.json()

    except httpx.TimeoutException:
        logger.error(f"✗ Core service timeout (>10s) - Service may be unresponsive")
    except httpx.HTTPStatusError as e:
        logger.error(f"✗ Core service HTTP error | code={e.response.status_code} | reason={e.response.reason_phrase}")
    except httpx.RequestError as e:
        logger.error(f"✗ Core service connection error: {type(e).__name__} - Cannot reach {CORE_URL}")
    except Exception as e:
        logger.error(f"✗ Core send error: {type(e).__name__}: {e}", exc_info=True)
    
    logger.warning("⚠ Returning None due to core service error")
    return None


if __name__ == '__main__':
    import asyncio

    async def test():
        while True:
            # Run blocking input() in a background thread so it doesn't block the async loop
            text = await asyncio.to_thread(input, "Type text\n")

            speach_info = {
                "text": text,
                "language": "ru",
                "speaker": "Владелец"
            }

            await send_to_core(speach_info)

    # Run the entire application inside a single event loop
    asyncio.run(test())


