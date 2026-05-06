import asyncio
import logging

from fastapi import FastAPI

from pipeline import CV
from client import *

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    cv = CV()
    logger.info("CV service initialized")
except Exception as e:
    logger.error(f"CV service init error: {type(e).__name__}: {e}", exc_info=True)
    raise

@app.get("/cv/monitor")
async def monitor():
    logger.info("CV monitor started")
    try:
        text = await asyncio.create_task(cv.analyze_screen())
        logger.info(f"CV analysis complete | text_len={len(str(text))}")

        # await send_to_chat(text)

        return {'status': 'ok'}
    except Exception as e:
        logger.error(f"CV monitor error: {type(e).__name__}: {e}", exc_info=True)
        raise
        
@app.on_event("startup")
async def startup_event():
    logger.info("CV startup event")
    await monitor()