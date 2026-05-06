# 8001
import logging
from fastapi import FastAPI, BackgroundTasks
import httpx
from pydantic import BaseModel

from chat_module import ChatModule
from logger import setup_logging, setup_uvicorn_logging

setup_logging()
setup_uvicorn_logging()

logger = logging.getLogger(__name__)

app = FastAPI()

try:
    chat = ChatModule()
    logger.info("Chat service initialized")
except Exception as e:
    logger.error(f"Chat service init error: {type(e).__name__}: {e}", exc_info=True)
    raise


class Context(BaseModel):
    context: list[dict]

@app.post("/chat/generate")
def generate_chat_response(context: Context):
    logger.info(f"Chat request | context_len={len(context.context)}")
    try:
        result = chat(context.context)
        logger.info(f"Chat response | result_len={len(str(result))}")
        return result
    except Exception as e:
        logger.error(f"Chat generate error: {type(e).__name__}: {e}", exc_info=True)
        raise