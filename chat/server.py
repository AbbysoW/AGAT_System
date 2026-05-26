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

logger.info("=" * 60)
logger.info("CHAT SERVICE INITIALIZATION")
logger.info("=" * 60)

app = FastAPI()

try:
    logger.info("Creating ChatModule instance...")
    chat = ChatModule()
    logger.info("✓ Chat service initialized successfully")
    logger.info("Server ready to accept requests on port 8001")
except Exception as e:
    logger.error(f"✗ Chat service init error: {type(e).__name__}: {e}", exc_info=True)
    raise

logger.info("=" * 60)


class Context(BaseModel):
    context: list[dict]

@app.post("/chat/generate")
def generate_chat_response(context: Context):
    logger.info(f"📥 Chat generation request | context_len={len(context.context)}")
    logger.debug(f"Context structure: {[list(el.keys()) for el in context.context]}")
    
    try:
        logger.debug("Calling chat model for generation...")
        result = chat(context.context)
        
        logger.info(f"✓ Chat response generated | response_len={len(str(result))}")
        logger.debug(f"Response preview: {str(result)[:150]}..." if len(str(result)) > 150 else f"Response: {result}")
        
        return result
    except Exception as e:
        logger.error(f"✗ Chat generate error: {type(e).__name__}: {e}", exc_info=True)
        raise

@app.on_event("startup")
def startup_event():
    logger.info("=" * 60)
    logger.info("CHAT SERVICE STARTUP COMPLETED")
    logger.info("=" * 60)

@app.on_event("shutdown")
def shutdown_event():
    logger.info("=" * 60)
    logger.info("CHAT SERVICE SHUTTING DOWN")
    logger.info("=" * 60)