# 8000
import logging

from fastapi import FastAPI, BackgroundTasks
import httpx
from pydantic import BaseModel

from main_module import Core
from logger import setup_logging, setup_uvicorn_logging

setup_logging()
setup_uvicorn_logging()

logger = logging.getLogger(__name__)

logger.info("=" * 60)
logger.info("CORE SERVICE INITIALIZATION")
logger.info("=" * 60)

app = FastAPI()

# INPUT
try:
    logger.info("Creating Core instance...")
    core = Core()
    logger.info("✓ Core service initialized successfully")
    logger.info("Server ready to accept requests on port 8000")
except Exception as e:
    logger.error(f"✗ Core init error: {type(e).__name__}: {e}", exc_info=True)
    raise

logger.info("=" * 60)


# peripheral
class STT_INPUT(BaseModel):
    text: str
    language: str
    speaker: str

@app.post("/core/stt")
def add_speach(speach_info: STT_INPUT, background_tasks: BackgroundTasks):
    logger.info(f"📥 STT POST request received | lang={speach_info.language} | text_len={len(speach_info.text)}")
    logger.debug(f"STT request details | speaker={speach_info.speaker} | text='{speach_info.text[:100]}...' if len(speach_info.text) > 100 else '{speach_info.text}'")
    
    try:
        background_tasks.add_task(
            core.add_stt_input,
            speach_info.text,
            speach_info.speaker,
            speach_info.language
        )
        logger.debug("✓ STT background task added to queue")
        return {'status': 'OK'}
    except Exception as e:
        logger.error(f"✗ Error adding STT task: {type(e).__name__}: {e}", exc_info=True)
        return {'status': 'ERROR', 'detail': str(e)}

@app.post("/core/cv")
def add_cv_input():
    logger.info("📥 CV POST request received (not yet implemented)")
    logger.debug("CV module functionality pending implementation")
    pass

@app.post("/core/system_info")
def add_system_info():
    logger.info("📥 System info POST request received (not yet implemented)")
    logger.debug("System info module functionality pending implementation")
    pass

# chat model
@app.post("/core/post_process/text")
def post_process_text():
    logger.info("📥 Post-process text POST request received (not yet implemented)")
    logger.debug("Post-process functionality pending implementation")
    pass

@app.on_event("startup")
def startup_event():
    logger.info("=" * 60)
    logger.info("CORE SERVICE STARTUP COMPLETED")
    logger.info("=" * 60)

@app.on_event("shutdown")
def shutdown_event():
    logger.info("=" * 60)
    logger.info("CORE SERVICE SHUTTING DOWN")
    logger.info("=" * 60)