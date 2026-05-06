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

app = FastAPI()

# INPUT
try:
    core = Core()
    logger.info("Core service initialized")
except Exception as e:
    logger.error(f"Core init error: {type(e).__name__}: {e}", exc_info=True)
    raise


# peripheral
class STT_INPUT(BaseModel):
    text: str
    language: str
    speaker: str

@app.post("/core/stt")
def add_speach(speach_info: STT_INPUT, background_tasks: BackgroundTasks):
    logger.debug(f"Received STT POST | lang={speach_info.language} | text_len={len(speach_info.text)}")
    background_tasks.add_task(core.add_stt_input, speach_info.text, speach_info.speaker, speach_info.language)
    return {'status': 'OK'}

@app.post("/core/cv")
def add_speach():
    pass

@app.post("/core/system_info")
def add_speach():
    pass

# chat model
@app.post("/core/post_process/text")
def add_speach():
    pass