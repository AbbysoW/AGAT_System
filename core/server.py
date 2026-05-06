# 8000
from  fastapi import FastAPI, BackgroundTasks
import httpx

from pydantic import BaseModel

from .main_module import Core



app = FastAPI()

# INPUT
core = Core()


# peripheral
class STT_INPUT(BaseModel):
    text: str
    language: str
    speaker: str
@app.post("/core/stt")
def add_speach(speach_info: STT_INPUT, background_tasks: BackgroundTasks):
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