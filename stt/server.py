import asyncio

from fastapi import FastAPI

from pipeline import STT
from client import send_to_chat


app = FastAPI()

stt = STT()


@app.get("/stt/listen") # ПЕРЕПИСАТЬ не в цикле
async def start_to_listen():
    text = await asyncio.create_task(stt.listen())

    await send_to_chat(text)

    return {'status': 'ok'}
        
@app.on_event("startup")
async def startup_event():
    await start_to_listen()
