import asyncio

from fastapi import FastAPI

from pipeline import Pipeline
from client import send_to_chat


app = FastAPI()

pipline = Pipeline()


@app.get("/stt/listen")
async def start_to_listen():
    text = await asyncio.create_task(pipline.listen())

    await send_to_chat(text)

    return {'status': 'ok'}
        
@app.on_event("startup")
async def startup_event():
    await start_to_listen()
