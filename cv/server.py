import asyncio

from fastapi import FastAPI

from pipeline import CV
from client import *


app = FastAPI()

cv = CV()

@app.get("/cv/monitor")
async def monitor():
    text = await asyncio.create_task(cv.analyze_screen())

    # await send_to_chat(text)

    return {'status': 'ok'}
        
@app.on_event("startup")
async def startup_event():
    await monitor()