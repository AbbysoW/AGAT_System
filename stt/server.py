import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager

from pipeline import STT
from client import send_to_core

stt = STT()

# STATUP
async def stt_worker():
    print("STT Worker запущен...")
    while True:
        try:
            text = await asyncio.to_thread(stt.listen)
            print('got text')
            
            if text:
                print('send')
                await send_to_core(text)
                
        except Exception as e:
            print(f"Ошибка в цикле STT: {e}")
            await asyncio.sleep(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(stt_worker())
    
    yield

    task.cancel()
    print("STT Worker остановлен")

app = FastAPI(lifespan=lifespan)

@app.get("/status")
async def get_status():
    return {"status": "running"}


# INPUT
@app.get("/stt/listen")
async def start_to_listen():
    text = await asyncio.to_thread(stt.listen)

    await send_to_core(text)

    return {'status': 'ok'}


if __name__ == '__main__':
    import uvicorn


    uvicorn.run(
        "server:app", 
        host="127.0.0.1", 
        port=8000, 
        )