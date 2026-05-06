# 8004
import asyncio
import signal
import sys

from fastapi import FastAPI
from contextlib import asynccontextmanager

from .stt_module import STT
from .client import send_to_core, http_client

stt = STT()

# STATUP
async def stt_worker():
    print("STT Worker запущен...")
    try:
        while True:
            try:
                speach_info = await asyncio.to_thread(stt.listen)
                
                await send_to_core(speach_info)
                    
            except Exception as e:
                print(f"Ошибка в цикле STT: {e}")
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        print("STT Worker отменён")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(stt_worker())
    
    yield

    # Корректное завершение задачи
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Закрываем http_client
    await http_client.aclose()
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


def handle_shutdown(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    print("\nПолучен сигнал завершения, выключаемся...")
    sys.exit(0)


if __name__ == '__main__':
    import uvicorn
    
    # Регистрируем обработчики сигналов для Ctrl+C
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    try:
        uvicorn.run(
            "stt.server:app", 
            host="127.0.0.1", 
            port=8004,
            reload=False
        )
    except KeyboardInterrupt:
        print("\nПриложение завершено пользователем")
    finally:
        print("Очистка ресурсов...")