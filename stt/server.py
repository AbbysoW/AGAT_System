# 8004
import asyncio
import signal
import sys
import logging

from fastapi import FastAPI
from contextlib import asynccontextmanager

from stt_module import STT
from client import send_to_core, http_client
from logger import setup_logging, setup_uvicorn_logging

setup_logging()
setup_uvicorn_logging()

logger = logging.getLogger(__name__)

stt = None

# STATUP
async def stt_worker():
    logger.info("STT worker started")
    try:
        while True:
            try:
                logger.debug("STT listening...")
                speach_info = await asyncio.to_thread(stt.listen)
                logger.info(f"Speech received | text_len={len(speach_info['text'])}")
                
                await send_to_core(speach_info)
                logger.debug("Speech sent to core")
                    
            except asyncio.CancelledError:
                logger.info("STT worker cancelled")
                raise
            except Exception as e:
                logger.error(f"STT loop error: {type(e).__name__}: {e}", exc_info=True)
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt
    try:
        logger.info("STT server starting")
        stt = STT()
        logger.info("STT instance created")
        
        task = asyncio.create_task(stt_worker())
        logger.info("STT worker task created")
        
        yield
        
        logger.info("STT server shutting down")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        await http_client.aclose()
        logger.info("STT server stopped")
    except Exception as e:
        logger.error(f"STT lifespan error: {type(e).__name__}: {e}", exc_info=True)
        raise

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