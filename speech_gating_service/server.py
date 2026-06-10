# 8004
import asyncio
import signal
import sys
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from speech_gating_service import SpeechGatingPipeline
from client import send_to_core, http_client
from logger import setup_logging, setup_uvicorn_logging

setup_logging()
setup_uvicorn_logging()

logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("STT SERVICE MODULE LOADING")
logger.info("=" * 50)

stt = None

# STARTUP
async def stt_worker():
    """Main STT worker task - continuously listens for speech and sends to core."""
    logger.info("🔄 STT worker task started")
    logger.debug("Worker initialization | state=ready | listening=False")
    
    iteration = 0
    total_audio_processed = 0
    total_errors = 0
    
    try:
        while True:
            iteration += 1
            logger.debug(f"Worker iteration #{iteration} | uptime_check | state=listening")
            
            try:
                start_time = time.time()
                logger.debug(f"[Iteration #{iteration}] Waiting for speech input...")
                
                speach_info = await asyncio.to_thread(stt.listen)
                
                listen_duration = time.time() - start_time
                logger.info(f"🎤 Speech captured | iteration=#{iteration} | duration={listen_duration:.2f}s | text_len={len(speach_info['text'])}")
                logger.debug(f"Speech details | speaker='{speach_info.get('speaker')}' | language='{speach_info.get('language')}'")
                logger.debug(f"Speech content: '{speach_info['text'][:100]}...' if len(speach_info['text']) > 100 else '{speach_info['text']}'")
                
                total_audio_processed += 1
                
                logger.debug(f"Preparing to send speech to core service | speaker={speach_info.get('speaker')} | lang={speach_info.get('language')}")
                send_start = time.time()
                
                await send_to_core(speach_info)
                
                send_duration = time.time() - send_start
                logger.info(f"✓ Speech sent to core | send_time={send_duration:.3f}s | iteration=#{iteration}")
                logger.debug(f"Total audio processed: {total_audio_processed} | Iteration time: {listen_duration + send_duration:.3f}s")
                    
            except asyncio.CancelledError:
                logger.info("🛑 STT worker received cancellation signal")
                logger.debug(f"Worker stats at cancellation | iteration=#{iteration} | total_audio={total_audio_processed} | errors={total_errors}")
                raise
                
            except Exception as e:
                total_errors += 1
                logger.error(f"✗ STT loop error [Iteration #{iteration}]: {type(e).__name__}: {e}", exc_info=True)
                logger.debug(f"Error recovery | total_errors={total_errors} | waiting 1s before retry...")
                await asyncio.sleep(1)
                
    except asyncio.CancelledError:
        logger.info(f"STT worker task terminated gracefully | iterations={iteration} | audio_processed={total_audio_processed} | errors={total_errors}")
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage STT service lifecycle - startup and shutdown."""
    global stt
    startup_start = time.time()
    
    logger.info("=" * 50)
    logger.info("STT SERVICE LIFESPAN: STARTUP PHASE")
    logger.info("=" * 50)
    
    try:
        logger.debug("Phase 1: Initializing STT module...")
        logger.debug(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        stt_init_start = time.time()
        stt = SpeechGatingPipeline()
        stt_init_duration = time.time() - stt_init_start
        
        logger.info(f"✓ STT instance created | init_time={stt_init_duration:.3f}s")
        logger.debug(f"STT configuration | samplerate=16000Hz | models_loaded=2")
        
        logger.debug("Phase 2: Creating worker task...")
        task_start = time.time()
        
        task = asyncio.create_task(stt_worker())
        
        task_creation_duration = time.time() - task_start
        logger.info(f"✓ STT worker task created | task_id={task.get_name()} | creation_time={task_creation_duration:.3f}s")
        logger.debug(f"Task state | done={task.done()} | cancelled={task.cancelled()}")
        
        total_startup_time = time.time() - startup_start
        logger.info(f"✓ STT SERVICE READY | total_startup_time={total_startup_time:.3f}s")
        logger.debug(f"Service status | listening=True | workers_active=1 | http_client_ready=True")
        logger.info("=" * 50)
        logger.info("SERVICE READY TO ACCEPT CONNECTIONS")
        logger.info("=" * 50)
        
        yield
        
        logger.info("=" * 50)
        logger.info("STT SERVICE LIFESPAN: SHUTDOWN PHASE")
        logger.info("=" * 50)
        
        shutdown_start = time.time()
        logger.info("🛑 STT service shutting down...")
        logger.debug("Phase 1: Stopping worker task...")
        
        task.cancel()
        logger.debug(f"Worker task cancellation signal sent | task={task.get_name()}")
        
        try:
            logger.debug("Waiting for worker task to complete cancellation...")
            cancel_start = time.time()
            await task
        except asyncio.CancelledError:
            cancel_duration = time.time() - cancel_start
            logger.debug(f"✓ Worker task cancelled successfully | cancel_time={cancel_duration:.3f}s")
            pass
        
        logger.debug("Phase 2: Closing HTTP client...")
        client_close_start = time.time()
        await http_client.aclose()
        client_close_duration = time.time() - client_close_start
        logger.debug(f"✓ HTTP client closed | close_time={client_close_duration:.3f}s")
        
        total_shutdown_time = time.time() - shutdown_start
        logger.info(f"✓ STT SERVICE STOPPED | shutdown_time={total_shutdown_time:.3f}s")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"✗ STT lifespan error: {type(e).__name__}: {e}", exc_info=True)
        logger.debug(f"Error occurred during: {'startup' if 'task' not in locals() else 'shutdown'}")
        raise

app = FastAPI(lifespan=lifespan)

@app.get("/status")
async def get_status():
    """Health check endpoint."""
    logger.debug("Health check request received")
    
    status_info = {
        "status": "running",
        "service": "STT",
        "port": 8004,
        "stt_initialized": stt is not None
    }
    
    logger.debug(f"Health check response | status={status_info['status']} | stt_ready={status_info['stt_initialized']}")
    return status_info


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event."""
    logger.info("=" * 50)
    logger.info("FASTAPI STARTUP EVENT TRIGGERED")
    logger.info("=" * 50)
    logger.debug("All initialization complete | server=ready | listening=enabled")


@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event."""
    logger.info("=" * 50)
    logger.info("FASTAPI SHUTDOWN EVENT TRIGGERED")
    logger.info("=" * 50)
    logger.debug("Cleanup in progress...")


# INPUT
@app.get("/stt/listen")
async def start_to_listen():
    """Manual speech listen endpoint."""
    logger.info("📥 Manual listen endpoint called")
    logger.debug("Processing manual listen request...")
    
    try:
        if stt is None:
            logger.warning("⚠ STT not initialized | request ignored")
            return {'status': 'error', 'message': 'STT not ready'}
        
        logger.debug("Calling stt.listen()...")
        listen_start = time.time()
        text = await asyncio.to_thread(stt.listen)
        listen_duration = time.time() - listen_start
        
        logger.info(f"✓ Speech captured via listen endpoint | duration={listen_duration:.2f}s | text_len={len(text['text'])}")
        logger.debug(f"Captured text: '{text['text'][:100]}...' if len(text['text']) > 100 else '{text['text']}'")

        logger.debug("Sending captured speech to core service...")
        send_start = time.time()
        await send_to_core(text)
        send_duration = time.time() - send_start
        
        logger.info(f"✓ Speech forwarded to core | send_time={send_duration:.3f}s")
        logger.debug(f"Request complete | total_time={listen_duration + send_duration:.3f}s")

        return {'status': 'ok', 'text': text}
        
    except Exception as e:
        logger.error(f"✗ Error in listen endpoint: {type(e).__name__}: {e}", exc_info=True)
        logger.debug(f"Request failed | error_type={type(e).__name__}")
        return {'status': 'error', 'message': str(e)}


def handle_shutdown(signum, frame):
    """Signal handler for graceful shutdown."""
    signal_name = signal.Signals(signum).name
    logger.info(f"🛑 Received {signal_name} signal - initiating graceful shutdown")
    logger.debug(f"Signal details | signum={signum} | signal_name={signal_name}")
    logger.info("Exiting STT service...")
    print("\n")
    sys.exit(0)


if __name__ == '__main__':
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("STT SERVER MAIN: INITIALIZATION")
    logger.info("=" * 50)
    logger.debug(f"Python version: {sys.version.split()[0]}")
    logger.debug(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Register signal handlers for graceful shutdown
    logger.debug("Registering signal handlers...")
    signal.signal(signal.SIGINT, handle_shutdown)
    logger.debug("✓ SIGINT (Ctrl+C) handler registered")
    
    signal.signal(signal.SIGTERM, handle_shutdown)
    logger.debug("✓ SIGTERM (terminate) handler registered")
    
    logger.info("Signal handlers registered successfully")
    
    # Configuration
    HOST = "127.0.0.1"
    PORT = 8004
    RELOAD = False
    
    logger.info(f"STT Server Configuration:")
    logger.info(f"  - Host: {HOST}")
    logger.info(f"  - Port: {PORT}")
    logger.info(f"  - Auto-reload: {RELOAD}")
    logger.info(f"  - Workers: 1")
    
    logger.info("=" * 50)
    logger.info("STARTING STT UVICORN SERVER")
    logger.info("=" * 50)
    
    try:
        logger.info(f"Launching uvicorn on {HOST}:{PORT}")
        logger.debug("Loading app module: stt.server:app")
        
        uvicorn.run(
            app,
            host=HOST,
            port=PORT,
            reload=RELOAD
        )
        
    except KeyboardInterrupt:
        logger.info("\nSTT service interrupted by user")
    except Exception as e:
        logger.error(f"✗ Server error: {type(e).__name__}: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        logger.info("=" * 50)
        logger.info("STT SERVER SHUTDOWN COMPLETE")
        logger.info("=" * 50)