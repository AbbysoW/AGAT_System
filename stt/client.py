import logging

from fastapi import FastAPI
import httpx


logger = logging.getLogger(__name__)


app = FastAPI()

CORE_URL = "http://127.0.0.1:8000"

# OUTPUT
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(10.0, connect=2.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    trust_env=False
)

async def send_to_core(speach_info: dict):
    try:
        response = await http_client.post(
            f"{CORE_URL}/core/stt",  
            json={
                "text": speach_info['text'],
                "language": speach_info['language'],
                "speaker": speach_info['speaker']
            }
        )
        response.raise_for_status()
        return response.json()

    except httpx.TimeoutException:
        print("Core Service не ответил за 10 секунд (Таймаут)")
    except httpx.HTTPStatusError as e:
        print(f"Core Service вернул ошибку {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Ошибка сети при запросе к Core: {e}")
    except Exception as e:
        print("Непредвиденная ошибка при отправке в Core")
    
    return None

@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    


