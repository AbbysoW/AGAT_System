from multiprocessing import Process
import uvicorn

from logger import setup_logging, setup_uvicorn_logging

def chat():
    setup_logging()
    setup_uvicorn_logging()
    uvicorn.run(
        "chat.chat:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=False, 
        log_config=None, 
        access_log=False
        )

def rag():
    setup_logging()
    setup_uvicorn_logging()
    uvicorn.run(
        "rag.data_search:app", 
        host="127.0.0.1", 
        port=8001, 
        reload=False, 
        log_config=None, 
        access_log=False
        )

def chat_model():
    setup_logging()
    setup_uvicorn_logging()
    uvicorn.run(
        "models.chat_model.chat_model:app", 
        host="127.0.0.1", 
        port=8002, 
        reload=False, 
        log_config=None, 
        access_log=False
        )


if __name__ == "__main__":
    # Создаем процессы для каждого сервиса
    p1 = Process(target=chat)
    p2 = Process(target=rag)
    p3 = Process(target=chat_model)

    # Запускаем их параллельно
    p1.start()
    p2.start()
    p3.start()

    print("test  point")

    # Ждем завершения (это обычно будет бесконечно, пока сервисы работают)
    p1.join()
    p2.join()
    p3.join()