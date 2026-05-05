# 8001
from  fastapi import FastAPI, BackgroundTasks
import httpx
from pydantic import BaseModel

from .chat_module import ChatModule
# from .client  import 

app = FastAPI()

chat  = ChatModule(32)


class Context(BaseModel):
    context: list[dict]
@app.post("/chat/generate")
def generate_chat_response(context: Context):
    return chat(context.context)