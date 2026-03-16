from  fastapi import FastAPI, BackgroundTasks
import httpx

import main_module


app = FastAPI()

# =INPUT=

# peripheral
app.post("/core/stt")
def add_speach():
    pass

app.post("/core/cv")
def add_speach():
    pass

app.post("/core/system_info")
def add_speach():
    pass

# chat model
app.post("/core/post_process/text")
def add_speach():
    pass