from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from ml.recognition import compare_images


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

server = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})



@server.get("/")
async def root():
    return {"message": "Hello World"}

@server.post("/")
async def read_root(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    
    return {"samePerson": compare_images(image)}

def run_server():
    uvicorn.run(server, host="0.0.0.0", port=30_000)
