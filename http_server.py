from fastapi import FastAPI
import uvicorn

server = FastAPI()



@server.get("/")
async def root():
    return {"message": "Hello World"}

def run_server():
    uvicorn.run(server, host="0.0.0.0", port=30_000)
