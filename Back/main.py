from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

# Se seu front estiver em outra origem, ajuste aqui:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/pose")
async def pose_endpoint(image: UploadFile = File(...)):
    # opcional: content = await image.read()
    result = random.choice(["pose correta", "pose errada"])
    return {"result": result}
