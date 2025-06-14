from fastapi import FastAPI, UploadFile, File, Form
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import random
import cv2
import numpy as np
from Scripts.funcoes import analise_imagem
from Scripts.runmodelo import predict_landmarks


app = FastAPI()

# Se seu front estiver em outra origem, ajuste aqui:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.post("/api/pose")
async def pose_endpoint(image: UploadFile = File(...), target_pose: str = Form(...)):
    # 1) ler bytes da imagem
    content = await image.read()
    # 2) converter para array OpenCV
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # 3) extrair landmarks (passando img e path=False)
    _, landmark_data, landmark_list = analise_imagem(img, labels=False, path=False)
    # 4) roda o modelo de previsão sobre os landmarks extraídos
    flat_landmarks = np.array(landmark_list).reshape(-1).tolist()
    # 4) validar tamanho
    if len(flat_landmarks) != 33 * 4:
        raise HTTPException(
            status_code=400,
            detail=f"Não foi possível detectar sua pose ;( )",
        )
    # 5) chamar predict e tratar erro
    try:
        predictions = predict_landmarks(flat_landmarks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 5) retorna landmarks, pose alvo e probabilidades por classe
    return {
        "predictions": predictions,
    }
