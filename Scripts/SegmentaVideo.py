import cv2
from funcoes import draw_landmarks_on_image, calcula_todos_angulos
import numpy as np
import sys

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture('../Images/Treino/dangun.mp4')

if not cap.isOpened():
    print("Erro ao abrir vídeo ou câmera")


while True:
    ret, frame = cap.read()
    if not ret:
        # chegou ao fim do vídeo ou falha na captura
        break

    cv2.imshow('Reprodução', frame)

    # sai ao pressionar 'q'
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
