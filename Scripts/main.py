import cv2
from funcoes import draw_landmarks_on_image, calcula_todos_angulos, normalize_and_detect, analise_imagem
import numpy as np
import sys
import os

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

""" Legenda:

0 - nose
1 - left eye (inner)
2 - left eye
3 - left eye (outer)
4 - right eye (inner)
5 - right eye
6 - right eye (outer)
7 - left ear
8 - right ear
9 - mouth (left)
10 - mouth (right)
11 - left shoulder
12 - right shoulder
13 - left elbow
14 - right elbow
15 - left wrist
16 - right wrist
17 - left pinky
18 - right pinky
19 - left index
20 - right index
21 - left thumb
22 - right thumb
23 - left hip
24 - right hip
25 - left knee
26 - right knee
27 - left ankle
28 - right ankle
29 - left heel
30 - right heel
31 - left foot index
32 - right foot index

"""
# Detecta o sistema operacional
is_windows = sys.platform == "win32"
is_mac = sys.platform == "darwin"

if is_windows:
    img_path = r".\Images\Treino\Dangun\Pose 1\padding_dois_lados_video2_dangun_1.png"
elif is_mac:
    img_path = "./Images/Treino/padding_dois_lados_video2_dangun_1.png"
else:
    raise Exception("Sistema operacional não suportado")

annotated_image, landmark_data = analise_imagem(img_path, labels=False)

cv2.imshow("Resultado com Pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

dic_final,vetor = calcula_todos_angulos(landmark_data)
print(dic_final)
with open("Scripts/vetores.txt", "r") as file:
    conteudo = file.read()
    
with open("Scripts/vetores.txt", "w") as file:
    for i in vetor:
        conteudo = conteudo + str(i) + ','
    conteudo = conteudo + "\n"
    file.write(conteudo)