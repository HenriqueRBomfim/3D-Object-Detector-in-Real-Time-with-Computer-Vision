import cv2
from funcoes import draw_landmarks_on_image, calcula_todos_angulos, normalize_and_detect, analise_imagem, data_augmentation
import numpy as np
import sys
import os
import time

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

# Caminho para a pasta "ImagensModelo"
tempo_inicial = time.time()
paths = os.listdir("./Images/Poses")
for folder_path in paths[3:10]:
    nome_saida = folder_path
    folder_path = "./Images/Poses/" + folder_path

    folder_contents = os.listdir(folder_path)
    for item in folder_contents:
        print(item)
        path = folder_path + "/" + item
        data_augmentation(path, "Images/Saidas/"+nome_saida, -10, 10, 0.5, 1.5, 30, 30)
        current_time = time.time()
        print(f"Tempo de execução atual: {current_time - tempo_inicial:.2f} segundos")

tempo_final = time.time()
print(f"Tempo de execução total: {tempo_final - tempo_inicial:.2f} segundos")
# tempo_inicial = time.time()
# data_augmentation(img_path, -10, 10, 0.5, 1.5, 30, 30)
# tempo_final = time.time()
# print(f"Tempo de execução: {tempo_final - tempo_inicial:.2f} segundos")


# annotated_image, landmark_data, landmark_list = analise_imagem(img_path, labels=False)

# cv2.imshow("Resultado com Pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# dic_final,vetor = calcula_todos_angulos(landmark_data)
# print(dic_final)
# with open("Scripts/vetores.txt", "r") as file:
#     conteudo = file.read()
    
# with open("Scripts/vetores.txt", "w") as file:
#     for i in vetor:
#         conteudo = conteudo + str(i) + ','
#     conteudo = conteudo + "\n"
#     file.write(conteudo)