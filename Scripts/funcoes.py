from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def formata_pontos(pontos : dict):
    melhor_pontos = {}
    for k in pontos.keys():
        melhor_pontos[k] = [pontos[k]['x'], pontos[k]['y'], pontos[k]['z']]
    return melhor_pontos

def obtem_pontos_interesse(pontos : dict):
    melhor_pontos = formata_pontos(pontos)
    lista_final = []
    eixos_interesse=[['left shoulder', 'left elbow', 'left wrist'], ['right shoulder', 'right elbow', 'right wrist'], ['left elbow', 'left shoulder', 'left hip'], ['right elbow', 'right shoulder', 'right hip'], ['left shoulder', 'left hip', 'left knee'], ['right shoulder', 'right hip', 'right knee'], ['left hip', 'left knee', 'left ankle'], ['right hip', 'right knee', 'right ankle']]
    
    for eixo in eixos_interesse:
        ponto1 = melhor_pontos[eixo[0]]
        ponto2 = melhor_pontos[eixo[1]]
        ponto3 = melhor_pontos[eixo[2]]
        lista_final.append([ponto1, ponto2, ponto3])

    return lista_final

def calcula_angulo(eixo):
    ponto1 = eixo[0]
    ponto2 = eixo[1]
    ponto3 = eixo[2]

    # Calcular os vetores
    vetor1 = [ponto2[0] - ponto1[0], ponto2[1] - ponto1[1], ponto2[2] - ponto1[2]]
    vetor2 = [ponto3[0] - ponto2[0], ponto3[1] - ponto2[1], ponto3[2] - ponto2[2]]

    # Calcular o produto escalar
    produto_escalar = sum(vetor1[i] * vetor2[i] for i in range(3))

    # Calcular as magnitudes dos vetores
    magnitude_vetor1 = (sum(v ** 2 for v in vetor1)) ** 0.5
    magnitude_vetor2 = (sum(v ** 2 for v in vetor2)) ** 0.5

    # Calcular o ângulo em radianos
    angulo_rad = np.arccos(produto_escalar / (magnitude_vetor1 * magnitude_vetor2))

    # Converter para graus
    angulo_graus = np.degrees(angulo_rad)

    return angulo_graus