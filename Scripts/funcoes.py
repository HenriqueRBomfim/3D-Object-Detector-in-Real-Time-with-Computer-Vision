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
        melhor_pontos[k] = [pontos[k]['x'], pontos[k]['y']]
    return melhor_pontos

def obtem_pontos_interesse(melhor_pontos : dict):
    lista_final = []
    eixos_interesse=[['left wrist', 'left elbow', 'left shoulder'], ['right wrist', 'right elbow', 'right shoulder'], ['left elbow', 'left shoulder', 'left hip'], ['right elbow', 'right shoulder', 'right hip'], ['left shoulder', 'left hip', 'left knee'], ['right shoulder', 'right hip', 'right knee'], ['left hip', 'left knee', 'left ankle'], ['right hip', 'right knee', 'right ankle']]
    
    for eixo in eixos_interesse:
        ponto1 = melhor_pontos[eixo[0]]
        ponto2 = melhor_pontos[eixo[1]]
        ponto3 = melhor_pontos[eixo[2]]
        lista_final.append([ponto1, ponto2, ponto3])

    return lista_final, eixos_interesse

def calcula_angulo(eixo):
  ponto1, ponto2, ponto3 = np.array(eixo[0]), np.array(eixo[1]), np.array(eixo[2])
  
  vetor1 = ponto1 - ponto2
  vetor2 = ponto3 - ponto2
  
  produto_escalar = np.dot(vetor1, vetor2)
  magnitude1 = np.linalg.norm(vetor1)
  magnitude2 = np.linalg.norm(vetor2)
  
  cos_theta = produto_escalar / (magnitude1 * magnitude2)
  angulo = np.arccos(np.clip(cos_theta, -1.0, 1.0))
  
  return round(np.degrees(angulo), 3)

def calcula_todos_angulos(landmark_data):
    pontos_formatados = formata_pontos(landmark_data)
    pontos_interesse, legenda = obtem_pontos_interesse(pontos_formatados)
    dic_final ={}
    i = 1
    vetor = []
    for indice,eixo in enumerate(pontos_interesse):
        label = "eixo" + str(i)
        dic_final[label] = {}
        dic_final[label][legenda[indice][0]] = pontos_formatados[legenda[indice][0]]
        dic_final[label][legenda[indice][1]] = pontos_formatados[legenda[indice][1]]
        dic_final[label][legenda[indice][2]] = pontos_formatados[legenda[indice][2]]
        dic_final[label]['angulo'] = calcula_angulo(eixo)
        i += 1
        vetor.append(dic_final[label]['angulo'])
        vetor.append(pontos_formatados[legenda[indice][0]])
        vetor.append(pontos_formatados[legenda[indice][1]])
        vetor.append(pontos_formatados[legenda[indice][2]])

    return dic_final, vetor