import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

# atalhos para desenho
mp_drawing = mp.solutions.drawing_utils
mp_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2)
mp_connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
_LM = mp.solutions.pose.PoseLandmark


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

def draw_landmarks_on_video_frame(frame: np.ndarray, detection_result) -> np.ndarray:
    """Desenha todos os pose_landmarks do resultado sobre o frame BGR in-place."""
    if not detection_result.pose_landmarks:
        return frame

    for single_landmarks in detection_result.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        for lm in single_landmarks:
            proto.landmark.add(x=lm.x, y=lm.y, z=lm.z)
        mp_drawing.draw_landmarks(
            frame, proto, POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_spec,
            connection_drawing_spec=mp_connection_spec
        )
    return frame

def formata_pontos_video(detection_result):
    """
    Converte detection_result.pose_landmarks em:
      { pessoa_idx: { 'left wrist': [x,y], ... }, ... }
    """
    formatted = {}
    if not detection_result.pose_landmarks:
        return formatted

    for pid, lm_list in enumerate(detection_result.pose_landmarks):
        pontos = {}
        for idx, lm in enumerate(lm_list):
            name = _LM(idx).name.lower().replace('_', ' ')
            pontos[name] = [lm.x, lm.y]
        formatted[pid] = pontos
    return formatted

def calcula_todos_angulos_video(detection_result):
    """
    A partir de detection_result:
      1) formata cada pessoa em dict de pontos (x,y)
      2) identifica eixos de interesse
      3) calcula ângulo em cada eixo
    Retorna:
      { pid: { 'eixo1': {..., 'angulo': …}, 'eixo2': …, … }, … }
    """
    # define quais tríades de pontos compõem cada "eixo"
    eixos = [
        ['left wrist','left elbow','left shoulder'],
        ['right wrist','right elbow','right shoulder'],
        ['left elbow','left shoulder','left hip'],
        ['right elbow','right shoulder','right hip'],
        ['left shoulder','left hip','left knee'],
        ['right shoulder','right hip','right knee'],
        ['left hip','left knee','left ankle'],
        ['right hip','right knee','right ankle']
    ]

    resultados = {}
    pontos_por_pessoa = formata_pontos_video(detection_result)

    for pid, pontos in pontos_por_pessoa.items():
        dic = {}
        for i, eixo in enumerate(eixos, start=1):
            # pega as coordenadas
            p1, p2, p3 = np.array(pontos[eixo[0]]), np.array(pontos[eixo[1]]), np.array(pontos[eixo[2]])
            # calcula vetor e ângulo
            v1, v2 = p1 - p2, p3 - p2
            cosθ = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
            ang = float(np.degrees(np.arccos(np.clip(cosθ, -1.0, 1.0))))
            # monta dict para este eixo
            label = f"eixo{i}"
            dic[label] = {
                eixo[0]: pontos[eixo[0]],
                eixo[1]: pontos[eixo[1]],
                eixo[2]: pontos[eixo[2]],
                'angulo': round(ang,3)
            }
        resultados[pid] = dic

    return resultados
