import mediapipe as mp
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from tqdm import tqdm


# atalhos para desenho
mp_drawing = mp.solutions.drawing_utils
mp_drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2)
mp_connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
_LM = mp.solutions.pose.PoseLandmark


def draw_landmarks_on_image(rgb_image, detection_result, label=True):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Lista de nomes das partes do corpo
    landmark_names = [
        "", "", "", "", "",
        "", "", "", "", "",
        "", "left shoulder", "right shoulder", "left elbow", "right elbow",
        "left wrist", "right wrist", "", "", "",
        "", "", "", "left hip", "right hip",
        "left knee", "right knee", "left ankle", "right ankle", "",
        "", "", ""
    ]

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
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )

    # Add text labels for each landmark
    for i, landmark in enumerate(pose_landmarks):
        if i < len(landmark_names):
            name = landmark_names[i]
            x = int(landmark.x * annotated_image.shape[1])
            y = int(landmark.y * annotated_image.shape[0])
            
            if label:
                # Desenha o contorno preto
                cv2.putText(
                    annotated_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA
                )
                # Desenha o texto branco por cima
                cv2.putText(
                    annotated_image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )

    # Retorna a imagem anotada após processar todos os pontos
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

def normalize_and_detect(image_np, detector, output_size=(350, 550), padding_ratio=0.1):
    """
    Normaliza a imagem com base nos pontos detectados, adiciona padding e redimensiona para um tamanho padrão,
    preservando a proporção original.
    """
    # Criar um objeto Mediapipe Image a partir do array NumPy
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

    # Detectar os pontos na imagem original
    detection_result = detector.detect(mp_image)
    if not detection_result.pose_landmarks:
        raise ValueError("Nenhum ponto detectado na imagem.")

    # Obter os pontos extremos
    landmarks = detection_result.pose_landmarks[0]
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]

    # Calcular os limites dos pontos
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Adicionar padding
    height, width, _ = image_np.shape
    padding_x = int((max_x - min_x) * width * padding_ratio)
    padding_y = int((max_y - min_y) * height * padding_ratio)

    # Calcular os limites com padding
    start_x = max(0, int(min_x * width) - padding_x)
    end_x = min(width, int(max_x * width) + padding_x)
    start_y = max(0, int(min_y * height) - padding_y)
    end_y = min(height, int(max_y * height) + padding_y)

    # Recortar a imagem com base nos limites
    cropped_image = image_np[start_y:end_y, start_x:end_x]

    # Preservar a proporção ao redimensionar
    cropped_height, cropped_width, _ = cropped_image.shape
    scale = min(output_size[1] / cropped_height, output_size[0] / cropped_width)
    new_width = int(cropped_width * scale)
    new_height = int(cropped_height * scale)

    # Redimensionar a imagem mantendo a proporção
    resized_image = cv2.resize(cropped_image, (new_width, new_height))

    # Adicionar bordas para atingir o tamanho final
    delta_w = output_size[0] - new_width
    delta_h = output_size[1] - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    # Criar um novo objeto Mediapipe Image para a imagem redimensionada
    padded_image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=padded_image)

    # Detectar novamente os pontos na imagem redimensionada
    detection_result_resized = detector.detect(padded_image_mp)

    # Retornar a imagem normalizada e os pontos recalculados
    return padded_image, detection_result_resized

def analise_imagem(img_path, labels=True, path=True):
    # Carregar a imagem
    if path:
        img = cv2.imread(img_path)
    else:
        img = img_path
    # Verificar se a imagem foi carregada corretamente
    if img is None:
        raise FileNotFoundError(f"Não foi possível carregar a imagem no caminho: {img_path}")

    # cv2.imshow("Minha Imagem", img)
    # cv2.waitKey(0)  # Espera até uma tecla ser pressionada
    # cv2.destroyAllWindows()  # Fecha a janela depois disso

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

    # Converter a imagem Mediapipe para NumPy antes de normalizar
    image_np = image.numpy_view()

    # Normalizar a imagem e detectar os pontos novamente
    normalized_image, detection_result = normalize_and_detect(image_np, detector)

    landmark_names = [
        "nose",
        "left eye (inner)",
        "left eye",
        "left eye (outer)",
        "right eye (inner)",
        "right eye",
        "right eye (outer)",
        "left ear",
        "right ear",
        "mouth (left)",
        "mouth (right)",
        "left shoulder",
        "right shoulder",
        "left elbow",
        "right elbow",
        "left wrist",
        "right wrist",
        "left pinky",
        "right pinky",
        "left index",
        "right index",
        "left thumb",
        "right thumb",
        "left hip",
        "right hip",
        "left knee",
        "right knee",
        "left ankle",
        "right ankle",
        "left heel",
        "right heel",
        "left foot index",
        "right foot index",
    ]

    # Salvar a imagem normalizada
    # cv2.imwrite("normalized_image.png", normalized_image)

    # Salvar os pontos recalculados
    if len(detection_result.pose_landmarks) > 0:
        landmarks = detection_result.pose_landmarks[0]
        landmark_data = {}
        landmark_list = []
        for idx, landmark in enumerate(landmarks):
            name = landmark_names[idx] if idx < len(landmark_names) else f"landmark {idx}"
            # print(
            #     f"{idx:02d} - {name:<20} -> x: {landmark.x:.3f}, y: {landmark.y:.3f}, z: {landmark.z:.3f}, visibility: {landmark.visibility:.2f}"
            # )
            landmark_data[name] = {
                "x": round(landmark.x, 3),
                "y": round(landmark.y, 3),
                "z": round(landmark.z, 3),
                "visibility": round(landmark.visibility, 2)
            }
            landmark_list.append((landmark.x, landmark.y, landmark.z, landmark.visibility))


        # Salvar os pontos recalculados em um arquivo
        # with open("normalized_landmarks.txt", "w") as file:
        #     for idx, landmark in enumerate(landmarks):
        #         file.write(
        #             f"{idx:02d} -> x: {landmark.x:.3f}, y: {landmark.y:.3f}, z: {landmark.z:.3f}, visibility: {landmark.visibility:.2f}\n"
        #         )

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(normalized_image, detection_result, label=labels)
        return annotated_image, landmark_data, landmark_list
    return -1,-1,-1

def data_augmentation(image_path, saida_path, min_angle, max_angle, min_scale, max_scale, min_padding, max_padding):
    """
    Realiza data augmentation na imagem, aplicando rotação, escala e padding.
    """
    image = cv2.imread(image_path)
    # cria uma imagem para cada angulo no intervalo min_angle e max_angle
    for angle in tqdm(range(min_angle, max_angle + 1), desc="Processing angles"):
        # Rotaciona a imagem
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # # Calcula as dimensões da nova imagem para garantir que nenhuma parte fique de fora
        # abs_cos = abs(rotation_matrix[0, 0])
        # abs_sin = abs(rotation_matrix[0, 1])
        # bound_w = int(height * abs_sin + width * abs_cos)
        # bound_h = int(height * abs_cos + width * abs_sin)

        # # Ajusta a matriz de rotação para o centro da nova imagem
        # rotation_matrix[0, 2] += (bound_w - width) / 2
        # rotation_matrix[1, 2] += (bound_h - height) / 2

        # # Rotaciona a imagem com as novas dimensões
        # rotated_image = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

        #cria uma imagem para cada escala no intervalo min_scale e max_scale
        for scale in np.arange(min_scale, max_scale + 0.1, 0.1):
            # Aplica escala
            scaled_image = cv2.resize(rotated_image, None, fx=scale, fy=scale)

            
            #cria uma imagem para cada combinacao de padding no intervalo min_padding e max_padding
            for padding in range(min_padding, max_padding + 1, 5):
                for left in range(0,1):
                    for right in range(0,1):
                        for top in range(0,1):
                            for bottom in range(0,1):
                                # Aplica padding
                                padded_image = cv2.copyMakeBorder(
                                    scaled_image, top*padding, bottom*padding, left*padding, right*padding, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                                )

                                annotated_image, landmark_data, landmark_list = analise_imagem(padded_image, labels=False, path=False)

                                    
                                # Mostrar a imagem com padding até que o usuário pressione 'q'
                                # cv2.imshow("Padded Image", padded_image)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()
                                
                                if landmark_list != -1:
                                    with open(saida_path, "r") as file:
                                        conteudo = file.read()
                                        
                                    with open(saida_path, "w") as file:
                                        for i in landmark_list:
                                            conteudo = conteudo + str(i) + ','
                                        conteudo = conteudo + "\n"
                                        file.write(conteudo)

    return 1