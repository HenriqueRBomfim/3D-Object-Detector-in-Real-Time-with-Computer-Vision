import cv2
import mediapipe as mp
import numpy as np

# Configurações do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Parâmetros de detecção de imobilidade
detection_frames = 5      # número de frames em sequência para considerar imóvel
movement_threshold = 0.02  # limiar de movimento das landmarks

# Função para calcular movimento entre dois conjuntos de landmarks
def pose_movement(landmarks_prev, landmarks_cur):
    if landmarks_prev is None or landmarks_cur is None:
        return np.inf
    pts_prev = np.array([[lm.x, lm.y] for lm in landmarks_prev])
    pts_cur  = np.array([[lm.x, lm.y] for lm in landmarks_cur])
    return np.mean(np.linalg.norm(pts_cur - pts_prev, axis=1))

cap = cv2.VideoCapture('../Images/Treino/dangun.mp4')

prev_landmarks = None
still_counter = 0
frame_index = 0

while cap.isOpened():
    print(f"Frame: {frame_index}")	
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    cur_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    mov = pose_movement(prev_landmarks, cur_landmarks)

    # Verifica se abaixo do limiar
    if mov < movement_threshold:
        still_counter += 1
    else:
        still_counter = 0

    # Se imóvel por X frames, salva print
    if still_counter == detection_frames:
        filename = f"pose_still_{frame_index}.png"
        # Desenha esqueleto no frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(filename, frame)
        print(f"Imagem salva: {filename}")

    prev_landmarks = cur_landmarks
    frame_index += 1

cap.release()
pose.close()
cv2.destroyAllWindows()
