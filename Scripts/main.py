import cv2
from funcoes import draw_landmarks_on_image, calcula_todos_angulos
import numpy as np
import sys

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
    img_path = r".\Images\Treino\ImagemInteira.png"
elif is_mac:
    img_path = "./Images/Treino/ImagemInteira.png"
else:
    raise Exception("Sistema operacional não suportado")
img = cv2.imread(img_path)
# cv2.imshow("Minha Imagem", img)
cv2.waitKey(0)  # Espera até uma tecla ser pressionada
cv2.destroyAllWindows()  # Fecha a janela depois disso

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=True
)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)
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

landmarks = detection_result.pose_landmarks[0]

landmark_data = {}

for idx, landmark in enumerate(landmarks):
    name = landmark_names[idx] if idx < len(landmark_names) else f"landmark {idx}"
    print(
        f"{idx:02d} - {name:<20} -> x: {landmark.x:.3f}, y: {landmark.y:.3f}, z: {landmark.z:.3f}, visibility: {landmark.visibility:.2f}"
    )
    landmark_data[name] = {
        "x": round(landmark.x, 3),
        "y": round(landmark.y, 3),
        "z": round(landmark.z, 3),
        "visibility": round(landmark.visibility, 2)
    }

dic_final = calcula_todos_angulos(landmark_data)
print(dic_final)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Resultado com Pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
# visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
# cv2.imshow("Máscara de Segmentação", visualized_mask.astype(np.uint8))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
