import cv2
from draw_landmarks_on_image import draw_landmarks_on_image
import numpy as np
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

img = cv2.imread(".\Images\image.png")
# cv2.imshow("Minha Imagem", img)
cv2.waitKey(0)  # Espera até uma tecla ser pressionada
cv2.destroyAllWindows()  # Fecha a janela depois disso

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2.imshow("Resultado com Pose", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
cv2.imshow("Máscara de Segmentação", visualized_mask.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()