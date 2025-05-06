import sys
import cv2
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)

from funcoes import (
    draw_landmarks_on_video_frame,
    calcula_todos_angulos_video
)

def main():
    # 1) Inicializa o detector em modo VÍDEO
    base = BaseOptions(model_asset_path="../pose_landmarker.task")
    opts = PoseLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.VIDEO,
        output_segmentation_masks=True
    )
    detector = PoseLandmarker.create_from_options(opts)

    # 2) Abre o vídeo
    cap = cv2.VideoCapture('../Images/Treino/dangun.mp4')
    if not cap.isOpened():
        print("Erro ao abrir vídeo/câmera"); sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    ts = 0

    # 3) Processa quadro a quadro
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # BGR→RGB & mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # inferência
        result = detector.detect_for_video(mp_img, ts)

        # desenho de landmarks
        frame = draw_landmarks_on_video_frame(frame, result)

        # cálculo de ângulos
        angulos = calcula_todos_angulos_video(result)
        for pid, dic in angulos.items():
            for eixo, info in dic.items():
                θ = info['angulo']
                num = int(eixo.replace('eixo',''))
                y  = 30 + 20*(num-1) + pid*200
                cv2.putText(
                    frame,
                    f"P{pid}-{eixo}:{int(θ)}°",
                    (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2
                )

        cv2.imshow('Reprodução', frame)
        ts += int(1000/fps)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__=="__main__":
    main()
