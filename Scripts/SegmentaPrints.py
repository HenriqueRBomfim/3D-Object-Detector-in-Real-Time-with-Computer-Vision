import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Configurações do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# Parâmetros de detecção de imobilidade
detection_frames = 8      # número de frames em sequência para considerar imóvel
movement_threshold = 0.005  # limiar de movimento das landmarks

# Função para calcular movimento entre dois conjuntos de landmarks
def pose_movement(landmarks_prev, landmarks_cur):
    if landmarks_prev is None or landmarks_cur is None:
        return np.inf
    pts_prev = np.array([[lm.x, lm.y] for lm in landmarks_prev])
    pts_cur  = np.array([[lm.x, lm.y] for lm in landmarks_cur])
    return np.mean(np.linalg.norm(pts_cur - pts_prev, axis=1))

# Caminhos de arquivos
input_video_path = '../Images/Treino/DanGunSlow.mp4'
output_video_with_counter = 'video_taekwondo_with_counter.mp4'

# Preparar captura e escrita de vídeo com contador se necessário
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Não foi possível abrir o vídeo {input_video_path}")

# Configurar writer somente se o arquivo não existir
generate_video = not os.path.exists(output_video_with_counter)
if generate_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_video_with_counter, fourcc, fps, (width, height))
    print(f"Gerando vídeo com contador: {output_video_with_counter}")
else:
    print(f"Vídeo com contador já existe, pulando geração: {output_video_with_counter}")

# Inicialização de variáveis
frame_idx = 0
prev_landmarks = None
still_counter = 0
movements = []
stationary_frames = []  # índices onde a pose está imóvel

# Loop de processamento
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção de pose
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    cur_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None
    mov = pose_movement(prev_landmarks, cur_landmarks)
    movements.append(mov)

    # Desenhar contador de frames e escrever no vídeo se necessário
    if generate_video:
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)

    # Verifica imobilidade
    if mov < movement_threshold:
        still_counter += 1
    else:
        still_counter = 0

    stop = False
    # Bloco interativo de confirmação de frames estacionários
    if still_counter == detection_frames:
        stationary_frames.append(frame_idx)
        window_name = f"Frame {frame_idx} – S=salvar, N=pular, K=sair"
        cv2.imshow(window_name, frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (ord('s'), ord('S')):
                img_name = f"pose_still_{frame_idx}.png"
                cv2.imwrite(img_name, frame)
                print(f"[S] Imagem salva: {img_name}")
                break
            elif key in (ord('n'), ord('N')):
                print(f"[N] Imagem descartada: frame {frame_idx}")
                break
            elif key in (ord('k'), ord('K')):
                print("Tecla K pressionada. Encerrando o programa.")
                stop = True
                break
            else:
                continue
        cv2.destroyWindow(window_name)
        if stop:
            break

    # Verifica tecla geral K durante reprodução normal
    if cv2.waitKey(1) & 0xFF in (ord('k'), ord('K')):
        print("Tecla K pressionada. Encerrando o programa.")
        break

    prev_landmarks = cur_landmarks
    frame_idx += 1

# Finaliza captura e writer
cap.release()
if generate_video:
    writer.release()
cv2.destroyAllWindows()
print("Processamento de vídeo concluído.")

# Salvar movimentos em arquivo de texto
with open('movements.txt', 'w') as f:
    for idx, m in enumerate(movements):
        f.write(f"Frame {idx}: {m}\n")
print('Arquivo movements.txt gerado com os valores de movimento por frame.')

# Plotar gráfico de movimentos com destaques automáticos
plt.figure()
plt.plot(movements, label='Movimento médio')
for fh in stationary_frames:
    plt.axvline(fh, linestyle='--', linewidth=1)
    plt.scatter(fh, movements[fh], zorder=5)
plt.title('Movimento médio por frame (destaques automáticos)')
plt.xlabel('Índice do frame')
plt.ylabel('Movimento médio')
plt.legend()
plt.grid(True)
plt.savefig('movements_plot.png')
print('Gráfico salvo em movements_plot.png com destaques automáticos.')
