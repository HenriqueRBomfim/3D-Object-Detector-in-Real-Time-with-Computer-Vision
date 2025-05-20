# Pré-requisitos:

Criar um ambiente virtual:

```
py -m venv venv
```

Ativar o ambiente virtual:

```
.\venv\Scripts\Activate.ps1
```

Adicionar um .gitignore com o seguinte conteúdo:
venv

Instalar no ambiente virtual:

```
pip install mediapipe
```

```
curl -o pose_landmarker.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Versão final do .gitignore:

- venv
- .venv
- pose_landmarker.task
- Scripts\__pycache__


