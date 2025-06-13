import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load


def parse_line(line: str) -> np.ndarray:
    """
    Extrai todos os números (ints e floats) de uma linha de texto
    e retorna um array de floats.
    """
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return np.array([float(f) for f in floats], dtype=float)


def load_pose_files(folder: str, label: str):
    """
    Carrega todos os arquivos .txt de uma pasta, extrai features com parse_line
    e atribui o mesmo rótulo (label) a todas as linhas válidas.
    Retorna X (lista de arrays) e y (lista de labels).
    """
    X, y = [], []
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, "r") as f:
            for line in f:
                feats = parse_line(line)
                if feats.size == 33 * 4:
                    X.append(feats)
                    y.append(label)
    return X, y


# carrega uma única vez quando o módulo é importado
clf = load("model.joblib")


def predict_landmarks(features) -> dict:
    """
    Recebe uma sequência de 132 valores (33 landmarks × 4 coordenadas)
    e retorna um dicionário {pose: probabilidade}.
    """
    arr = np.array(features, dtype=float)
    if arr.shape != (33 * 4,):
        raise ValueError(f"Esperado 132 valores, recebeu {arr.shape}")
    proba = clf.predict_proba([arr])[0]
    return dict(zip(clf.classes_, proba.tolist()))
