import os
import glob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


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


# === Montagem do dataset ===
X, y = [], []

# Exemplo 1: se você tiver subpastas por pose:
# for pose_name in os.listdir("poses"):
#     folder = os.path.join("poses", pose_name)
#     if os.path.isdir(folder):
#         Xi, yi = load_pose_files(folder, pose_name)
#         X.extend(Xi)
#         y.extend(yi)

# Exemplo 2: leitura direta de .txt em Images/Saidas, extraindo label do nome do arquivo
saidas_folder = "./Images/Saidas"
for path in glob.glob(os.path.join(saidas_folder, "*.txt")):
    label = os.path.basename(path).split("_")[0]
    with open(path, "r") as f:
        for line in f:
            feats = parse_line(line)
            if feats.size == 33 * 4:
                X.append(feats)
                y.append(label)

X = np.array(X)
y = np.array(y)

# === Treinamento ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


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


# === Execução direta para avaliação ===
if __name__ == "__main__":
    print("=== Classification Report ===")
    print(classification_report(y_test, clf.predict(X_test)))
    cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
    print("=== Confusion Matrix ===")
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=clf.classes_,
        yticklabels=clf.classes_,
    )
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()
