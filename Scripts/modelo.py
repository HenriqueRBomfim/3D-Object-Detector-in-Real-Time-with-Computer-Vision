import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
from joblib import dump


def parse_line(line):
    # Extrai todos os números float da linha
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    return np.array([float(f) for f in floats])


def load_pose_files(folder, label):
    X, y = [], []
    print("Procurando arquivos em:", os.path.abspath(folder))
    for file in glob.glob(os.path.join(folder, "*.txt")):
        print("Lendo arquivo:", file)
        with open(file, "r") as f:
            for line in f:
                features = parse_line(
                    line.replace("),(", ")|(")
                    .replace("),", ")|")
                    .replace(",(", "|(")
                    .replace(" ", "")
                    .replace("),|(", ")|(")
                    .replace("),|", ")|")
                    .replace("|", "")
                )
                if len(features) == 33 * 4:
                    X.append(features)
                    y.append(label)
    return X, y


# Exemplo de uso:
# Estrutura de pastas:
# ./poses/Pose1/ -> arquivos .txt da pose 1
# ./poses/Pose2/ -> arquivos .txt da pose 2
# Adicione mais pastas para mais poses

X, y = [], []
saidas_folder = "./Images/Saidas"

print("Procurando arquivos em:", os.path.abspath(saidas_folder))
for file in glob.glob(os.path.join(saidas_folder, "*.txt")):
    print("Lendo arquivo:", file)
    # Extrai o nome da pose do nome do arquivo (antes do primeiro "_")
    base = os.path.basename(file)
    label = base.split("_")[0]  # Exemplo: Pose1_001.txt -> Pose1
    with open(file, "r") as f:
        for line in f:
            features = parse_line(line)
            if len(features) == 33 * 4:
                X.append(features)
                y.append(label)

# Treinamento do modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# dump(clf, "model.joblib")
# print("Modelo treinado e salvo em model.joblib")

# Avaliação
print(classification_report(y_test, clf.predict(X_test)))

# Matriz de confusão
cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
print("Matriz de confusão:")
print(cm)

# Visualização gráfica (opcional)
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


def predict_pose(input_line):
    features = parse_line(
        input_line.replace("),(", ")|(")
        .replace("),", ")|")
        .replace(",(", "|(")
        .replace(" ", "")
        .replace("),|(", ")|(")
        .replace("),|", ")|")
        .replace("|", "")
    )
    if len(features) != 33 * 4:
        raise ValueError("Linha de entrada inválida")
    proba = clf.predict_proba([features])[0]
    for i, pose in enumerate(clf.classes_):
        print(f"{pose}: {proba[i]*100:.2f}% de chance")


# Exemplo de uso:
# Percorre cada linha do resultado.txt e faz a predição
with open("resultado.txt", "r") as f:
    for idx, line in enumerate(f):
        print(f"Linha {idx+1}:")
        try:
            predict_pose(line.strip())
        except Exception as e:
            print(f"Erro na linha {idx+1}: {e}")
        print("-" * 30)
