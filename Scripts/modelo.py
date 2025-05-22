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
                features = parse_line(line.replace('),(', ')|(').replace('),', ')|').replace(',(', '|(').replace(' ', '').replace('),|(', ')|(').replace('),|', ')|').replace('|', ''))
                if len(features) == 33*4:
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
            if len(features) == 33*4:
                X.append(features)
                y.append(label)

# Treinamento do modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
print(classification_report(y_test, clf.predict(X_test)))

# Matriz de confusão
cm = confusion_matrix(y_test, clf.predict(X_test), labels=clf.classes_)
print("Matriz de confusão:")
print(cm)

# Visualização gráfica (opcional)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

def predict_pose(input_line):
    features = parse_line(input_line.replace('),(', ')|(').replace('),', ')|').replace(',(', '|(').replace(' ', '').replace('),|(', ')|(').replace('),|', ')|').replace('|', ''))
    if len(features) != 33*4:
        raise ValueError("Linha de entrada inválida")
    proba = clf.predict_proba([features])[0]
    for i, pose in enumerate(clf.classes_):
        print(f"{pose}: {proba[i]*100:.2f}% de chance")

# Exemplo de uso:
input_line = "(0.5585532188415527, 0.17267394065856934, -0.7137294411659241, 0.9995809197425842),(0.5531966686248779, 0.1540546417236328, -0.667835533618927, 0.9990164041519165),(0.555539608001709, 0.1533566415309906, -0.6686184406280518, 0.9991170763969421),(0.5577192306518555, 0.15286636352539062, -0.6686463952064514, 0.9994208812713623),(0.5385573506355286, 0.15215688943862915, -0.7230825424194336, 0.9994926452636719),(0.530597448348999, 0.15016216039657593, -0.7237758040428162, 0.999574601650238),(0.5210666060447693, 0.14832144975662231, -0.7241455316543579, 0.999629020690918),(0.5149073600769043, 0.15716898441314697, -0.4055883288383484, 0.9994884729385376),(0.4735812544822693, 0.14800745248794556, -0.652755856513977, 0.9990517497062683),(0.5421549677848816, 0.19354042410850525, -0.6059331893920898, 0.9996657371520996),(0.5246965885162354, 0.1903780698776245, -0.6826900243759155, 0.9997068047523499),(0.5662856698036194, 0.27548840641975403, -0.17608042061328888, 0.9999251365661621),(0.2780623137950897, 0.25896769762039185, -0.6178491115570068, 0.9994693398475647),(0.6987813115119934, 0.36791282892227173, -0.3095148205757141, 0.798475980758667),(0.30993136763572693, 0.40489184856414795, -0.9483827352523804, 0.9015385508537292),(0.8836042881011963, 0.3272243142127991, -0.8167695999145508, 0.7359421849250793),(0.5250511765480042, 0.3498830795288086, -1.3054553270339966, 0.6631137132644653),(0.9568316340446472, 0.3096345067024231, -0.9028756618499756, 0.7255997657775879),(0.5723265409469604, 0.33699074387550354, -1.4402363300323486, 0.5797786116600037),(0.9581940174102783, 0.2978627681732178, -0.9671192169189453, 0.7506601810455322),(0.5712047815322876, 0.3231363594532013, -1.4344205856323242, 0.5777595043182373),(0.9236832857131958, 0.30598926544189453, -0.8569507002830505, 0.6932233572006226),(0.5591161847114563, 0.3307359218597412, -1.3158292770385742, 0.41491061449050903),(0.435094952583313, 0.5216220021247864, 0.10662302374839783, 0.9998236298561096),(0.25489768385887146, 0.5072826147079468, -0.10667479783296585, 0.9996628761291504),(0.5853893756866455, 0.6526968479156494, -0.30961304903030396, 0.9221196174621582),(0.13831187784671783, 0.6548526883125305, -0.8158664703369141, 0.9545859098434448),(0.6601827144622803, 0.8453329801559448, 0.09535114467144012, 0.9247651696205139),(0.08751314878463745, 0.8098468780517578, -0.20942291617393494, 0.8809317946434021),(0.6277896165847778, 0.8829469680786133, 0.11168236285448074, 0.8027611970901489),(0.09872588515281677, 0.8424293994903564, -0.16179855167865753, 0.5636754035949707),(0.7722713947296143, 0.9024455547332764, -0.21684826910495758, 0.9097921848297119),(0.01452520489692688, 0.8629066944122314, -0.4562439024448395, 0.8158169388771057)"
predict_pose(input_line)