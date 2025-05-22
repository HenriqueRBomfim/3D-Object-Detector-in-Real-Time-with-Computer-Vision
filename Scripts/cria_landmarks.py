from funcoes import analise_imagem

result = analise_imagem(".\pose_still_116.png", labels=True, path=True)[2]

with open("resultado.txt", "a", encoding="utf-8") as f:
    f.write(str(result)[1:-1] + "\n")
