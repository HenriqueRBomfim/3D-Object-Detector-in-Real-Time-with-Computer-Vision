import cv2

img = cv2.imread(".\Images\image.png")
cv2.imshow("Minha Imagem", img)
cv2.waitKey(0)  # Espera até uma tecla ser pressionada
cv2.destroyAllWindows()  # Fecha a janela depois disso
