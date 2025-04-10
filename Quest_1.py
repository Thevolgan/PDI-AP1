# Etapas do script:

#Carregar duas imagens.
#Calcular a diferença absoluta entre elas.
#Aplicar um limiar (threshold) para segmentar a imagem, resultando em uma imagem binária (preto e branco).

import cv2
import numpy as np

# PARA CARREGAR as duas imagens em escala de cinza
imagem1 = cv2.imread('Imagens/cena1.png', cv2.IMREAD_GRAYSCALE)
imagem2 = cv2.imread('Imagens/cena2.png', cv2.IMREAD_GRAYSCALE)

# Verifica se as imagens foram carregadas corretamente
if imagem1 is None or imagem2 is None:
    print("Erro ao carregar as imagens.")
    exit()

# Calcula a diferença absoluta entre as duas imagens
diferenca = cv2.absdiff(imagem1, imagem2)

# Aplica uma limiarização para binarizar a imagem (valores acima de 30 viram branco, abaixo viram preto)
_, imagem_binaria = cv2.threshold(diferenca, 30, 255, cv2.THRESH_BINARY)

# Exibe as imagens
cv2.imshow('Diferenca Absoluta', diferenca)
cv2.imshow('Imagem Binarizada', imagem_binaria)
cv2.waitKey(0)
#cv2.destroyAllWindows()

# Salva a imagem binarizada
cv2.imwrite('saida_binaria.jpg', imagem_binaria)    