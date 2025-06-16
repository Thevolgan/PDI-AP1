import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem
imagem = cv2.imread('Imagens/hubble1.jpg')

# Converte para RGB (OpenCV lê como BGR)
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# Converte para HSV para facilitar o isolamento da cor vermelha
imagem_hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

# Máscaras para a cor vermelha (dois intervalos no HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(imagem_hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(imagem_hsv, lower_red2, upper_red2)
mascara_vermelha = cv2.bitwise_or(mask1, mask2)

# Operações morfológicas para melhorar a máscara
kernel = np.ones((5, 5), np.uint8)
mascara_vermelha = cv2.dilate(mascara_vermelha, kernel, iterations=1)
mascara_vermelha = cv2.erode(mascara_vermelha, kernel, iterations=1)

# Contornos das regiões destacadas
contornos, _ = cv2.findContours(mascara_vermelha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Cria uma máscara em branco e desenha os contornos
mascara_destacada = np.zeros(imagem.shape[:2], dtype=np.uint8)
cv2.drawContours(mascara_destacada, contornos, -1, 255, -1)

# Converte imagem original para tons de cinza
imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

# Aplica a máscara
resultado = cv2.bitwise_and(imagem_cinza, imagem_cinza, mask=mascara_destacada)

# Mostra o resultado
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Regiões destacadas")
plt.imshow(mascara_destacada, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Imagem com apenas os pontos destacados")
plt.imshow(resultado, cmap='gray')
plt.show()
