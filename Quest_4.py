import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carrega a imagem colorida (em BGR por padrão no OpenCV)
img_color = cv2.imread('Imagens/paisagem.jpg')

# Verifica se a imagem foi carregada corretamente
if img_color is None:
    print("Erro ao carregar a imagem.")
    exit()

# Converte de BGR para RGB (pra usar a fórmula corretamente)
img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# Separa os canais
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# Aplica a fórmula de conversão para tons de cinza
Y1 = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

# Exibe a imagem em tons de cinza
plt.figure(figsize=(6, 6))
plt.title("Imagem em Tons de Cinza (Y1)")
plt.imshow(Y1, cmap='gray')
plt.axis('off')
plt.show()