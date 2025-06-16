import cv2
import numpy as np
import matplotlib.pyplot as plt

# Parâmetro de realce (quanto maior k, mais nítido)
k = 1.5

# Carregar a imagem em tons de cinza
img = cv2.imread('Imagens/paisagem.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Aplicar suavização com filtro Gaussiano (pode ser media também)
f_barra = cv2.GaussianBlur(img, (5, 5), 0)

# Calcular g_mask = f - f_barra
g_mask = cv2.subtract(img, f_barra)

# Calcular imagem final: g = f + k * g_mask
g_highboost = cv2.addWeighted(img, 1.0, g_mask, k, 0)

# Exibir resultados
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.title("Imagem Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Máscara (Detalhes)")
plt.imshow(g_mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"High Boost (k = {k})")
plt.imshow(g_highboost, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
