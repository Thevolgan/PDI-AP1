import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar as imagens em escala de cinza
img1 = cv2.imread('Imagens/imagem-01.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('Imagens/imagem-02.tif', cv2.IMREAD_GRAYSCALE)

# Verifica se as imagens foram carregadas corretamente
if img1 is None or img2 is None:
    print("Erro ao carregar as imagens.")
    exit()

# Calcular a diferença absoluta entre as imagens
diff = cv2.absdiff(img1, img2)

# Aplicar limiar para destacar diferenças (em branco)
_, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Plotar as imagens lado a lado
plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.title("Imagem 01")
plt.imshow(img1, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Imagem 02")
plt.imshow(img2, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Diferença")
plt.imshow(diff, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Diferenças destacadas")
plt.imshow(diff_bin, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()