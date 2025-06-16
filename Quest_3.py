import cv2
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
img = cv2.imread('Imagens/coluna.jpg', cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Calcular a imagem negativa
negativa = 255 - img

# Exibir a original e a negativa
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Negativa")
plt.imshow(negativa, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()