import cv2
import matplotlib.pyplot as plt

# Carregar a imagem em escala de cinza
img = cv2.imread("Imagens/placa-02.jpg", cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Aplicar Otsu para segmentar automaticamente
_, img_segmentada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Exibir a imagem original e a segmentada
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem Original - Placa")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentação - Letras com Otsu")
plt.imshow(img_segmentada, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
