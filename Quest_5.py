import cv2
import matplotlib.pyplot as plt

# Carregar imagem com ruído
img_ruido = cv2.imread('Imagens/barbara02.png', cv2.IMREAD_GRAYSCALE)

# Verifica se a imagem foi carregada corretamente
if img_ruido is None:
    print("Erro ao carregar a imagem.")
    exit()

# Aplicar filtro da mediana (kernel 3x3 é o mais comum)
img_filtrada = cv2.medianBlur(img_ruido, 3) #kernel

# Exibir as imagens lado a lado
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Imagem com Ruído (Salt and Pepper)")
plt.imshow(img_ruido, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Imagem Filtrada (Mediana)")
plt.imshow(img_filtrada, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()