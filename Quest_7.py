#Funciona muito bem em imagens bimodais (com dois tons predominantes: objeto e fundo).

#Escolhe o threshold ideal com base na distribuição dos pixels (histograma).

import cv2
import matplotlib.pyplot as plt

def segmentar_otsu(caminho_img, titulo_objeto):
    # Carregar imagem em escala de cinza
    img = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Erro ao carregar a imagem: {caminho_img}")
        return

    # Aplicar Otsu (thresholding automático)
    _, img_segmentada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Exibir a imagem original e a segmentada
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Imagem Original")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Segmentação Otsu - {titulo_objeto}")
    plt.imshow(img_segmentada, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# a) Letras da placa
segmentar_otsu("Imagens/placa-02.jpg", "Letras da Placa")

# b) Barco e pessoas
segmentar_otsu("Imagens/barco.jpg", "Barco e Pessoas")