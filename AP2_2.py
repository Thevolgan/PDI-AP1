import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Cria a máscara passa-altas
# -------------------------------
def criar_mascara_passa_altas(shape, raio=30):
    rows, cols = shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.ones((rows, cols), np.float32)
    y, x = np.ogrid[:rows, :cols]
    centro = (crow, ccol)
    area_baixa = (x - centro[1]) ** 2 + (y - centro[0]) ** 2 <= raio**2
    mask[area_baixa] = 0
    return mask

# -------------------------------
# 2. Aplica a equação do enunciado
# -------------------------------
def aplicar_filtro_alta_frequencia(img_gray, k=2, raio=30):
    # Fourier 2D + shift
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    # Máscara passa-altas
    H_hp = criar_mascara_passa_altas(img_gray.shape, raio)

    # Aplica a equação
    G = (1 + k * H_hp) * fshift

    # Inverso de Fourier
    img_realcada = np.fft.ifft2(np.fft.ifftshift(G))
    img_realcada = np.abs(img_realcada)
    img_realcada = np.clip(img_realcada, 0, 255).astype(np.uint8)

    return img_realcada

# -------------------------------
# 3. Leitura e processamento
# -------------------------------
def processar_imagem(caminho, k=2, raio=30):
    imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        print(f"Erro: Imagem '{caminho}' não encontrada.")
        return None, None
    imagem_filtro = aplicar_filtro_alta_frequencia(imagem, k, raio)
    return imagem, imagem_filtro

# -------------------------------
# 4. Execução principal
# -------------------------------
def main():
    # Parâmetros ajustáveis
    k = 2
    raio = 30

    caminho1 = "Imagens/ultrasound_triplets.jpg"
    caminho2 = "Imagens/radiograph1.jpg"

    original1, realcada1 = processar_imagem(caminho1, k, raio)
    original2, realcada2 = processar_imagem(caminho2, k, raio)

    # Exibe os resultados lado a lado
    plt.figure(figsize=(12, 8))

    # Ultrasound
    plt.subplot(2, 2, 1)
    plt.imshow(original1, cmap='gray')
    plt.title("Ultrasound - Original")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(realcada1, cmap='gray')
    plt.title("Ultrasound - Realçada")
    plt.axis('off')

   # Radiograph
    plt.subplot(2, 2, 3)
    plt.imshow(original2, cmap='gray')
    plt.title("Radiograph - Original")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(realcada2, cmap='gray')
    plt.title("Radiograph - Realçada")
    plt.axis('off')

    plt.suptitle("Realce de Altas Frequências no Domínio da Frequência", fontsize=14)
    plt.tight_layout()
    plt.show()

# Executa
main()
