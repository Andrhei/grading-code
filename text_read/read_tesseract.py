import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import unicodedata
import json

class RedacaoOCR:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def preprocess_image(self, image_path, region=None):
        img = cv2.imread(image_path)

        # Redimensionar imagem (dobrar tamanho)
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

        if region:
            x1, y1, x2, y2 = region
            img = img[y1:y2, x1:x2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        return Image.fromarray(dilated)

    def postprocess_text(self, text):
        """Corrige erros comuns de OCR e tenta normalizar o texto."""

        # Substituições comuns
        substitutions = {
            '0': 'o',
            '1': 'l',
            '5': 's',
            '|': 'l',
            '—': '-',
        }

        for wrong, right in substitutions.items():
            text = text.replace(wrong, right)

        # Remover espaços entre letras de palavras quebradas (ex: "e s p e r a n ç a")
        text = re.sub(r'\b(?:\w\s+){2,}\w\b', lambda m: m.group(0).replace(' ', ''), text)

        # Capitalizar frases
        frases = re.split(r'(?<=[.!?])\s+', text)
        frases = [frase.capitalize() for frase in frases]
        text = ' '.join(frases)

        # Corrigir múltiplos espaços
        text = re.sub(r'\s{2,}', ' ', text)

        # Tentar normalizar acentuação perdida (exemplo básico)
        # Aqui só aplicamos se a palavra virar algo reconhecível — pode ser aprimorado com dicionário
        text = unicodedata.normalize("NFKC", text)

        return text.strip()

    def extract_text(self, image_path):
        image = self.preprocess_image(image_path)
        custom_config = '--psm 6 --tessdata-dir "/opt/homebrew/share/"'
        raw = pytesseract.image_to_string(image, lang="por", config=custom_config)
        return self.postprocess_text(raw)

    def extract_text_by_region(self, image_path, regions):
        extracted = {}
        for i, region in enumerate(regions):
            image = self.preprocess_image(image_path, region=region)
            raw = pytesseract.image_to_string(image, lang="por")
            processed = self.postprocess_text(raw)
            extracted[f'regiao_{i+1}'] = processed
        return extracted


ocr = RedacaoOCR()

regioes = [(92, 492, 947, 523)]

image = Image.open("text_read/folha_redacao_preenchida1.png")
for reg in regioes:
    image.crop(reg).show()

resultado = ocr.extract_text_by_region("text_read/folha_redacao_preenchida1.png", regioes)


output_relative_path = "text_read/redacao_text.txt"
for regiao, texto in resultado.items():
    print(f"--- {regiao} ---")
    try:
        if output_relative_path:
            with open(output_relative_path, 'w', encoding='utf-8') as f:
                json.dump(texto, f, ensure_ascii=False, indent=2)
            print(f"Resultado gravado em: {output_relative_path}")
        else:
            print(json.dumps(texto, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f'Erro ao gravar à correção - {e}')
    print(texto)
