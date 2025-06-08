import pytesseract
from PIL import Image
import cv2
import numpy as np
import re
import unicodedata
import easyocr
import json

class RedacaoOCR:
    def __init__(self, tesseract_cmd=None, confidence_threshold=15):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.reader = easyocr.Reader(['pt'], gpu=False)
        self.confidence_threshold = confidence_threshold  # Usado para decidir quando cair para EasyOCR

    def preprocess_image(self, image_path, region=None):
        img = cv2.imread(image_path)

        if region:
            x1, y1, x2, y2 = region
            img = img[y1:y2, x1:x2]

        # Redimensiona para melhorar OCR
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

        gray = self.remove_horizontal_lines(img)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        return Image.fromarray(dilated), dilated  # Retorna PIL e OpenCV para EasyOCR também

    def remove_horizontal_lines(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)

        # Detectar linhas horizontais
        horizontal = bin_img.copy()
        cols = horizontal.shape[1]
        horizontal_size = cols // 30

        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        detected_lines = cv2.erode(horizontal, horizontal_structure)
        detected_lines = cv2.dilate(detected_lines, horizontal_structure)

        # Subtrair as linhas da imagem
        no_lines = cv2.bitwise_and(gray, gray, mask=~detected_lines)
        return no_lines

    
    def postprocess_text(self, text):
        substitutions = {
            '0': 'o', '1': 'l', '5': 's', '|': 'l', '—': '-',
        }
        for wrong, right in substitutions.items():
            text = text.replace(wrong, right)
        text = re.sub(r'\b(?:\w\s+){2,}\w\b', lambda m: m.group(0).replace(' ', ''), text)
        frases = re.split(r'(?<=[.!?])\s+', text)
        frases = [frase.capitalize() for frase in frases]
        text = ' '.join(frases)
        text = re.sub(r'\s{2,}', ' ', text)
        text = unicodedata.normalize("NFKC", text)
        return text.strip()

    def extract_with_tesseract(self, pil_image):
        custom_config = '--psm 6'
        raw = pytesseract.image_to_string(pil_image, lang="por", config=custom_config)
        return raw.strip()

    def extract_with_easyocr(self, cv_image):
        results = self.reader.readtext(cv_image, detail=0)
        return " ".join(results).strip()

    def extract_text(self, image_path):
        pil_img, cv_img = self.preprocess_image(image_path)
        text = self.extract_with_tesseract(pil_img)

        if len(text) < self.confidence_threshold:
            text = self.extract_with_easyocr(cv_img)

        return self.postprocess_text(text)

    def extract_text_by_region(self, image_path, regions):
        extracted = {}
        for i, region in enumerate(regions):
            pil_img, cv_img = self.preprocess_image(image_path, region=region)
            text = self.extract_with_tesseract(pil_img)

            if len(text) < self.confidence_threshold:
                text = self.extract_with_easyocr(cv_img)

            processed = self.postprocess_text(text)
            extracted[f'regiao_{i+1}'] = processed
        return extracted


ocr = RedacaoOCR(confidence_threshold=20)

regioes = [(0, 492, 947, 523)]

image = Image.open("text_read/folha_redacao_preenchida2.png")
for reg in regioes:
    image.crop(reg).show()

resultado = ocr.extract_text_by_region("text_read/folha_redacao_preenchida2.png", regioes)

output_relative_path = "text_read/redacao_text.txt"

for regiao, texto in resultado.items():
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

    print(f"{regiao}: {texto}")
