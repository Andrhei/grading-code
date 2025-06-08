from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps
import torch

class RedacaoOCR:
    def __init__(self, model_name="microsoft/trocr-base-handwritten"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def preprocess_image(self, image_path, region=None):
        image = Image.open(image_path).convert("L")  # escala de cinza
        if region:
            x1, y1, x2, y2 = region
            image = image.crop((x1, y1, x2, y2))
        image = ImageOps.invert(image)  # inverter se fundo for escuro
        image = ImageOps.autocontrast(image)  # aumenta o contraste automaticamente
        return image.convert("RGB")

    def extract_text(self, image_path):
        image = self.preprocess_image(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def extract_text_by_region(self, image_path, regions):
        extracted = {}
        for i, region in enumerate(regions):
            image = self.preprocess_image(image_path, region=region)
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
            generated_ids = self.model.generate(pixel_values)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            extracted[f"regiao_{i+1}"] = text.strip()
        return extracted


ocr = RedacaoOCR("microsoft/trocr-large-handwritten")

# Região da redação completa
regioes = [(92, 492, 947, 523)]
# regioes = [(0, 0, 1024, 1536)]

image = Image.open("text_read/folha_redacao_preenchida1.png")
for reg in regioes:
    image.crop(reg).show()


resultado = ocr.extract_text_by_region("text_read/folha_redacao_preenchida1.png", regioes)

for regiao, texto in resultado.items():
    print(f"{regiao}:\n{texto}\n")
