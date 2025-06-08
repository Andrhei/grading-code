from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import boto3

class RedacaoOCR:
    def __init__(self, model_name="microsoft/trocr-base-handwritten", aws_region="us-east-1"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.textract = boto3.client("textract", region_name=aws_region)

    def preprocess_image(self, image_path, region=None):
        image = Image.open(image_path).convert("RGB")
        if region:
            x1, y1, x2, y2 = region
            image = image.crop((x1, y1, x2, y2))
        return image

    def extract_with_trocr(self, pil_image):
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()

    def extract_with_textract(self, pil_image):
        import io
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format="PNG")
        image_bytes = byte_io.getvalue()

        response = self.textract.detect_document_text(Document={'Bytes': image_bytes})
        lines = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
        return "\n".join(lines).strip()

    def extract_text(self, image_path):
        image = self.preprocess_image(image_path)
        text = self.extract_with_trocr(image)
        if not text or text.lower().strip() in ["", "0", "00"]:
            text = self.extract_with_textract(image)
        return text

    def extract_text_by_region(self, image_path, regions):
        extracted = {}
        for i, region in enumerate(regions):
            image = self.preprocess_image(image_path, region=region)
            text = self.extract_with_trocr(image)
            if not text or text.lower().strip() in ["", "0", "00"]:
                text = self.extract_with_textract(image)
            extracted[f"regiao_{i+1}"] = text
        return extracted

ocr = RedacaoOCR()

# Região estimada da redação
regioes = [(80, 500, 1000, 1350)]

resultado = ocr.extract_text_by_region("pagina_3_highres.png", regioes)

for regiao, texto in resultado.items():
    print(f"{regiao}:\n{texto}\n")
