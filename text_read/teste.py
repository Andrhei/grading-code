from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Cortar manualmente a região e salvar
image = Image.open("text_read/folha_redacao_preenchida1.png").convert("RGB")
crop = image.crop((80, 500, 1000, 1350))
crop.save("recorte_redacao.png")

# OCR direto
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

pixel_values = processor(images=crop, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(text)
