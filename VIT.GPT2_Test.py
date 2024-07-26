import torch
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt

model_dir = '/Users/baristuzemen/Desktop/vit-gpt2'
image_path = '/Users/baristuzemen/Desktop/test6.jpg'  

model = VisionEncoderDecoderModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
processor = ViTImageProcessor.from_pretrained(model_dir)

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

output_ids = model.generate(**inputs, max_length=50)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Görüntüyü ve metni görselleştirme
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')  # Eksenleri kapat
plt.title("Generated Caption: " + caption, size=15)  
plt.show()



