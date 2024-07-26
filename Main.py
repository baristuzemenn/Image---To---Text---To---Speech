import os
import torch
import torchaudio
import re
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from transformers import VisionEncoderDecoderModel, GPT2Tokenizer, ViTImageProcessor
from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN

# Yerel dosya yollarınız
image_model_dir = '/Users/baristuzemen/Desktop/vit-gpt2'
tts_model_dir = "/Users/baristuzemen/Desktop/pretrained_models/tts-mstacotron2-libritts"
vocoder_model_dir = "/Users/baristuzemen/Desktop/pretrained_models/tts-hifigan-libritts-22050Hz"
image_path = '/Users/baristuzemen/Desktop/test5.jpg'
reference_speech_path = "/Users/baristuzemen/Desktop/pretrained_models/voicesample.wav"

vision_model = VisionEncoderDecoderModel.from_pretrained(image_model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(image_model_dir)
processor = ViTImageProcessor.from_pretrained(image_model_dir)

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
output_ids = vision_model.generate(**inputs, max_length=50)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

def simplify_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

simplified_caption = simplify_text(caption)

ms_tacotron2 = MSTacotron2.from_hparams(source=tts_model_dir, savedir=tts_model_dir)
hifi_gan = HIFIGAN.from_hparams(source=vocoder_model_dir, savedir=vocoder_model_dir)

output_dir = "/Users/baristuzemen/Desktop/"
os.makedirs(output_dir, exist_ok=True)
audio_path = os.path.join(output_dir, 'audio5.wav')

mel_outputs, mel_lengths, alignments = ms_tacotron2.clone_voice(simplified_caption, reference_speech_path)
waveforms = hifi_gan.decode_batch(mel_outputs)
torchaudio.save(audio_path, waveforms[0], 22050)

fig, ax = plt.subplots(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title("Generated Caption: " + simplified_caption, size=15)

def play_audio(event):
    os.system(f"afplay {audio_path}")

ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
button = Button(ax_button, 'Play Audio')
button.on_clicked(play_audio)

plt.show()
