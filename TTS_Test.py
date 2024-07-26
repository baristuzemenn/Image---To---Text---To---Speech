import os
import torch
import torchaudio
from speechbrain.inference.TTS import MSTacotron2
from speechbrain.inference.vocoders import HIFIGAN

def TTS(input_text: str) -> str:
    ms_tacotron2 = MSTacotron2.from_hparams(
        source="/Users/baristuzemen/Desktop/pretrained_models/tts-mstacotron2-libritts", 
        savedir="pretrained_models/tts-mstacotron2-libritts"
    )
    hifi_gan = HIFIGAN.from_hparams(
        source="/Users/baristuzemen/Desktop/pretrained_models/tts-hifigan-libritts-22050Hz", 
        savedir="pretrained_models/tts-hifigan-libritts-22050Hz"
    )

    output_dir = "/Users/baristuzemen/Desktop/"
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, 'TTS-output.wav')

    reference_speech = "/Users/baristuzemen/Desktop/pretrained_models/voicesample.wav"
    mel_outputs, mel_lengths, alignments = ms_tacotron2.clone_voice(input_text, reference_speech)
    waveforms = hifi_gan.decode_batch(mel_outputs)
    torchaudio.save(audio_path, waveforms[0], 22050)

    return audio_path

text = "Thank you to everyone who listened to this presentation. I hope you will like it"

output_file = TTS(text)
print(f"Conversion successful! Audio saved to {output_file}")



