import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

model = None


def load_model():
    global model
    if model is None:
        if torch.cuda.is_available():
            model = pretrained.master64().cuda()
        else:
            model = pretrained.master64().cpu()


def denoise(input, output):
    wav, sr = torchaudio.load(input)
    if torch.cuda.is_available():
        wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    else:
        wav = convert_audio(wav.cpu(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0]
        torchaudio.save(output, denoised[None][0].cpu(), model.sample_rate)

load_model()
denoise("dataset/alex_noisy.mp3", "output.wav")
