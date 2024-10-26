import argparse
import codecs
import re
import tempfile
from pathlib import Path
import requests
import numpy as np
import soundfile as sf
import tomli
import torch
import torchaudio
import tqdm
from cached_path import cached_path
from einops import rearrange
from pydub import AudioSegment, silence
from vocos import Vocos
import time

from model import CFM, DiT, MMDiT, UNetT
from model.utils import (convert_char_to_pinyin, get_tokenizer,
                         load_checkpoint, save_spectrogram)

# Global variables
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None
vocos = None
ema_model = None

def setup(config_path="inference-cli.toml", model_type="F5-TTS", load_vocoder_from_local=False):
    global vocos, ema_model
    
    config = tomli.load(open(config_path, "rb"))
    
    vocos_local_path = "../checkpoints/charactr/vocos-mel-24khz"

    if load_vocoder_from_local:
        print(f"Load vocos from local path {vocos_local_path}")
        vocos = Vocos.from_hparams(f"{vocos_local_path}/config.yaml")
        state_dict = torch.load(f"{vocos_local_path}/pytorch_model.bin", map_location=device)
        vocos.load_state_dict(state_dict)
        vocos.eval()
    else:
        print("[Core] Download Vocos from huggingface charactr/vocos-mel-24khz")
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    print(f"[Core] Using Device '{device}'")

    # Load the appropriate model
    if model_type == "F5-TTS":
        F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
        ema_model = load_model(model_type, "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000)
    elif model_type == "E2-TTS":
        E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
        ema_model = load_model(model_type, "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000)

def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = f"ckpts/{exp_name}/model_{ckpt_step}.pt"
    if not Path(ckpt_path).exists():
        ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

def download_audio(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(response.content)
            tmp_audio.flush()
            return tmp_audio.name
    except requests.exceptions.RequestException as e:
        print(f"[Core] Failed to download audio from {url}: {e}")
        return None

def chunk_text(text, max_chars=135):
    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[;:,.!?])\s+|(?<=[；：，。！？])', text)

    for sentence in sentences:
        if len(current_chunk.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode('utf-8')) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def infer_batch(ref_audio, ref_text, gen_text_batches, remove_silence, cross_fade_duration=0.15):
    audio, sr = ref_audio
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    generated_waves = []
    spectrograms = []

    for gen_text in tqdm.tqdm(gen_text_batches):
        if len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        ref_audio_len = audio.shape[-1] // hop_length
        zh_pause_punc = r"。，、；：？！"
        ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
        gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
        duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)

        with torch.inference_mode():
            generated, _ = ema_model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")

        # Measure CPU decoding time
        cpu_start_time = time.perf_counter()
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        cpu_end_time = time.perf_counter()
        cpu_decode_time = cpu_end_time - cpu_start_time
        
        # Measure GPU decoding time
        gpu_start_time = time.perf_counter()
        generated_wave_gpu = vocos.decode(generated_mel_spec.to(device))
        torch.cuda.synchronize()  # Ensure GPU operations are completed
        gpu_end_time = time.perf_counter()
        gpu_decode_time = gpu_end_time - gpu_start_time

        print(f"CPU decoding time: {cpu_decode_time:.4f} seconds")
        print(f"GPU decoding time: {gpu_decode_time:.4f} seconds")


        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        generated_wave = generated_wave.squeeze().cpu().numpy()
        
        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves with cross-fading
    if cross_fade_duration <= 0:
        final_wave = np.concatenate(generated_waves)
    else:
        final_wave = generated_waves[0]
        for i in range(1, len(generated_waves)):
            prev_wave = final_wave
            next_wave = generated_waves[i]

            cross_fade_samples = int(cross_fade_duration * target_sample_rate)
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))

            if cross_fade_samples <= 0:
                final_wave = np.concatenate([prev_wave, next_wave])
                continue

            prev_overlap = prev_wave[-cross_fade_samples:]
            next_overlap = next_wave[:cross_fade_samples]

            fade_out = np.linspace(1, 0, cross_fade_samples)
            fade_in = np.linspace(0, 1, cross_fade_samples)

            cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in

            new_wave = np.concatenate([
                prev_wave[:-cross_fade_samples],
                cross_faded_overlap,
                next_wave[cross_fade_samples:]
            ])

            final_wave = new_wave

    if remove_silence:
        audio_segment = AudioSegment(
            final_wave.tobytes(),
            frame_rate=target_sample_rate,
            sample_width=final_wave.dtype.itemsize,
            channels=1
        )
        non_silent_segs = silence.split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        final_wave = np.array(non_silent_wave.get_array_of_samples()).astype(np.float32) / 32768.0

    combined_spectrogram = np.concatenate(spectrograms, axis=1)
    
    return final_wave, combined_spectrogram

def infer(ref_audio_orig, ref_text, gen_text, remove_silence, cross_fade_duration=0.15):
    print("[Core] Converting audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio_orig)

        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave

        audio_duration = len(aseg)
        if audio_duration > 15000:
            print("[Core] Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    if not ref_text.strip():
        raise ValueError("No reference text provided")
    else:
        print("[Core] Using custom reference text...")

    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    audio, sr = torchaudio.load(ref_audio)
    max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (25 - audio.shape[-1] / sr))
    gen_text_batches = chunk_text(gen_text, max_chars=max_chars)
    print(f'[Core] ref_text: {ref_text}')
    for i, gen_text in enumerate(gen_text_batches):
        print(f'[Core] gen_text {i}: {gen_text}')
    
    print(f"[Core] Generating audio in {len(gen_text_batches)} batches...")
    audio_data, spectrogram_data = infer_batch((audio, sr), ref_text, gen_text_batches, remove_silence, cross_fade_duration)
    return audio_data, spectrogram_data
