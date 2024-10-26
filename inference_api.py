from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import soundfile as sf
import tempfile
from inference_core import setup, infer, download_audio, target_sample_rate  # Import from inference_core
import os
import base64
from contextlib import asynccontextmanager
import io
import numpy as np
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse, PlainTextResponse

# Load the Whisper model for transcription
is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_ready
    # Initialize the model and vocoder
    setup(config_path="inference-cli.toml", model_type="F5-TTS", load_vocoder_from_local=False)
    # Set the ready flag to True after initialization is complete
    is_ready = True
    yield
    is_ready = False

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Request model for generating audio
class GenerateAudioRequest(BaseModel):
    gen_text: str
    ref_audio_url: str
    ref_text: str = None
    remove_silence: bool = False

@app.get("/health", response_class=PlainTextResponse)
async def health_check():
    if is_ready:
        return "OK"
    else:
        return PlainTextResponse("Not Ready", status_code=503)

# Route for generating audio
@app.post("/generate")
async def generate_audio(request: GenerateAudioRequest):
    print(request)
    # Download the reference audio from the provided URL
    if request.ref_audio_url:
        print(f"Downloading reference audio from URL: {request.ref_audio_url}")
        ref_audio_path = download_audio(request.ref_audio_url)
        if not ref_audio_path:
            return {"error": "Failed to download reference audio from URL"}
    else:
        return {"error": "No reference audio URL provided"}
    
    # Use Whisper to transcribe the reference audio (if no ref_text provided)
    ref_text = request.ref_text
    if not ref_text:
        return {"error": "No reference text provided (API no longer transcribes reference audio.)"}

    # Call the inference function to generate audio
    gen_text = request.gen_text
    remove_silence = request.remove_silence

    audio_data, spectrogram_data = infer(ref_audio_path, ref_text, gen_text, remove_silence)
    if(audio_data is None):
        return {"error": "Failed to generate audio"}
    # Convert audio data to base64
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, target_sample_rate, format='wav')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return audio_base64

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
