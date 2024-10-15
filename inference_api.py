from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import torchaudio
from transformers import pipeline
import soundfile as sf
import tempfile
from inference_cli import infer, download_audio  # Assuming inference-cli.py functions are in the same directory
import os
import base64

app = FastAPI()

# Load the Whisper model for transcription
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=device)

# Request model for generating audio
class GenerateAudioRequest(BaseModel):
    gen_text: str
    ref_audio_url: str
    ref_text: str = None
    model: str
    remove_silence: bool = False

def audio_to_base64(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

# Route for transcribing audio
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        # Save the uploaded file
        audio_data = await file.read()
        temp_audio.write(audio_data)
        temp_audio.flush()
        
        # Transcribe the audio
        transcription = whisper_model(temp_audio.name)["text"]
    
    return {"transcription": transcription}

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
        print("Transcribing reference audio with Whisper...")
        pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device=device)
        transcription = pipe(ref_audio_path)["text"]
        ref_text = transcription
        print("Transcription completed.")

    # Call the inference function to generate audio
    gen_text = request.gen_text
    model = request.model
    remove_silence = request.remove_silence

    output_file = infer(ref_audio_path, ref_text, gen_text, model, remove_silence)

    # Convert the generated audio to base64
    audio_base64 = audio_to_base64(output_file)

    # Optionally clean up the temporary files
    if os.path.exists(output_file):
        os.remove(output_file)

    # Return the base64-encoded audio in JSON
    return {"generated_audio_base64": audio_base64, "transcribe_audio": ref_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
