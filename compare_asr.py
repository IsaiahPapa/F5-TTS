import torch
from transformers import pipeline
import gc

def print_memory_usage():
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
    elif torch.cuda.is_available():
        print(f"CUDA Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  Cached:    {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    else:
        print("Running on CPU")

def load_and_check_memory(model_name):
    print(f"\nLoading {model_name}")
    gc.collect()
    print_memory_usage()
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = pipeline("automatic-speech-recognition", model=model_name, torch_dtype=torch.float32, device=device)
    
    print(f"After loading {model_name}")
    print_memory_usage()
    
    del model
    gc.collect()

# Check Whisper v2
load_and_check_memory("openai/whisper-large-v2")

# Check Whisper v3
load_and_check_memory("openai/whisper-large-v3")