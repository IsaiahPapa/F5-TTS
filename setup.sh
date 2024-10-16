# Setup conda environment
conda create -n f5tts python=3.10
conda activate f5tts

# Install ffmpeg
sudo apt-get install ffmpeg

# Install PyTorch and Torchaudio
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Run the API
python inference_api.py