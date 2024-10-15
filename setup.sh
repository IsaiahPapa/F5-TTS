# Make sure FFMPEG is installed
conda create -n f5tts python=3.10
conda activate f5tts
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python inference_api.py