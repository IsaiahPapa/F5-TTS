#!/bin/bash

# Update package list and upgrade existing packages
sudo apt-get update && sudo apt-get upgrade -y

# Install necessary dependencies
sudo apt-get install -y wget curl git ffmpeg

# Install Miniconda (if not already installed)
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
fi

# Setup conda environment
conda create -n f5tts python=3.10 -y
conda activate f5tts

# Install PyTorch and Torchaudio
pip install torch==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Run the API
python inference_api.py