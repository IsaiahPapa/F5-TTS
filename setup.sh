#!/bin/bash

# Exit on error
set -e

# Install necessary dependencies
sudo apt-get install -y wget curl git ffmpeg

# Install Miniconda if not already installed
if ! command -v conda &> /dev/null; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    
    # Add conda to PATH for this session
    CONDA_PATH="$HOME/miniconda/bin"
    export PATH="$CONDA_PATH:$PATH"
    
    # Initialize conda in .bashrc
    eval "$($CONDA_PATH/conda shell.bash hook)"
    $CONDA_PATH/conda init
else
    echo "Conda is already installed"
fi

# Ensure conda commands are available
eval "$(conda shell.bash hook)"

# Create and activate conda environment
# Remove existing environment if it exists
conda env remove -n f5tts --yes 2>/dev/null || true
conda create -n f5tts python=3.10 -y
source activate f5tts || conda activate f5tts

# Verify conda environment activation
if [[ "$(conda info --envs | grep '\*' | awk '{print $1}')" != "f5tts" ]]; then
    echo "Failed to activate conda environment"
    exit 1
fi

echo "Installing PyTorch and dependencies..."
# Install PyTorch and Torchaudio
pip install torch==2.3.0+cu118 torchaudio==2.3.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Check if requirements.txt exists before attempting to install
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found"
fi

# Check if inference_api.py exists before attempting to run
if [ -f inference_api.py ]; then
    echo "Starting API..."
    python inference_api.py
else
    echo "Error: inference_api.py not found"
    exit 1
fi