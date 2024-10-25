#!/bin/bash

# Get current username
CURRENT_USER=$(whoami)

# Detect conda installation
if [ -d "$HOME/miniconda" ]; then
    CONDA_PATH="$HOME/miniconda"
elif [ -d "$HOME/miniconda3" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -d "$HOME/anaconda3" ]; then
    CONDA_PATH="$HOME/anaconda3"
else
    echo "Error: Could not find Conda installation"
    exit 1
fi

echo "Found Conda installation at: $CONDA_PATH"

# Create service file content
cat > /tmp/inference-server.service << EOF
[Unit]
Description=Inference Server Service
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
Group=$CURRENT_USER

# Set working directory
WorkingDirectory=/home/$CURRENT_USER/F5-TTS

# Environment setup and command execution
ExecStart=/bin/bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh && \
                       conda activate f5tts && \
                       python inference_api.py'

# Restart on failure
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install the service
echo "Installing service for user: $CURRENT_USER"
sudo mv /tmp/inference-server.service /etc/systemd/system/inference-server.service

# Set proper permissions
sudo chown root:root /etc/systemd/system/inference-server.service
sudo chmod 644 /etc/systemd/system/inference-server.service

# Reload systemd and enable service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable inference-server

echo "Service installed successfully!"
echo "To start the service, run: sudo systemctl start inference-server"
echo "To check status, run: sudo systemctl status inference-server"
echo "To view logs, run: sudo journalctl -u inference-server"

# Optionally start the service
read -p "Would you like to start the service now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    sudo systemctl start inference-server
    echo "Service started! Checking status..."
    sleep 2
    sudo systemctl status inference-server
fi