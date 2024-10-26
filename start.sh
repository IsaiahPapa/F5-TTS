# Check if inference_api.py exists before attempting to run
if [ -f inference_api.py ]; then
    echo "Starting API..."
    python inference_api.py
else
    echo "Error: inference_api.py not found"
    exit 1
fi