#!/bin/bash
# Move to the directory where this script is located
cd "$(dirname "$0")"

echo "Checking environment..."
# Check if a virtual environment exists, if not, create one
if [ ! -d "venv" ]; then
    echo "First time setup: Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install requirements
source venv/bin/activate
echo "Updating libraries (this may take a moment on first run)..."
pip install -r requirements.txt

# Launch the app
echo "Launching the Stats App..."
python app.py