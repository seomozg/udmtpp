#!/bin/bash

# Deployment script for UdmTPP RAG system
# This script creates a virtual environment and installs dependencies

# Exit on any error
set -e

echo "Starting deployment..."

# Check if python3-venv is available
if ! python3 -m venv --help > /dev/null 2>&1; then
    echo "Installing python3-venv..."
    apt update

    # Try to install python3-venv, if fails try python3.12-venv (for specific Python versions)
    if ! apt install -y python3-venv; then
        echo "Trying python3.12-venv..."
        apt install -y python3.12-venv
    fi

    # Also install python3-full as backup
    apt install -y python3-full
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# Set environment variables if needed
# export DEEPSEEK_API_KEY=your_api_key_here

# Run the application
echo "Starting the application..."
python -m uvicorn src.app:app --host 0.0.0.0 --port 8001 --reload

# Deactivate virtual environment (won't reach here if running in foreground)
# deactivate
