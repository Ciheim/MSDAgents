#!/bin/bash

# Exit on error
set -e

echo "Starting GFDL Chatbot Assistant Setup..."

# 1. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# 2. Activate and Install Dependencies
echo "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip

if [ -f "pyproject.toml" ]; then
    pip install .
else
    if [ ! -f "requirements.txt" ]; then
        echo "ERROR! The requirements.txt file is missing! Aborting setup!"
	    exit 1
    fi
    pip install -r requirements.txt
fi

echo ""
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To start the assistant in a browser, run: python fre-cli/fre_cli_chatbot/frontend.py ui"
echo "To start the assistant in the terminal, run: python fre-cli/fre_cli_chatbot/frontend.py query"
