#!/bin/bash

# Upgrade pip to the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install required Python packages from requirements.txt
echo "Installing required Python packages..."
pip install -r requirements.txt

echo "Setup completed successfully!"
