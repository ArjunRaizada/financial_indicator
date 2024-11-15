#!/bin/bash

# Download and extract TA-Lib
echo "Downloading TA-Lib source..."
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz

# Navigate to TA-Lib directory, configure, make, and install
cd ta-lib
echo "Configuring and installing TA-Lib..."
./configure --prefix=/usr
make
sudo make install
cd ..

# Install TA-Lib Python package
echo "Installing TA-Lib Python package..."
pip install TA-Lib

# Install other Python packages
echo "Installing required Python packages..."
pip install streamlit pandas numpy joblib yfinance scikit-learn

echo "Setup completed successfully!"
