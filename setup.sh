#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display messages
echo_info() {
    echo -e "\e[32m[INFO]\e[0m $1"
}

echo_error() {
    echo -e "\e[31m[ERROR]\e[0m $1" >&2
}

# 1. Check for Python 3.10.15, install if not present
PYTHON_VERSION="3.10.15"
PYTHON_EXEC="python3.10"

echo_info "Checking for Python $PYTHON_VERSION..."

if ! command -v $PYTHON_EXEC &> /dev/null
then
    echo_info "Python $PYTHON_VERSION not found. Installing..."

    # Install prerequisites for building Python
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    tk-dev libffi-dev wget

    # Download Python source
    cd /usr/src
    sudo wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz

    # Extract, compile, and install
    sudo tar xzf Python-$PYTHON_VERSION.tgz
    cd Python-$PYTHON_VERSION
    sudo ./configure --enable-optimizations
    sudo make altinstall

    # Clean up
    cd ..
    sudo rm -f Python-$PYTHON_VERSION.tgz
    sudo rm -rf Python-$PYTHON_VERSION

    echo_info "Python $PYTHON_VERSION installed successfully."
else
    echo_info "Python $PYTHON_VERSION is already installed."
fi

# Verify Python version
INSTALLED_PYTHON_VERSION=$($PYTHON_EXEC --version 2>&1)
echo_info "Installed Python version: $INSTALLED_PYTHON_VERSION"

# 2. Install system-level dependencies
echo_info "Installing system-level dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential wget

# 3. Download, compile, and install TA-Lib
echo_info "Installing TA-Lib..."

TA_LIB_VERSION="0.4.0"
TA_LIB_TAR="ta-lib-$TA_LIB_VERSION-src.tar.gz"
TA_LIB_DIR="ta-lib"

# Download TA-Lib source if not already downloaded
if [ ! -f "$TA_LIB_TAR" ]; then
    echo_info "Downloading TA-Lib..."
    wget http://prdownloads.sourceforge.net/ta-lib/$TA_LIB_TAR
else
    echo_info "TA-Lib source tarball already exists. Skipping download."
fi

# Extract TA-Lib
if [ ! -d "$TA_LIB_DIR" ]; then
    echo_info "Extracting TA-Lib..."
    tar -xzvf $TA_LIB_TAR
else
    echo_info "TA-Lib directory already exists. Skipping extraction."
fi

# Compile and install TA-Lib
cd $TA_LIB_DIR
echo_info "Configuring TA-Lib..."
./configure --prefix=/usr

echo_info "Building TA-Lib..."
make

echo_info "Installing TA-Lib..."
sudo make install

cd ..

# Clean up TA-Lib source files
echo_info "Cleaning up TA-Lib files..."
rm -rf $TA_LIB_DIR $TA_LIB_TAR

echo_info "TA-Lib installed successfully."

# 4. Set up Python virtual environment
VENV_DIR="venv"

echo_info "Setting up Python virtual environment..."

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_EXEC -m venv $VENV_DIR
    echo_info "Virtual environment created at ./$VENV_DIR."
else
    echo_info "Virtual environment already exists at ./$VENV_DIR."
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo_info "Upgrading pip..."
pip install --upgrade pip

# 5. Install Python dependencies
echo_info "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo_info "Setup completed successfully."

echo_info "To activate the virtual environment, run: source $VENV_DIR/bin/activate"
