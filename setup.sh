!pip install streamlit
!pip install yfinance
!pip install joblib
!pip install TA-Lib
!pip install scikit-learn

# Install ta-lib if it's not already installed
!wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
!tar -xzvf ta-lib-0.4.0-src.tar.gz
import os
os.chdir('ta-lib')
!./configure --prefix=/usr
!make
!make install
os.chdir('../')
!pip install TA-Lib
