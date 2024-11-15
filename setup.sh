sudo apt-get update
sudo apt-get install -y build-essential libxml2-dev libxslt1-dev
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzvf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
pip install TA-Lib
