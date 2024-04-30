cd dataset/
mkdir mag
git clone https://github.com/luoxiaojian/xtrapulp.git
cd xtrapulp
make
make libxtrapulp
cd ../
python prepare_mag.py 
cd .. 
