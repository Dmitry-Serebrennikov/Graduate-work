sudo apt update
sudo apt install python3-dev libproj-dev proj-data proj-bin libgeos-dev --yes
sudo apt install libgeos++-dev libgeos-3.8.0 libgeos-c1v5 libgeos-dev libgeos-doc --assume-yes
sudo apt -y install libproj-dev --yes
sudo apt-get install libgeos-dev=3.7

pip3 install pyproj
pip3 install cartopy==0.18.0

pip3 install shapely
pip3 install pyshp
pip3 install aacgmv2

sudo apt-get install -y python3-opencv --yes
pip3 install scikit-image