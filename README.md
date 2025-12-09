# install DEVCOMSuitepython
```
# Create the virtual environment
python3.10 -m venv flowsigenv

# Activate the virtual environment
source flowsigenv/bin/activate

# Clone the repository
git clone https://github.com/axelalmet/flowsig.git
cd ./flowsig/

# Install
pip3 install .

git clone https://github.com/guanhaowu123/DEVCOM-Suite-python.git
cd DEVCOM-Suite-python
conda create -n DEVCOMSuitepy -f DEVCOMSuitepy.yaml
conda activate DEVCOMSuitepy

pip install git+https://github.com/guanhaowu123/DEVCOM-Suite-python.git

import devcom_suite
print(devcom_suite.__version__)
```
