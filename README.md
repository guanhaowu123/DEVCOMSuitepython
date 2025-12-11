# install DEVCOMSuitepython
```
conda create -n DEVCOMSuitepython python=3.10.18 -y
conda activate DEVCOMSuitepython

conda install -c conda-forge \
    scanpy anndata numpy pandas scipy matplotlib \
    networkx scikit-learn squidpy==1.6.5 umap-learn -y

pip install \
    pyliger==0.2.4 tensorflow_probability==0.25.0 \
    tensorflow==2.19.1 tf_keras==2.19.0 causaldag \
    mpart==2.2.2 pymp==0.0.6 spatial_factorization==0.0.1 \
    "git+https://github.com/willtownes/spatial-factorization-py.git#egg=spatial-factorization"

git clone https://github.com/axelalmet/flowsig.git
cd ./flowsig
python -m pip install --no-deps .

cd ~
git clone https://github.com/guanhaowu123/DEVCOMSuitepython.git
cd DEVCOMSuitepython
pip install .

```
