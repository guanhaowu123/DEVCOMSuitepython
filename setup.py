from setuptools import setup, find_packages

setup(
    name="DEVCOMSuitepython",         
    version="0.1",
    description="DEVCOM Suite Python utilities for FlowSig + SURD + DEVCOM.",
    author="Guan Haowu",
    url="https://github.com/guanhaowu123/DEVCOMSuitepython",
    license="MIT",               
    #  python/devcomsuite 
    packages=["DEVCOMSuitepython","DEVCOMSuitepython.utils"],
    # install_requires=[
    #     "numpy",
    #     "pandas",
    #     "matplotlib",
    #     "networkx",
    #     "scipy",
    #     "scikit-learn",
    #     "anndata",
    #     "scanpy",
    #     "statsmodels",
    #     "flowsig"
    # ],
    python_requires=">=3.9",
)
