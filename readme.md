#1: Install mamba

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

#2 Install sagemath

```
mamba create -n sage106 sage=10.6 python=3.12
```

#3 activate the environment and install dependencies

```
mamba activate sage106
pip install -r requirements.txt
```
