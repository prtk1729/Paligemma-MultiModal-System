#!/bin/bash

# Create a new Conda environment
conda create -n env3d python=3.8 -y
conda init bash
source activate env3d

# Install pclpy, PyTorch with CUDA support, and open3d
conda install openblas-devel -c anaconda -y



# Reinstall PyTorch with CUDA support
conda uninstall pytorch torchvision torchaudio pytorch-cuda=11.8 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Create and run test_cuda.py to check CUDA availability
echo "import torch
print(torch.cuda.is_available())" > test_cuda.py
python test_cuda.py
conda install -c davidcaron pclpy -y
pip install open3d


# Check NVCC version and set CUDA_HOME
/usr/local/cuda-11.8/bin/nvcc --version
export CUDA_HOME=/usr/local/cuda-11.8
export CXX=c++

# Clone and install MinkowskiEngine
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --blas=openblas --force_cuda

# Return to the original directory
cd ..

# Install scikit-learn and scipy
conda install scikit-learn scipy -y