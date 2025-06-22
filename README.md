This project implements a comprehensive framework for predicting voltage profiles in the IEEE 33-bus distribution system using two advanced machine learning approaches: Gaussian Processes (GP) with multiple kernels and Neural Networks (NN) with various architectures. The framework includes automated data sampling, model optimization, GPU acceleration, and detailed results analysis.

Author: Naman Meena

----------------------------------------------------------------------------------------------------------------

This project provides a complete solution for power flow prediction in distribution systems, combining the probabilistic modeling capabilities of Gaussian Processes with the deep learning power of Neural Networks. The implementation focuses on the IEEE 33-bus system and includes comprehensive performance evaluation across multiple model architectures.

Key Features:

`Dual Approach` : Implements both GP and NN methodologies for comparative analysis
`Multi-Kernel GP` : Supports RBF, Polynomial, and Linear kernels with hyperparameter optimization
`Diverse NN Architectures` : Four different neural network configurations (2/3 layers, ReLU/Sigmoid activations)
`GPU Acceleration` : Full CUDA support for both PyTorch (GP) and TensorFlow (NN)
`Automated Optimization` : Uses Optuna for GP hyperparameter tuning
`Comprehensive Evaluation` : Generates detailed per-bus and summary performance metrics
`Export Capabilitie`s : Saves results as CSV files for further analysis

File Structure:

`CFPF_multikernel_GP.py` : for GP model training, optimization, and evaluation ->Gaussian Process implementation	
`CFPF_poly_NN.py` : for NN model training and testing ->Neural Network implementation	
`case33bw.m	` : IEEE 33-bus system data	


Hardware Requirements:

`CPU`: Multi-core processor (recommended for parallel processing)
`RAM`: Minimum 8GB, recommended 16GB or higher
`GPU`: NVIDIA GPU with CUDA capability (optional but highly recommended)
`Storage`: At least 2GB free space for dependencies and results

Software Requirements:

`Python`: Version 3.8 or higher
`pip`: Python package manager
`CUDA Toolkit`: Version compatible with your GPU (for GPU acceleration)
`cuDNN`: NVIDIA Deep Neural Network library (for TensorFlow GPU support)

------------------------------------------------------------------------------------------------------------------

Installation

# Create new virtual environment
python -m venv power_flow_env

# Activate environment
# On Windows:
power_flow_env\Scripts\activate
# On macOS/Linux:
source power_flow_env/bin/activate

Install Core Dependencies:

# Core scientific computing libraries
pip install numpy pandas scikit-learn

# Power system analysis
pip install pandapower

# Parallel processing
pip install joblib

# Machine learning frameworks
pip install torch gpytorch tensorflow keras

# Hyperparameter optimization
pip install optuna


-- GPU Setup Guide-- ( if available otherwise CPU will do your work)

Download and Install GPU Drivers

Download CUDA Toolkit
-->Visit [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
-->Choose version compatible with your PyTorch/TensorFlow installation ( mine were cuda 11.8 , gpytorch (2.10) it supports gpu)

> Add CUDA to System PATH
# Windows (add to environment variables):
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\libnvvp

# Linux/macOS (add to ~/.bashrc or ~/.zshrc):
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


Install cuDNN
-->Download and Install cuDNN
-->Visit [NVIDIA cuDNN](https://developer.nvidia.com/cuda-toolkit)
-->Download version matching your CUDA installation  (IMP)
-->Extract and copy files to CUDA installation directory

>Verify GPU Setup
Test PyTorch GPU Support:

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

>Test TensorFlow GPU Support
import tensorflow as tf
print(f"GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
print(f"GPU names: {[device.name for device in tf.config.list_physical_devices('GPU')]}")

-----------------------------------------------------------------------------------------------------------------

Running the Code:

python CFPF_multikernel_GP.py (run this python script)
python CFPF_poly_NN.py (run this script)



>> You will get output results save in your directory and then run plots python scripts for plots
>> The variables are self explanatory when the code is viewed (explained there)
