# Autoencoder MLP with PyTorch and MNIST

This repository contains a simple implementation of an autoencoder with a 4-layer multi-layer perceptron (MLP) using PyTorch. The model is trained on the MNIST dataset for image reconstruction and denoising tasks.

## Usage

### Prerequisites

- Python 3
- PyTorch
- Matplotlib

### Installation
  pip install -r requirements.txt
### Running the Code
**Single Test
To perform a single test, run the following command:
  python your_script.py -l path/to/saved_parameters.pth
  
You will be prompted to enter an image index for testing.

**Test with Noise
To test with noise, run the following command:
  python your_script.py -l path/to/saved_parameters.pth
  
You will be prompted to enter an image index for testing with noise.

**Interpolation Test
To perform an interpolation test, run the following command:
  python your_script.py -l path/to/saved_parameters.pth
  
You will be prompted to enter two image indices for interpolation testing.
