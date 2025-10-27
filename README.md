# CUDA_accelerated Real-Time object Detection (python + openCv)

### Author
Developed by Sathvik Chintalacheruvu
M.S. in computer engineering, NJIT

## overview
This project implements a ***real-time video processing pipeline*** using ***CUDA*** (via Numba) and ***openCV***
grayscale --> normalization --> Gaussian blur and passed into a face detector for live face detection.

## Features
1. GPU accelerated with CUDA kernels written in python using 'numba.cuda'
2. Real-time webcam feed processing
3. Live FPS display and CPU VS GPU timing comparison
4. Haar Cascade face detection overlay on blurred output
5. Demonstrates 4X speedup over CPU-based preprocessing

## tech stack
1. Language - Python 3.9+
2. Libraries - openCV, NumPy, Numba
3. Hardware - NVIDIA GPU ( with CUDA toolbox )
4. IDE - Pycharm
5. OS - Windows


