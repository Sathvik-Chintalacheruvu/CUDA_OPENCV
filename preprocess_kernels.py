import numpy as np
from numba import cuda, float32

# ---------------- RGB → Grayscale Kernel ---------------- #
@cuda.jit
def rgb_to_gray_kernel(input_img, output_img):
    x, y = cuda.grid(2)
    if x < input_img.shape[0] and y < input_img.shape[1]:
        r = input_img[x, y, 2]
        g = input_img[x, y, 1]
        b = input_img[x, y, 0]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        output_img[x, y] = gray


# ---------------- Normalization Kernel ---------------- #
@cuda.jit
def normalize_kernel(input_img, output_img):
    x, y = cuda.grid(2)
    if x < input_img.shape[0] and y < input_img.shape[1]:
        output_img[x, y] = input_img[x, y] / 255.0


# ---------------- Gaussian Blur Kernel ---------------- #
@cuda.jit
def blur_kernel(input_img, output_img, kernel):
    x, y = cuda.grid(2)
    if 1 <= x < input_img.shape[0] - 1 and 1 <= y < input_img.shape[1] - 1:
        val = 0.0
        weight_sum = 0.0

        for i in range(-1, 2):
            for j in range(-1, 2):
                weight = kernel[i + 1, j + 1]
                val += input_img[x + i, y + j] * weight
                weight_sum += weight

        output_img[x, y] = val / weight_sum
