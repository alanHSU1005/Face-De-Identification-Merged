import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_mse(img1, img2):
    """
    Calculates Mean Squared Error between two numpy arrays.
    """
    return np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)

def calculate_ssim(img1, img2):
    """
    Calculates SSIM between two numpy arrays.
    """
    # Use win_size smaller than image dimensions if images are small.
    # CIFAR-10 is 32x32, default win_size is 7 or 11.
    win_size = min(7, img1.shape[0], img1.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
        
    return ssim(img1, img2, channel_axis=-1, data_range=255, win_size=win_size)
