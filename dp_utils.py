import numpy as np

def apply_dp_noise(image, epsilon):
    """
    Adds Laplacian noise to an image based on epsilon for Differential Privacy.
    image: numpy array (H, W, C) or (H, W)
    epsilon: privacy budget (float)
    """
    # Sensitivity of an image channel pixel is 255.
    # Laplace scale b = Sensitivity / epsilon
    b = 255.0 / epsilon
    
    # Generate Laplacian noise
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Generate grayscale noise (H, W, 1) and broadcast to all 3 channels
        noise = np.random.laplace(0, b, (image.shape[0], image.shape[1], 1))
    else:
        noise = np.random.laplace(0, b, image.shape)
    
    # Add noise and clip
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image
