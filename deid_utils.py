import cv2
import numpy as np

def apply_fft_lowpass(image, radius):
    if radius <= 0:
        return image
        
    img_np = np.array(image)
    
    # 統一轉為 (H, W, C) 方便處理
    if len(img_np.shape) == 2:
        h, w = img_np.shape
        img_np = img_np[:, :, np.newaxis]
        is_gray = True
    else:
        h, w, c = img_np.shape
        is_gray = False

    # 建立低通濾波遮罩
    crow, ccol = h // 2, w // 2
    mask = np.zeros((h, w), np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    
    processed_channels = []
    # 逐通道處理 FFT
    for i in range(img_np.shape[2]):
        ch = img_np[:, :, i].astype(np.float32)
        f = np.fft.fft2(ch)
        fshift = np.fft.fftshift(f)
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.abs(np.fft.ifft2(f_ishift))
        processed_channels.append(img_back)
    
    # 合併回影像格式
    result = np.stack(processed_channels, axis=2)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    if is_gray:
        return result[:, :, 0] # 返回 (H, W)
    return result # 返回 (H, W, C)

def apply_pixelization(image, b):
    """
    Applies pixelization to an image using cv2.resize.
    image: numpy array (H, W, C)
    b: block size (integer)
    """
    if b <= 1:
        return image
    h, w = image.shape[:2]
    # Downscale
    small = cv2.resize(image, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_LINEAR)
    # Upscale
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return pixelated

def apply_gaussian_blur(image, k):
    """
    Applies Gaussian blur to an image.
    image: numpy array (H, W, C)
    k: kernel size (integer, must be odd)
    """
    if k <= 1:
        return image
    if k % 2 == 0:
        k += 1 # Ensure k is odd
    blurred = cv2.GaussianBlur(image, (k, k), 0)
    return blurred
