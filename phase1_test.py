import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
from deid_utils import apply_pixelization, apply_gaussian_blur, apply_fft_lowpass
from model import get_model

# 定義 FGSM 攻擊機制 (from Branch 2)
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def test_phase1(dataset_name='MNIST'):
    os.makedirs('results', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB')), 
        transforms.ToTensor(),
    ])

    # Load model for adversarial gradient calculation
    num_classes = 10 if dataset_name == 'MNIST' else 40
    model = get_model(num_classes=num_classes).to(device)
    model.eval()

    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=False, download=True)
    elif dataset_name == 'AT&T':
        from data_loader import ATTFacesDataset
        dataset = ATTFacesDataset(train=False, transform=None)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    # 1. Traditional Obfuscation Visualization
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    b_values = [4, 8, 16]
    k_values = [15, 45, 99]
    fft_params = [3, 5, 10] 
    
    for i in range(3):
        img, _ = dataset[i]
        img_np = np.array(img.convert('RGB'))
            
        axes[i, 0].imshow(img_np)
        if i == 0: axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        pix = apply_pixelization(img_np, b_values[i])
        axes[i, 1].imshow(pix)
        if i == 0: axes[i, 1].set_title("Pixelization")
        axes[i, 1].axis("off")
        
        blur1 = apply_gaussian_blur(img_np, k_values[0])
        axes[i, 2].imshow(blur1)
        if i == 0: axes[i, 2].set_title(f"Blur k={k_values[0]}")
        axes[i, 2].axis("off")
        
        blur2 = apply_gaussian_blur(img_np, k_values[1])
        axes[i, 3].imshow(blur2)
        if i == 0: axes[i, 3].set_title(f"Blur k={k_values[1]}")
        axes[i, 3].axis("off")

        fft = apply_fft_lowpass(img_np, fft_params[0])
        axes[i, 4].imshow(fft)
        if i == 0: axes[i, 4].set_title(f"FFT r={fft_params[0]}")
        axes[i, 4].axis("off")
        
    plt.tight_layout()
    plt.savefig(f'results/phase1_traditional_{dataset_name}.png')
    plt.close()

    # 2. Adversarial Attack Visualization (from Branch 2)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    epsilons = [0.02, 0.1, 0.3] 
    
    for i in range(3):
        img_raw, label = dataset[i]
        img_tensor = transform(img_raw).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
        
        output = model(img_tensor)
        loss = F.cross_entropy(output, torch.tensor([label]).to(device))
        model.zero_grad()
        loss.backward()
        data_grad = img_tensor.grad.data

        # Original
        orig_np = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(orig_np)
        if i == 0: axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        # Adv Samples
        for j, eps in enumerate(epsilons):
            adv_tensor = fgsm_attack(img_tensor, eps, data_grad)
            adv_np = adv_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            axes[i, j+1].imshow(adv_np)
            if i == 0: axes[i, j+1].set_title(f"Adv (eps={eps})")
            axes[i, j+1].axis("off")
            
    plt.tight_layout()
    plt.savefig(f'results/phase1_adversarial_{dataset_name}.png')
    plt.close()
    print(f"Saved visualizations for {dataset_name}")

if __name__ == "__main__":
    test_phase1('MNIST')
    test_phase1('AT&T')
