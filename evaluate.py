import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from data_loader import get_dataloaders, get_base_transform
from deid_utils import apply_pixelization, apply_gaussian_blur, apply_fft_lowpass
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class DeidTransform:
    def __init__(self, deid_type, param):
        self.deid_type = deid_type
        self.param = param
        
    def __call__(self, img):
        # 1. 如果輸入是 Tensor (C, H, W)，先轉回 PIL 影像處理
        is_tensor = isinstance(img, torch.Tensor)
        if is_tensor:
            # 假設 Tensor 範圍在 [0, 1]
            img_pil = transforms.ToPILImage()(img)
        else:
            img_pil = img

        # 2. 轉換為 NumPy RGB 陣列供 OpenCV/FFT 使用
        img_np = np.array(img_pil.convert('RGB'))
        
        # 3. 執行去識別化
        if self.deid_type == 'pixel':
            img_np = apply_pixelization(img_np, self.param)
        elif self.deid_type == 'blur':
            img_np = apply_gaussian_blur(img_np, self.param)
        elif self.deid_type == 'fft':
            img_np = apply_fft_lowpass(img_np, self.param)
            
        # 4. 轉回 PIL
        result_pil = Image.fromarray(img_np)
        
        # 5. 如果原本輸入是 Tensor，則必須回傳 Tensor 格式
        if is_tensor:
            return transforms.ToTensor()(result_pil)
        
        return result_pil

# --- Adversarial Attack Mechanism (from Branch 2) ---
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def evaluate_model(model, test_loader, device='cuda', epsilon=0):
    model.eval()
    top1_correct = 0
    #整合Top-5計算
    top5_correct = 0
    #
    total = 0
    
    for inputs, labels in tqdm(test_loader, desc=f"Evaluating eps={epsilon}" if epsilon > 0 else "Evaluating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if epsilon > 0:
            inputs.requires_grad = True
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = inputs.grad.data
            perturbed_data = fgsm_attack(inputs, epsilon, data_grad)
        else:
            perturbed_data = inputs
            
        with torch.no_grad():
            outputs_adv = model(perturbed_data)
            pred = outputs_adv.argmax(dim=1)
            top1_correct += (pred == labels).sum().item()
            
             # Top-5
            _, top5_pred = outputs_adv.topk(min(5, outputs_adv.size(1)), dim=1)
            top5_correct += top5_pred.eq(labels.view(-1, 1)).sum().item()

            total += labels.size(0)
            
    return (top1_correct / total) * 100, (top5_correct / total) * 100

def run_attack_evaluation(model, dataset_name='MNIST', device='cuda', subset_size=None):
    model = model.to(device)
    model.eval()
    results = []
    
    # 1. Traditional Obfuscation Tests (from Branch 1)
    print(f"Evaluating traditional obfuscation for {dataset_name}...")
    _, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=32, subset_size=subset_size)
    
    # Baseline
    acc, acc_top5 = evaluate_model(model, test_loader, device)
    results.append({'Deid_Type': 'None', 'Param': 0, 'Top1_Acc': acc, 'Top5_Acc': acc_top5})
    
    b_values = [4, 8, 16]
    k_values = [15, 45, 99]
    fft_params = [3, 5, 10]
    base_t = get_base_transform()
    
    for b in b_values:
        custom_transform = transforms.Compose([DeidTransform('pixel', b), base_t])
        _, loader = get_dataloaders(dataset_name, 32, subset_size, test_transform=custom_transform)
        acc, acc_top5 = evaluate_model(model, loader, device)
        results.append({'Deid_Type': 'Pixelization', 'Param': b, 'Top1_Acc': acc, 'Top5_Acc': acc_top5})
        
    for k in k_values:
        custom_transform = transforms.Compose([DeidTransform('blur', k), base_t])
        _, loader = get_dataloaders(dataset_name, 32, subset_size, test_transform=custom_transform)
        acc, acc_top5 = evaluate_model(model, loader, device)
        results.append({'Deid_Type': 'Gaussian Blur', 'Param': k, 'Top1_Acc': acc, 'Top5_Acc': acc_top5})

    for r in fft_params:
        custom_transform = transforms.Compose([DeidTransform('fft', r), base_t])
        _, loader = get_dataloaders(dataset_name, 32, subset_size, test_transform=custom_transform)
        acc, acc_top5 = evaluate_model(model, loader, device)
        results.append({'Deid_Type': 'FFT', 'Param': r, 'Top1_Acc': acc, 'Top5_Acc': acc_top5})

    # 2. Adversarial Attack Tests (from Branch 2)
    print(f"Evaluating adversarial attacks for {dataset_name}...")
    epsilons = [0.01, 0.05, 0.1, 0.3]
    _, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=32, subset_size=subset_size)
    
    for eps in epsilons:
        acc, acc_top5 = evaluate_model(model, test_loader, device, epsilon=eps)
        results.append({'Deid_Type': 'Adversarial (FGSM)', 'Param': eps, 'Top1_Acc': acc, 'Top5_Acc': acc_top5})

    df = pd.DataFrame(results)
    if 'Dataset' not in df.columns:
        df.insert(0, 'Dataset', dataset_name)
    os.makedirs('results', exist_ok=True)
    df.to_csv(f'results/combined_attack_results_{dataset_name}.csv', index=False)
    print(f"Saved combined_attack_results_{dataset_name}.csv to results directory.")
    return df

def generate_phase2_samples(model, dataset_name='MNIST', device='cuda'):
    model = model.to(device)
    model.eval()
    
    if dataset_name == 'MNIST':
        from torchvision import datasets
        dataset = datasets.MNIST(root='./data', train=False, download=True)
    else:
        from data_loader import ATTFacesDataset
        dataset = ATTFacesDataset(train=False, transform=None)
        
    base_t = get_base_transform()
    
    # Traditional Samples
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for i in range(3):
        img, true_label = dataset[i]
        img_np = np.array(img.convert('RGB'))
        
        # Original
        axes[i, 0].imshow(img_np)
        if i == 0: axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        # Pixel
        pix_np = apply_pixelization(img_np, 8)
        axes[i, 1].imshow(pix_np)
        if i == 0: axes[i, 1].set_title("Pixel (b=8)")
        axes[i, 1].axis("off")
        
        # Blur
        blur_np = apply_gaussian_blur(img_np, 45)
        axes[i, 2].imshow(blur_np)
        if i == 0: axes[i, 2].set_title("Blur (k=45)")
        axes[i, 2].axis("off")
        
        # FFT
        fft_np = apply_fft_lowpass(img_np, 5)
        axes[i, 3].imshow(fft_np)
        if i == 0: axes[i, 3].set_title("FFT (r=5)")
        axes[i, 3].axis("off")
        
    plt.tight_layout()
    plt.savefig(f'results/phase2_traditional_samples_{dataset_name}.png')
    plt.close()

    # Adversarial Samples
    epsilons = [0.1, 0.3]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for i in range(3):
        img, true_label = dataset[i]
        label_tensor = torch.tensor([true_label]).long().to(device)
        img_tensor = base_t(img).unsqueeze(0).to(device)
        img_tensor.requires_grad = True
        
        output = model(img_tensor)
        loss = F.cross_entropy(output, label_tensor)
        model.zero_grad()
        loss.backward()
        data_grad = img_tensor.grad.data
        
        # Original
        orig_np = img_tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(orig_np)
        if i == 0: axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        # Adv
        for j, eps in enumerate(epsilons):
            perturbed_data = fgsm_attack(img_tensor, eps, data_grad)
            adv_np = perturbed_data.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            axes[i, j+1].imshow(adv_np)
            if i == 0: axes[i, j+1].set_title(f"Adv (eps={eps})")
            axes[i, j+1].axis("off")
            
    plt.tight_layout()
    plt.savefig(f'results/phase2_adversarial_samples_{dataset_name}.png')
    plt.close()
    print(f"Saved samples for {dataset_name} to results directory.")
