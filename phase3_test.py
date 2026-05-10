import pandas as pd
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
from evaluate import evaluate_model
from data_loader import get_dataloaders, get_base_transform
from dp_utils import apply_dp_noise
from deid_utils import apply_pixelization
from metrics import calculate_mse, calculate_ssim
from PIL import Image

class DPTransform:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        
    def __call__(self, img):
        img_np = np.array(img)
        noisy_np = apply_dp_noise(img_np, self.epsilon)
        return Image.fromarray(noisy_np)

def test_phase3(model, dataset_name='MNIST', device='cuda', subset_size=None):
    epsilons = [0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 5.0]
    results = []
    
    # For metrics, load a batch of test data (no transform for SSIM/MSE calculation)
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=False, download=True)
    elif dataset_name == 'AT&T':
        from data_loader import ATTFacesDataset
        dataset = ATTFacesDataset(train=False, transform=None)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
        
    num_samples = min(100, len(dataset))
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    mse_vals = []
    ssim_vals = []
    
    base_t = get_base_transform()
    
    print(f"Running DP evaluation for {dataset_name}...")
    for eps in epsilons:
        print(f"Evaluating DP noise epsilon={eps}...")
        
        current_mses = []
        current_ssims = []
        for idx in indices:
            img, _ = dataset[idx]
            img_np = np.array(img.convert('RGB'))
            noisy_np = apply_dp_noise(img_np, eps)
            
            current_mses.append(calculate_mse(img_np, noisy_np))
            current_ssims.append(calculate_ssim(img_np, noisy_np))
            
        avg_mse = np.mean(current_mses)
        avg_ssim = np.mean(current_ssims)
        mse_vals.append(avg_mse)
        ssim_vals.append(avg_ssim)
        print(f"  MSE: {avg_mse:.4f}, SSIM: {avg_ssim:.4f}")
        
        custom_transform = transforms.Compose([
            DPTransform(eps),
            base_t
        ])
        _, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=32, subset_size=subset_size, test_transform=custom_transform)
        top1, top5 = evaluate_model(model, test_loader, device)
        print(f"  Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
        
        results.append({
            'Dataset': dataset_name,
            'Epsilon': eps,
            'MSE': avg_mse,
            'SSIM': avg_ssim,
            'Top1_Acc': top1,
            'Top5_Acc': top5
        })
        
    df = pd.DataFrame(results)
    df.to_csv(f'results/dp_defense_results_{dataset_name}.csv', index=False)
    print(f"Saved dp_defense_results_{dataset_name}.csv to results directory.")
    
    # --- Compute NP Baseline ---
    np_mses = []
    np_ssims = []
    for idx in indices:
        img, _ = dataset[idx]
        img_np = np.array(img.convert('RGB'))
        # Using b=8 for NP pixelization baseline
        np_img = apply_pixelization(img_np, 8)
        np_mses.append(calculate_mse(img_np, np_img))
        np_ssims.append(calculate_ssim(img_np, np_img))
        
    np_avg_mse = np.mean(np_mses)
    np_avg_ssim = np.mean(np_ssims)
    
    return {
        'epsilons': epsilons,
        'mse_vals': mse_vals,
        'ssim_vals': ssim_vals,
        'np_avg_mse': np_avg_mse,
        'np_avg_ssim': np_avg_ssim,
        'df': df
    }

def plot_combined_metrics(results_dict):
    """
    results_dict: dict with dataset_name as key, and values are the dict returned by test_phase3.
    Requires exactly two datasets 'AT&T' and 'MNIST'.
    """
    plt.figure(figsize=(10, 8))
    
    dp_color = '#4A773C'  # Greenish
    np_color = '#2A4B7C'  # Blueish
    
    datasets_order = ['AT&T', 'MNIST']
    
    for i, ds_name in enumerate(datasets_order):
        if ds_name not in results_dict:
            continue
            
        data = results_dict[ds_name]
        epsilons = data['epsilons']
        
        # MSE Plot
        ax1 = plt.subplot(2, 2, i + 1)
        ax1.plot(epsilons, data['mse_vals'], marker='^', color=dp_color, label='DP', linestyle='-', markersize=7)
        ax1.plot(epsilons, [data['np_avg_mse']]*len(epsilons), marker='*', color=np_color, label='NP', linestyle='-', markersize=8)
        
        for j, eps in enumerate(epsilons):
            ax1.annotate(str(eps), (epsilons[j], data['mse_vals'][j]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
            
        ax1.set_xscale('log')
        ax1.set_xticks([0.1, 1, 5])
        ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax1.set_xlabel(r'$\epsilon$', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
        ax1.legend(loc='best')
        ax1.set_title(f'({chr(97+i)}) MSE - {ds_name}', fontweight='bold', y=-0.25)
        
        # SSIM Plot
        ax2 = plt.subplot(2, 2, i + 3)
        ax2.plot(epsilons, data['ssim_vals'], marker='^', color=dp_color, label='DP', linestyle='-', markersize=7)
        ax2.plot(epsilons, [data['np_avg_ssim']]*len(epsilons), marker='*', color=np_color, label='NP', linestyle='-', markersize=8)
        
        for j, eps in enumerate(epsilons):
            ax2.annotate(str(eps), (epsilons[j], data['ssim_vals'][j]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
            
        ax2.set_xscale('log')
        ax2.set_xticks([0.1, 1, 5])
        ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.set_xlabel(r'$\epsilon$', fontsize=12, fontweight='bold')
        ax2.set_ylabel('SSIM', fontsize=12, fontweight='bold')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax2.legend(loc='best')
        ax2.set_title(f'({chr(99+i)}) SSIM - {ds_name}', fontweight='bold', y=-0.25)
        
    plt.tight_layout()
    plt.savefig('results/dp_metrics_plot.png', bbox_inches='tight')
    plt.close()
    print("Saved dp_metrics_plot.png to results directory.")

def generate_phase3_samples(dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=False, download=True)
    else:
        from data_loader import ATTFacesDataset
        dataset = ATTFacesDataset(train=False, transform=None)
        
    eps_to_show = [0.1, 1.0, 5.0]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    for i in range(3):
        img, _ = dataset[i]
        img_np = np.array(img.convert('RGB'))
        
        # Original
        axes[i, 0].imshow(img_np)
        if i == 0: axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")
        
        # DP noise
        for j, eps in enumerate(eps_to_show):
            noisy_np = apply_dp_noise(img_np, eps)
            axes[i, j+1].imshow(noisy_np)
            if i == 0: axes[i, j+1].set_title(f"DP Noise $\\epsilon={eps}$")
            axes[i, j+1].axis("off")
            
    plt.tight_layout()
    plt.savefig(f'results/phase3_samples_{dataset_name}.png')
    plt.close()
    print(f"Saved phase3_samples_{dataset_name}.png to results directory.")
