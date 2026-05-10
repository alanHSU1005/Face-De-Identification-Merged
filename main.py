import torch
import os
from data_loader import get_dataloaders
from model import get_model
from train import train_model
from evaluate import run_attack_evaluation, generate_phase2_samples
from phase1_test import test_phase1
from phase3_test import test_phase3, plot_combined_metrics, generate_phase3_samples
from generate_report import generate_report

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    os.makedirs('results', exist_ok=True)
    
    datasets_to_run = ['AT&T', 'MNIST']
    phase3_results_dict = {}
    
    for ds_name in datasets_to_run:
        print(f"\n==================================================")
        print(f"=== Starting Pipeline for Dataset: {ds_name} ===")
        print(f"==================================================")
        
        subset_size = 1000 if ds_name == 'MNIST' else None
        num_classes = 40 if ds_name == 'AT&T' else 10
        
        print(f"\n--- Phase 1: De-Identification Visualizations ({ds_name}) ---")
        # Now includes both traditional and adversarial visualizations
        test_phase1(dataset_name=ds_name)
        
        print(f"\n--- Phase 0 & 2: Setup and Baseline Model Training ({ds_name}) ---")
        train_loader, _ = get_dataloaders(dataset_name=ds_name, batch_size=32, subset_size=subset_size)
        model = get_model(num_classes=num_classes)
        # Train for 3 epochs for rapid testing
        model = train_model(model, train_loader, dataset_name=ds_name, epochs=3, device=device)
        
        print(f"\n--- Phase 2: Evaluating Attack Model (Traditional & Adversarial) ({ds_name}) ---")
        # Now evaluates both traditional obfuscation and adversarial attacks
        run_attack_evaluation(model, dataset_name=ds_name, device=device, subset_size=subset_size)
        generate_phase2_samples(model, dataset_name=ds_name, device=device)
        
        print(f"\n--- Phase 3: DP Defense & Metrics ({ds_name}) ---")
        res = test_phase3(model, dataset_name=ds_name, device=device, subset_size=subset_size)
        phase3_results_dict[ds_name] = res
        generate_phase3_samples(dataset_name=ds_name)

    print("\n--- Phase 3: Plotting Combined Metrics ---")
    plot_combined_metrics(phase3_results_dict)
    
    print("\n--- Phase 4: Report Generation ---")
    generate_report()
    
    print("\nPipeline execution complete! Check the results directory.")

if __name__ == "__main__":
    main()
