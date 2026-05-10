import pandas as pd
import os

def generate_report():
    # Load and combine attack results
    attack_dfs = []
    for ds in ['AT&T', 'MNIST']:
        path = f'results/attack_results_{ds}.csv'
        if os.path.exists(path):
            attack_dfs.append(pd.read_csv(path))
    
    if attack_dfs:
        attack_df = pd.concat(attack_dfs, ignore_index=True)
    else:
        attack_df = pd.DataFrame()

    # Load and combine DP results
    dp_dfs = []
    for ds in ['AT&T', 'MNIST']:
        path = f'results/dp_defense_results_{ds}.csv'
        if os.path.exists(path):
            dp_dfs.append(pd.read_csv(path))
            
    if dp_dfs:
        dp_df = pd.concat(dp_dfs, ignore_index=True)
    else:
        dp_df = pd.DataFrame()
        
    report_content = f"""# Face De-Identification, Attacks, and Defenses
## Report Draft

### Team Contributions
- **[Name 1]**: [Role/Contribution]
- **[Name 2]**: [Role/Contribution]

### Phase 2: Attack Results
Baseline and obfuscated test sets evaluation across datasets.

{attack_df.to_markdown(index=False) if not attack_df.empty else "*No data available*"}

### Phase 3: Differential Privacy Defense Results
Evaluating impact of Laplacian noise on metrics and model accuracy across datasets.

{dp_df.to_markdown(index=False) if not dp_df.empty else "*No data available*"}

### Visualizations

#### Combined DP vs NP Metrics Plot (Phase 3)
![DP Metrics](dp_metrics_plot.png)

#### Phase 1: Baseline De-Identification Visualizations
![Phase 1 Samples AT&T](phase1_samples_AT&T.png)
![Phase 1 Samples MNIST](phase1_samples_MNIST.png)

#### Phase 2: Attack Visualizations (Original vs Obfuscated Predictions)
![Phase 2 Samples AT&T](phase2_samples_AT&T.png)
![Phase 2 Samples MNIST](phase2_samples_MNIST.png)

#### Phase 3: DP Noise Visualizations
![Phase 3 Samples AT&T](phase3_samples_AT&T.png)
![Phase 3 Samples MNIST](phase3_samples_MNIST.png)
"""
    with open('results/Report_Draft.md', 'w') as f:
        f.write(report_content)
    
    print("Saved Report_Draft.md to results directory.")

if __name__ == "__main__":
    generate_report()
