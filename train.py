import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, dataset_name='MNIST', epochs=3, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    os.makedirs('results', exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss/total if total > 0 else 0., 
                                      'acc': 100.*correct/total})
            
        print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f} Acc: {100.*correct/total:.2f}%")
        
    torch.save(model.state_dict(), f'results/baseline_model_{dataset_name}.pth')
    print(f"Saved baseline_model_{dataset_name}.pth to results directory.")
    return model
