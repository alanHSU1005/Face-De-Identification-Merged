import torch.nn as nn
from torchvision import models

def get_model(num_classes=10):
    """
    Loads pre-trained ResNet18 and modifies the final fully connected layer.
    num_classes: number of output classes (10 for CIFAR-10)
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
