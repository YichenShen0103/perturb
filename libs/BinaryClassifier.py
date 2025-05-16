import torch.nn as nn
import torchvision.models as models

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        # Use ResNet50 as requested
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)  # Binary classification
        
    def forward(self, x):
        return self.model(x)