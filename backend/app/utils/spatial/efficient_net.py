import torch
import torch.nn as nn
from torchvision import models

class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, path_to_weights=None, pretrained=True):
        super(EfficientNetFeatureExtractor, self).__init__()

        # load EfficientNet-B4 backbone
        weights = 'DEFAULT' if pretrained else None
        self.backbone = models.efficientnet_b4(weights=weights)

        # load custom weights if provided
        if path_to_weights is not None:
            self.load_weights(path_to_weights)
            
        # feature extraction layers
        self.features = self.backbone.features
        
        # pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        features = self.features(x)
        pooled_features = self.avgpool(features)
        return features, pooled_features
        
    def save_weights(self, path):
        torch.save(self.backbone.state_dict(), path)
        
    def load_weights(self, path):
        state_dict = torch.load(path)
        self.backbone.load_state_dict(state_dict)
        

