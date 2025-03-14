import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BottleneckBlock, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.batch_norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.batch_norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.batch_norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        out = self.max_pool(out)
        
        out += residual
        return out
    
class ECGModel(nn.Module):
    def __init__(self):
        super(ECGModel, self).__init__()
        self.layer1 = self._make_layer(12, 16, 7, 1)
        self.layer2 = self._make_layer(16, 16, 7, 1)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.layer3 = self._make_layer(16, 32, 5, 1)
        self.layer4 = self._make_layer(32, 32, 5, 1)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.layer5 = self._make_layer(32, 64, 3, 1)
        self.layer6 = self._make_layer(64, 64, 3, 1)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(64)
        
        self.conv_final = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=64)
        self.batch_norm_final = nn.BatchNorm1d(128)
        self.relu_final = nn.ReLU()
        self.dropout_final = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, kernel_size, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(BottleneckBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, length)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout2(x)
        
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dropout3(x)
        
        x = self.avg_pool(x)
        
        x = self.conv_final(x)
        x = self.batch_norm_final(x)
        x = self.relu_final(x)
        
        x = x.view(x.size(0), -1)
        
        feature_vector = x

        classification_output = self.fc(feature_vector)  # Classification output
        classification_output = self.sigmoid(classification_output)

        return classification_output, feature_vector

class EGMModel(nn.Module):
    def __init__(self):
        super(EGMModel, self).__init__()
        self.layer1 = self._make_layer(1, 16, 7, 1)
        self.layer2 = self._make_layer(16, 16, 7, 2)
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.layer3 = self._make_layer(16, 32, 5, 1)
        self.layer4 = self._make_layer(32, 32, 5, 2)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.layer5 = self._make_layer(32, 64, 3, 1)
        self.layer6 = self._make_layer(64, 64, 3, 2)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(64)
        
        self.conv_final = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=64)
        self.batch_norm_final = nn.BatchNorm1d(128)
        self.relu_final = nn.ReLU()
        self.dropout_final = nn.Dropout(p=0.5)
        
        self.fc = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, kernel_size, blocks):
        layers = []
        for _ in range(blocks):
            layers.append(BottleneckBlock(in_channels, out_channels, kernel_size))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, channels, length)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.dropout1(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout2(x)
        
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.dropout3(x)
        
        x = self.avg_pool(x)
        
        x = self.conv_final(x)
        x = self.batch_norm_final(x)
        x = self.relu_final(x)
        
        x = x.view(x.size(0), -1)
        
        feature_vector = x

        classification_output = self.fc(feature_vector)  # Classification output
        classification_output = self.sigmoid(classification_output)

        return classification_output, feature_vector