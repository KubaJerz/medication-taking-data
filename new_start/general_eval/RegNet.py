import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, stride=2, groups=8):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.bn0 = nn.BatchNorm1d(hidden_channels)

        self.conv1 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=stride, padding=1, groups=groups) 
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = nn.Conv1d(hidden_channels, out_channels, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.conv1(x)
        x =  F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(residual)
        x = F.relu(x, inplace=True)
        return x
    
class Stage(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.block0 = Block(in_channels, hidden_channels, out_channels, stride=2)
        self.block1 = Block(out_channels, hidden_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = Stage(32, 32, 128)
        self.stage1 = Stage(128, 64, 256)
        self.stage2 = Stage(256, 128, 512)
        self.stage3 = Stage(512, 256, 1024)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, in_channels, dropout=0.0):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.l0 = nn.Linear(in_channels, 256)
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = (self.gap(x)).squeeze()
        x = F.relu(self.l0(x), inplace=True)
        x = self.dropout(x)
        x = self.out(x)
        return x
    
# putting it all together
class RegNet(nn.Module):
    def __init__(self, in_channels = 6, dropout=0.0):
        super().__init__()
        self.stem = Stem(in_channels)
        self.body = Body()
        self.class_head = ClassificationHead(1024, dropout)
    
    def forward(self, X):
        x = self.stem(X)
        x = self.body(x)
        x = self.class_head(x)
        return x
