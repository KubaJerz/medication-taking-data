import torch
import torch.nn as nn

class BASIC_MLP(nn.Module):
    def __init__(self):
        super(BASIC_MLP, self).__init__()
        self.num_classes = 2
        self.l0 = nn.Linear(1800,1024)
        self.relu0 = nn.ReLU()
        self.l1 = nn.Linear(1024,1)

    def forward(self,X):
        x = self.l0(X)
        x = self.relu0(x)
        x = self.l1(x)
        return x.squeeze() 
