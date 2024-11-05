import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 normalization: str = 'batch', dropout_rate: float = 0.0):
        super().__init__()
        self.dropout = dropout_rate

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Normalization layers
        if normalization == 'batch':
            self.norm1 = nn.BatchNorm1d(out_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        elif normalization == 'layer':
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        self.activation = nn.ReLU(inplace=True)
        if dropout_rate > 0:
            self.dropout_layer = nn.Dropout1d(dropout_rate)
        else:
            self.dropout_layer = nn.Identity()

        # Shortcut connection if dimensions change
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)]
            if normalization == 'batch':
                shortcut_layers.append(nn.BatchNorm1d(out_channels))
            elif normalization == 'layer':
                shortcut_layers.append(nn.LayerNorm(out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout_layer(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout_layer(out)

        out += identity
        out = self.activation(out)

        return out

class RESMODEL(nn.Module):
    def __init__(self, input_channels, output_size, hyperparams, num_classes):
        super().__init__()
        self.hidden_blocks = hyperparams['hidden_blocks']
        self.layer_zero_depth = hyperparams['layer_zero_depth']
        self.layer_scaler = hyperparams['layer_scaler']
        self.window_size = hyperparams['window_size']
        self.num_classes = num_classes

        
        # Build network architecture
        layers = []
        in_channels = input_channels
        
        for block in range(self.hidden_blocks):
            num_layer_pairs = self.layer_zero_depth + block * self.layer_scaler
            
            for pair in range(num_layer_pairs):
                out_channels = in_channels * 2 if pair == 0 else in_channels
                stride = 2 if pair == 0 else 1
                
                layers.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        normalization=hyperparams['normalization'],
                        dropout_rate=hyperparams['dropout_rate']
                    )
                )
                in_channels = out_channels
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Calculate output dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, self.window_size)
            dummy_output = self.feature_extractor(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.size(0)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x