import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthCNN(nn.Module):
    def __init__(self, input_channels=1, feature_dims=[32, 64, 128, 256]):
        """
        Convolutional Neural Network for depth map feature extraction.

        Args:
            input_channels: Number of input channels (1 for depth maps)
            feature_dims: List of feature dimensions for each conv block
        """
        super(DepthCNN, self).__init__()

        self.feature_dims = feature_dims

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, feature_dims[0], kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(feature_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                feature_dims[0], feature_dims[1], kernel_size=5, stride=2, padding=2
            ),
            nn.BatchNorm2d(feature_dims[1]),
            nn.ReLU(inplace=True),
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                feature_dims[1], feature_dims[2], kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(feature_dims[2]),
            nn.ReLU(inplace=True),
        )

        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                feature_dims[2], feature_dims[3], kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(feature_dims[3]),
            nn.ReLU(inplace=True),
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input depth map tensor of shape (B, 1, H, W)

        Returns:
            features: Extracted feature vector of shape (B, feature_dims[-1])
            intermediate: Dictionary of intermediate feature maps
        """
        # Store intermediate features for multi-scale analysis
        intermediate = {}

        x = self.conv1(x)
        intermediate["conv1"] = x

        x = self.conv2(x)
        intermediate["conv2"] = x

        x = self.conv3(x)
        intermediate["conv3"] = x

        x = self.conv4(x)
        intermediate["conv4"] = x

        # Global average pooling to get feature vector
        x = self.gap(x)
        features = x.view(x.size(0), -1)

        return features, intermediate

    def extract_features(self, x):
        """
        Extract only the final feature vector without intermediate maps.

        Args:
            x: Input depth map tensor of shape (B, 1, H, W)

        Returns:
            features: Extracted feature vector of shape (B, feature_dims[-1])
        """
        features, _ = self.forward(x)
        return features


# Example usage
if __name__ == "__main__":
    # Create model
    model = DepthCNN(input_channels=1, feature_dims=[32, 64, 128, 256])

    # Example depth map (batch_size=4, channels=1, height=224, width=224)
    depth_map = torch.randn(4, 1, 224, 224)

    # Extract features
    features, intermediate = model(depth_map)

    print(f"Input shape: {depth_map.shape}")
    print(f"Output features shape: {features.shape}")
    print("\nIntermediate feature map shapes:")
    for name, feat_map in intermediate.items():
        print(f"  {name}: {feat_map.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
