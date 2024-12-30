import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.feature_extractor(torch.zeros(1, *input_shape)).size()[-1]
        self.classifier = nn.Sequential(
            nn.Linear(size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x: torch.ByteTensor):
        xx = x / 255.0 #Normalization
        return self.classifier(self.feature_extractor(xx))