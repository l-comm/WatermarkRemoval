import torch
import torch.nn as nn

class BrightnessToAlphaBeta(nn.Module):
    def __init__(self):
        super(BrightnessToAlphaBeta, self).__init__()
        # A single linear layer that maps brightness [F,1] directly to [F,2]
        self.fc = nn.Linear(1, 2)

        # Initialize weights and biases so that at brightness=0, alpha=beta=0.5
        with torch.no_grad():
            self.fc.weight.zero_()    # W = 0
            self.fc.bias.fill_(0.5)   # b = [0.5, 0.5]

    def forward(self, brightness):
        # brightness: [F,1]
        # Output: [F,2] for alpha and beta
        return self.fc(brightness)
