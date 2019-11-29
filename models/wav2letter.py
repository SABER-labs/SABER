from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASRModel(nn.Module):
    def __init__(self, input_features=80, num_classes=128):
        super(ASRModel, self).__init__()

        # Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layers = nn.Sequential(
            nn.Conv1d(input_features, 250, 48, 3),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 250, 7),
            torch.nn.ReLU(),
            nn.Conv1d(250, 2000, 32),
            torch.nn.ReLU(),
            nn.Conv1d(2000, 2000, 1),
            torch.nn.ReLU(),
            nn.Conv1d(2000, num_classes, 1),
        )

    def forward(self, batch):
        y_pred = self.layers(batch)
        return y_pred

if __name__ == '__main__':
	print("ASRModel Summary")
	from torchsummary import summary
	from time import time
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	input_features = 80
	net = ASRModel(input_features=input_features).to(device)
	summary(net, (input_features, 400), batch_size=1, device=device)
	image = torch.randn(1, input_features, 400).to(device)
	start = time()
	y = net(image)
	print(y.shape)
	print(f"time taken is {time()-start:.3f}s")