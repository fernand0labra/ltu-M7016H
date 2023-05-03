import torch.nn as nn
import torchvision

class AlexNet(nn.Module):
    def __init__(self, mode):
        super().__init__()

        # AlexNet Model
        self.alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)

        # Alter last linear layer to output on the 3 class pneumonia dataset
        self.alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=3, bias=True)

        if mode == "feature_extraction":  # Freeze all layers except last linear one
            for param in self.alexnet.parameters():
                param.requires_grad = False

            for param in self.alexnet.classifier[6].parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.alexnet(x)