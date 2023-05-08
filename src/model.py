import torch.nn as nn
import torchvision

# ./docs/papers/article-2.pdf
# Multiclass Classification for Detection of COVID-19 Infection in Chest X-Rays Using CNN
class TabukNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Check stride and padding
        self.conv_3_32 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_32_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_64_32 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=3)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.maxpool(self.conv_3_32(x))
        x = self.maxpool(self.conv_32_64(x))
        
        for _ in range(2):
            x = self.maxpool(self.conv_64_32(x))
            x = self.maxpool(self.conv_32_64(x))

        return self.softmax(self.fc(x.view(-1)))


# ./docs/papers/article-3.pdf
# A Deep Transfer Learning Approach to Diagnose Covid-19 using X-ray Images
class ChittagongNet(nn.Module):  # TODO Histogram Equalization ¿?
    def __init__(self, mode):
        super().__init__()

        self.mode = mode

        # Very Deep Convolutional Networks for Large-Scale Image Recognition
        # https://arxiv.org/abs/1409.1556
        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        self.vgg19 = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)

        # MobileNetV2: Inverted Residuals and Linear Bottlenecks
        # https://arxiv.org/abs/1801.04381
        self.mobileNetV2 = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)

        # Replace fully connected layers
            
            # GlobalAveragePooling2D
            # Dense(1024, activation='relu')
            # Dense(1024, activation='relu')
            # Dense(512, activation='relu')
            # Dense(3, activation='softmax')
        
    def forward(self, x):
        return x  # TODO