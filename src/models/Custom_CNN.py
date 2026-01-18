from torch import nn
from models.Custom_BasicBlock_CNN import BasicBlock

class CustomCNNFeatureExtractor(nn.Module):
    def __init__(self, num_classes = 4):
        super(CustomCNNFeatureExtractor, self).__init__()
        self.in_channels = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size = 7, stride = 2, padding = 3, bias = False) # Input 1x176x208 -> Output 32x88x104
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) # Output 32x44x52

        self.layer1 = self.layer_block(BasicBlock, 32, num_blocks = 2, stride = 1) # Output 32x44x52
        self.layer2 = self.layer_block(BasicBlock, 64, num_blocks = 2, stride = 2) # Output 64x22x26
        self.layer3 = self.layer_block(BasicBlock, 128, num_blocks = 2, stride = 2) # Output 128x11x13

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Output 128x1x1

        self.dropout = nn.Dropout(p = 0.25)
        self.linear = nn.Linear(128, num_classes) # Classificazione in 4 classi
    
    def layer_block(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten

        out = self.dropout(out)
        
        out = self.linear(out)

        return out