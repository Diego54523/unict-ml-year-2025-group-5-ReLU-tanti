from torch import nn
from Custom_BasicBlock_CNN import BasicBlock
class CustomCNNFeatureExtractor(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNNFeatureExtractor, self).__init__()
        self.in_channels = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True) #Dimensione 16x176x206

        self.layer1 = self.layer_block(BasicBlock, 32, num_blocks=2, stride=1)  #Dimensione 32x176x206
        self.layer2 = self.layer_block(BasicBlock, 64, num_blocks=2, stride=2) #Dimensione 64x88x103
        self.layer3 = self.layer_block(BasicBlock, 128, num_blocks=2, stride=2) #Dimensione 128x44x52
        self.layer4 = self.layer_block(BasicBlock, 256, num_blocks=3, stride=2) #Dimensione 256x22x26 e poi 256x11x13

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #Output sarà di dimensione 256x1x1

        self.dropout = nn.Dropout(p=0.5)

        self.linear = nn.Linear(256, num_classes) #Classificazione in 4 classi
    
    def layer_block(self, block, out_channels, num_blocks, stride):
        if(num_blocks > 2):
            strides = [1] + (num_blocks - 1) * [stride]  # L'ultimo blocco può avere uno stride diverso
        else:
            strides = [1] * (num_blocks - 1) + [stride]
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten

        out = self.dropout(out)
        
        out = self.linear(out)

        return out