import torch.nn as nn

def DownSample(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels))

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_stride):
        super(BasicBlock, self).__init__()

        if downsample_stride == None:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.downsample = None
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
            self.downsample = DownSample(in_channels, out_channels, downsample_stride)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, inputs):
        identity = inputs

        x = self.bn1(self.conv1(inputs))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))

        if self.downsample != None:
            identity = self.downsample(identity)
        x = self.relu(x + identity)

        return x
    
class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64, None), BasicBlock(64, 64, None))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, (2, 2)), BasicBlock(128, 128, None))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, (2, 2)), BasicBlock(256, 256, None))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, (2, 2)), BasicBlock(512, 512, None))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 2)
        self.fc = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x

class Bottleneck(nn.Module):
    extend = 4
    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels*self.extend, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*self.extend)
        )
        self.relu = nn.ReLU()
        self.downsample = downsample
        
    def forward(self, inputs):

        identity = inputs
        x = self.block(inputs)
        
        if self.downsample != None:
            identity = self.downsample(inputs)

        x += identity
        x = self.relu(x)
        
        return x
        
class ResNet(nn.Module):
    def __init__(self, Block, layer, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self.make_layer(Block, layer[0], base=64)
        self.layer2 = self.make_layer(Block, layer[1], base=128, stride=2)
        self.layer3 = self.make_layer(Block, layer[2], base=256, stride=2)
        self.layer4 = self.make_layer(Block, layer[3], base=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512*Block.extend, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(512*Block.extend, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, num_classes)
        )
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.fc(x.reshape(x.shape[0], -1))
        return x
        
    def make_layer(self, Block, blocks, base, stride=1):
        downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != base*Block.extend:
            downsample = DownSample(self.in_channels, base*Block.extend, stride)
            
        layers.append(Block(self.in_channels, base, downsample=downsample, stride=stride))
        self.in_channels = base*Block.extend
        
        for _ in range(blocks-1):
            layers.append(Block(self.in_channels, base))
            
        return nn.Sequential(*layers)

def ResNet18(num_classes, channels=3):
    return resnet18()
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)