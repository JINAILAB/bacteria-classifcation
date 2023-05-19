from torchvision.models import swin_v2_b
from torchvision.models import Swin_V2_B_Weights
from torchvision.models import efficientnet_v2_s
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models import resnet18
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision.models import regnet_y_16gf
from torchvision.models.regnet import RegNet_Y_16GF_Weights
from torch import nn
from torchvision import models


class EModel(nn.Module):
    def __init__(self, pretrained, classes=19, embeddings=128):
        super().__init__()
        self.LN = nn.LayerNorm([3, 280, 280])
        self.pretrained = pretrained
        self.FC1 = nn.Linear(1000, embeddings)
        self.FC2 = nn.Linear(128, classes)
        
        

    def forward(self, x):
        #x = self.LN(x)
        x = self.pretrained(x)
        x1 = self.FC1(x)
        x2 = self.FC2(x1)
        return x1, x2

class MLP(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(2700, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, classes, bias=True),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.flatten(x)
        Net_Out = self.layer(x)
        return Net_Out
    




# resnet18 = EModel(resnet18(weights = ResNet18_Weights.IMAGENET1K_V1))
# regnet_16gf = EModel(regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_V2))
# effnetv2_s = EModel(efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1))
# mlp = MLP()

