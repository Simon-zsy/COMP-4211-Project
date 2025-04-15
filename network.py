from torch import nn
import torch
from ConvNeXt.models.convnext import convnext_base
from torch.nn import functional as F
# from torchvision.models import convnext_base #牛魔原来有库存

'''
ConvNeXt(
  (downsample_layers): ModuleList(...)  # 下采样层，包括初始卷积和每个阶段的降采样
  (stages): ModuleList(...)            # 特征提取的主要阶段
  (norm): LayerNorm(...)               # 归一化层
  (head): Linear(...)                  # 分类头（MLP）
)
'''

class ConvNeXtBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 加载官方预训练模型
        model = convnext_base(pretrained=pretrained)
        
        # # 分解特征提取层
        self.downsample_layers = model.downsample_layers  # 下采样层
        self.stages = model.stages
        
    def forward(self, x):
        features = []  # 存储中间特征用于跳跃连接
        for i in range(len(self.downsample_layers)): # 两个部分成对执行
            x = self.downsample_layers[i](x)  # 下采样
            features.append(x)  # 保存下采样后的特征
            x = self.stages[i](x)             # 特征提取

        return x, features  # 输出 [B, 1024, H/32, W/32]
    
'''
由于 ConvNeXt已经进行了downsampling, 所以之后直接进行upsampling
'''
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.up(x)  # 上采样
        skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 拼接跳跃连接
        x = self.conv(x)  # 卷积处理
        return x
    
class ConvNeXtUNet(nn.Module):
    def __init__(self, num_points=8, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtBackbone(pretrained=pretrained)
        
        # 解码器，skip_channels 与 features 的通道数对应
        self.up1 = UpBlock(1024, 512, skip_channels=1024)  # features[3]: 1024
        self.up2 = UpBlock(512, 256, skip_channels=512)    # features[2]: 512
        self.up3 = UpBlock(256, 128, skip_channels=256)    # features[1]: 256
        self.up4 = UpBlock(128, 64, skip_channels=128)     # features[0]: 128
        
        # 将空间特征转换为向量
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(64, num_points * 2)  # 输出 8 个点的 (x, y)，共 16 个值

        # 应用 He 初始化（不覆盖预训练权重）
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                # He 初始化权重（针对 ReLU）
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm 初始化为均值 1，方差 0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(self, x):
        x, features = self.encoder(x)  # x: [B, 1024, H/32, W/32]
        # features: [
        #   [B, 128, H/4, W/4],  # features[0]
        #   [B, 256, H/8, W/8],  # features[1]
        #   [B, 512, H/16, W/16],# features[2]
        #   [B, 1024, H/32, W/32]# features[3]
        # ]
        
        x = self.up1(x, features[3])  # [B, 512, H/16, W/16]
        x = self.up2(x, features[2])  # [B, 256, H/8, W/8]
        x = self.up3(x, features[1])  # [B, 128, H/4, W/4]
        x = self.up4(x, features[0])  # [B, 64, H/2, W/2]
        
        # 全局平均池化：[B, 64, H/2, W/2] -> [B, 64, 1, 1]
        x = self.pool(x)
        # 展平并通过全连接层：[B, 64, 1, 1] -> [B, 64] -> [B, 16]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

#test    
if __name__ == "__main__":
    model = ConvNeXtUNet(num_points=8, pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    print(out.shape)  