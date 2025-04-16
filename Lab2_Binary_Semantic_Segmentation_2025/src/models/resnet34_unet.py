# Implement your UNet model here
from torch import nn
import torch
# assert False, "Not implemented yet!"

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        shortcut = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu(out)
        return out

class Resnet34_Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Resnet34_Unet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Decoder with skip connections
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._decoder_block(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._decoder_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._decoder_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = self._decoder_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.apply(self._initialize_weights)
        # self.fc = nn.Linear(512, num_classes)
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=2),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # implement the forward pass of the UNet model here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_maxpool = self.maxpool(x)
        x1 = self.layer1(x_maxpool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder with skip connections
        up4 = self.upconv4(x4)
        # print(x4.shape)
        # print(up4.shape)
        # print(x3.shape)
        dec4 = self.dec4(torch.cat([up4, x3], dim=1))
        # print(dec4.shape)
        up3 = self.upconv3(dec4)
        # print(x2.shape)
        # print(up3.shape)
        dec3 = self.dec3(torch.cat([up3, x2], dim=1))
        
        up2 = self.upconv2(dec3)
        # print(x1.shape)
        # print(up2.shape)
        dec2 = self.dec2(torch.cat([up2, x1], dim=1))
        # print(dec2.shape)
        up1 = self.upconv1(dec2)
        # print(x.shape)
        # print(up1.shape)
        dec1 = self.dec1(torch.cat([up1, x], dim=1))
        
        return self.final_conv(dec1)

if __name__ == '__main__':
    model = Resnet34_Unet()
    # print(model)
    x = torch.randn(1, 3, 384, 384)
    print(model(x).shape)