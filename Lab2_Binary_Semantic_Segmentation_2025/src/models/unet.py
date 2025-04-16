import torch
import torch.nn as nn
import torch.nn.functional as F


def crop(enc_feat, dec_feat):
    _, _, H, W = dec_feat.shape
    _, _, H_enc, W_enc = enc_feat.shape

    delta_H = (H_enc - H) // 2
    delta_W = (W_enc - W) // 2

    # 中心裁剪
    return enc_feat[:, :, delta_H:delta_H + H, delta_W:delta_W + W]
'''
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU()
        )
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        bottleneck = self.bottleneck(x4)

        up4 = self.upconv4(bottleneck)
        x5 = torch.cat((up4, x4), dim=1)
        dec4 = self.dec4(x5)

        up3 = self.upconv3(dec4)
        x6 = torch.cat((up3, x3), dim=1)
        dec3 = self.dec3(x6)

        up2 = self.upconv2(dec3)
        x7 = torch.cat((up2, x2), dim=1)
        dec2 = self.dec2(x7)

        up1 = self.upconv1(dec2)
        x8 = torch.cat((up1, x1), dim=1)
        dec1 = self.dec1(x8)
        
        return dec1'
'''
class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(Unet, self).__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU()
        )
        self.encoder3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
            nn.Conv2d(256, 256, 3), nn.ReLU()
        )
        self.encoder4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3), nn.ReLU(),
            nn.Conv2d(512, 512, 3), nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3), nn.ReLU(),
            nn.Conv2d(1024, 1024, 3), nn.ReLU()
        )
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(1024, 512, 3), nn.ReLU(),
            nn.Conv2d(512, 512, 3), nn.ReLU()
        )
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3), nn.ReLU(),
            nn.Conv2d(256, 256, 3), nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3), nn.ReLU(),
            nn.Conv2d(128, 128, 3), nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
            nn.Conv2d(64, out_channels, 1)
        )
        self.upsample_to_192 = nn.Upsample(size=(192, 192), mode='bilinear', align_corners=False)
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # implement the forward pass of the UNet model here
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        bottleneck = self.bottleneck(x4)

        up4 = self.upconv4(bottleneck)
        x5 = torch.cat((up4, crop(x4, up4)), dim=1)
        dec4 = self.dec4(x5)

        up3 = self.upconv3(dec4)
        x6 = torch.cat((up3, crop(x3, up3)), dim=1)
        dec3 = self.dec3(x6)

        up2 = self.upconv2(dec3)
        x7 = torch.cat((up2, crop(x2, up2)), dim=1)
        dec2 = self.dec2(x7)
        up1 = self.upconv1(dec2)
        x8 = torch.cat((up1, crop(x1, up1)), dim=1)
        dec1 = self.dec1(x8)
        return self.upsample_to_192(dec1)