import torch 
import torch.nn as nn
import numpy as np
import sys, os, time
import segmentation_models_pytorch as smp

class UNetBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.Unet(
            encoder_name="resnet34",       # good starting point; try "resnet50", "convnext_tiny", "efficientnet-b3", etc.
            encoder_weights="imagenet",    # <-- THIS loads pretrained encoder weights
            in_channels=3,                 # your three velocity-bin brightness maps
            classes=2,                     # 2 output channels (u, v)
            activation=None                # regression: keep raw logits
        )
    def forward(self, x):
        return self.net(x)

class UNetDropout(nn.Module):
    def __init__(self,in_channels=3,out_channels=2,p=0.2):
        super().__init__()
        self.net = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels, classes=out_channels,
            activation=None
        )
        self.p = p
        self.drop = nn.Dropout2d(p)

        # Light-touch: add dropout after each decoder block output and at the bridge
        for i, block in enumerate(self.net.decoder.blocks):
            block.conv2.add_module("mc_dropout_after_conv2", nn.Dropout2d(p))
        # Also add one at the deepest features (center)
        #self.net.decoder.center.add_module("mc_center_dropout", nn.Dropout2d(p))

    def forward(self, x):
        return self.net(x)

class UNetMultiHeadProfiles(nn.Module):
    """
    maps:        (B, Cmap, H, W)  e.g. per-pixel u,v (and optionally logvar)
    mass_prof:   (B, K)
    flow_prof:   (B, L)
    """
    def __init__(self, in_channels=5, out_channels=3,
                 K=37, L=37, p=0.2,
                 encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None
        )
        
        # optional MC dropout in decoder
        for block in self.net.decoder.blocks:
            block.conv2.add_module("mc_dropout_after_conv2", nn.Dropout2d(p))

        enc_out_ch = self.net.encoder.out_channels[-1]
        self.mass_head = nn.Linear(enc_out_ch, K)
        self.flow_head = nn.Linear(enc_out_ch, L)

        def head_vec(out_dim):
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(p),
                nn.Linear(enc_out_ch, 256), nn.ReLU(inplace=True),
                nn.Dropout(p),
                nn.Linear(256, 256), nn.ReLU(inplace=True),
                nn.Dropout(p),
                nn.Linear(256, out_dim)    # raw (z-scored) outputs
            )

        self.mass_head = head_vec(K)
        self.flow_head = head_vec(L)

    def forward(self, x):
        #maps = self.net(x)
        feats = self.net.encoder(x)          # list of feature maps
        deep  = feats[-1]                    # (B, C_enc, H_enc, W_enc)
        #maps  = self.net.segmentation_head(dec)
        deep  = feats[-1]
        dec   = self.net.decoder(feats)
        maps  = self.net.segmentation_head(dec)
        Mvec  = self.mass_head(deep)   # (B,K)
        Fvec  = self.flow_head(deep)   # (B,L)
        return {"maps": maps, "mass_prof": Mvec, "flow_prof": Fvec}




class DoubleConv(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)


#####################################################################################
#####################################################################################
class model_UNetSmall(nn.Module):
    def __init__(self, base, dr, in_ch, out_ch):
        super(model_hp3_err, self).__init__()
        
        self.down1 = DoubleConv(in_ch, base)
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down5 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16))

        self.mid   = DoubleConv(base*16, base*32)

        self.up5   = nn.ConvTranspose2d(base*32, base*16, 2, stride=2)
        self.conv5 = DoubleConv(base*32, base*16)
        self.up4   = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.conv4 = DoubleConv(base*16, base*8)
        self.up3   = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.conv3 = DoubleConv(base*8, base*4)
        self.up2   = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.conv2 = DoubleConv(base*4, base*2)
        self.out   = nn.Conv2d(base*2, out_ch, 1)
        

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        d1 = self.down1(image)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d5)
        m  = self.mid(d5)
        u5 = self.up5(m); u5 = self.conv5(torch.cat([u5, d5], dim=1))
        u4 = self.up4(u5); u4 = self.conv4(torch.cat([u4, d4], dim=1))
        u3 = self.up3(u4); u3 = self.conv3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3); u2 = self.conv2(torch.cat([u2, d2], dim=1))
        y  = self.out(u2)

        return y
####################################################################################
####################################################################################

class model_hp3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_hp3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='zeros', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='zeros', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='zeros', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='zeros', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='zeros', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='zeros', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='zeros', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='zeros', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='zeros', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=2, stride=1, padding=0,
                            padding_mode='zeros', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  12)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        # enforce the errors to be positive
        y = torch.clone(x)
        y[:,6:12] = torch.square(x[:,6:12])

        return y
####################################################################################
####################################################################################
