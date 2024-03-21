# -*- coding: utf-8 -*-
import torch.nn as nn
import torch


class UNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=1):
        super(UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    Conv3x3(num_feat[3], num_feat[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0],
                                             num_classes,
                                             kernel_size=1),
                                   nn.Sigmoid())  # 改为sigmoid

    def forward(self, inputs, return_features=False):
        # print("inputs", inputs)
        down1_feat = self.down1(inputs)
        # print("down1_feat", down1_feat.size())
        down2_feat = self.down2(down1_feat)
        # print("down2_feat", down2_feat.size())
        down3_feat = self.down3(down2_feat)
        # print("down3_feat", down3_feat.size())
        down4_feat = self.down4(down3_feat)
        # print("down4_feat", down4_feat.size())
        bottom_feat = self.bottom(down4_feat)

        # print("bottom_feat", bottom_feat.size())
        up1_feat = self.up1(bottom_feat, down4_feat)
        # print("up1_feat_1", up1_feat.size())
        up1_feat = self.upconv1(up1_feat)
        # print("up1_feat_2", up1_feat.size())
        up2_feat = self.up2(up1_feat, down3_feat)
        # print("up2_feat_1", up2_feat.size())
        up2_feat = self.upconv2(up2_feat)
        # print("up2_feat_2", up2_feat.size())
        up3_feat = self.up3(up2_feat, down2_feat)
        # print("up3_feat_1", up3_feat.size())
        up3_feat = self.upconv3(up3_feat)
        # print("up3_feat_2", up3_feat.size())
        up4_feat = self.up4(up3_feat, down1_feat)
        # print("up4_feat_1", up4_feat.size())
        up4_feat = self.upconv4(up4_feat)
        # print("up4_feat_2", up4_feat.size())

        if return_features:
            outputs = up4_feat
        else:
            outputs = self.final(up4_feat)
            # print(outputs.size())
        # print("outputs: ", outputs)

        return outputs


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        # TODO: Upsampling required after deconv?
        # outputs = self.up(inputs)
        outputs = self.deconv(inputs)
        out = torch.cat([down_outputs, outputs], 1)
        return out

