import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, Upsample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1= DownSample(in_channels,64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = Upsample(1024, 512)
        self.up_convolution_2 = Upsample(512, 256)
        self.up_convolution_3 = Upsample(256, 128)
        self.up_convolution_4 = Upsample(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)


    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        bottle = self.bottle_neck(p4)
        up_1 = self.up_convolution_1(bottle, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out=self.out(up_4)
        return out
    
# if __name__ == "__main__":
#     double_conv=  DoubleConv(256,256)
#     print(double_conv)

#     input_image = torch.rand((1,3,512,512))
#     model = UNet(in_channels=3, num_classes=10)
#     output = model(input_image)
#     print(output.shape)