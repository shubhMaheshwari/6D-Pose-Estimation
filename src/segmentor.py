import torch
import torch.nn as nn
# import torchvision.transforms.functional as TF
# import torchvision.transforms as T
from skimage.transform import resize
from utils import *


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet,self).__init__()
        self.down1 = self.UnetDoubleConv(3,64)
        self.down2 = self.UnetDoubleConv(64,128)
        self.down3 = self.UnetDoubleConv(128,256)
        self.down4 = self.UnetDoubleConv(256,512)
        self.down5 = self.UnetDoubleConv(512,1024)

        self.up1 = self.UnetDoubleConv(64,82)
        self.up2 = self.UnetDoubleConv(128,64)
        self.up3 = self.UnetDoubleConv(256,128)
        self.up4 = self.UnetDoubleConv(512,256)
        self.up5 = self.UnetDoubleConv(1024,512)

        self.upConvT2 = self.UnetUpConv(128,64)
        self.upConvT3 = self.UnetUpConv(256,128)
        self.upConvT4 = self.UnetUpConv(512,256)
        self.upConvT5 = self.UnetUpConv(1024,512)

        self.softmax = torch.nn.Softmax(dim=1)

        self.final = nn.Conv2d(82,82,1)


    def UnetDoubleConv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def UnetUpConv(self,in_channels,out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,2,2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        convOut1 = self.down1(x)
        maxPool1 = nn.MaxPool2d(2)(convOut1)
        convOut2 = self.down2(maxPool1)
        maxPool2 = nn.MaxPool2d(2)(convOut2)
        convOut3 = self.down3(maxPool2)

        upPool3 = self.upConvT3(convOut3)
        upPool3Cat = torch.cat([upPool3,convOut2],dim=1)
        incConvOut2 = self.up3(upPool3Cat)
        upPool2 = self.upConvT2(incConvOut2)
        upPool2Cat = torch.cat([upPool2,convOut1],dim=1)
        incConvOut1 = self.up2(upPool2Cat)

        out1 = self.up1(incConvOut1)
        out = self.final(out1)

        return(out)


# summary(model, (3,180,320))


class Segmentor:
    def __init__(self): 
        self.device = torch.device('cuda' if CUDA else 'cpu')
        print(f"Loading segmenttion model to:{self.device}")
        self.model = Unet().to(self.device)
        model_params = torch.load(SEGMENTOR_SAVE_PATH)
        self.model.load_state_dict(model_params)
        

    def predict_mask(self,rgb):
        self.model.eval()
        with torch.no_grad():
            orig_shape = rgb.shape[1:3]
            rgb_small = rgb[:,::2,::2]  
            rgb_small =  rgb_small.permute((0,3,1,2))
            rgb_small = rgb_small.float()
            labels = self.model(rgb_small) 
            dis_label = torch.argmax(labels,dim=1)

            dis_label = dis_label.cpu().data.numpy()
            dis_label = np.array( [  resize(x,orig_shape, order=0, preserve_range=True, anti_aliasing=False) for x in dis_label])
            dis_label = torch.from_numpy(dis_label).to(self.device)
            return dis_label



