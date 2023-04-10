from torch import nn



# 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__() # 28*28 输入为一张图片，输出为检测网络对应的编号
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5), # in_channels, out_channels, kernel_size 24*24
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride 12*12
            nn.Conv2d(6, 16, 5), # 8*8
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2) # 4*4
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 3)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output