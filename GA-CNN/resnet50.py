import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 参数设置
batch = 64 #批量大小
lr = 0.001 #学习率
epoch = 65 #训练的轮数

# 模型构建
class block(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
            nn.Conv2d(
            self.in_channels,
            intermediate_channels * 4,
            kernel_size=1,
            stride=stride,
            bias=False
            ),
            nn.BatchNorm2d(intermediate_channels * 4),
        )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )
        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channel=3, num_classes=10):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=10):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=10):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)

model = ResNet50(img_channel=3, num_classes=100)
# class Model(nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=6,kernel_size=(1,1))
#         self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(1,1))
#         self.conv3 = nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(1,1))
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(in_features=120*4*4,out_features=120)
#         self.fc2 = nn.Linear(in_features=120,out_features=84)
#         self.fc3 = nn.Linear(in_features=84, out_features=10)
#
#     def forward(self,x):
#         x = F.avg_pool2d(F.relu(self.conv1(x)),2)
#         x = F.avg_pool2d(F.relu(self.conv2(x)),2)
#         x = F.avg_pool2d(F.relu(self.conv3(x)), 2)
#         x = self.flatten(x)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return F.softmax(input=x,dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#print(summary(model,input_size=(3,32,32)))
# for m in model.modules():
#     print(m.__class__)



import time 
at = time.time()

# 下载cifar10数据集
trian_data = torchvision.datasets.CIFAR100(root='C:\\Users\\echoj\\Desktop\\zj\\GA-CNN\\cifar100\\', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR100(root='C:\\Users\\echoj\\Desktop\\zj\\GA-CNN\\cifar100\\', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
#查看数据集大小
trian_data_size = len(trian_data)
test_data_size = len(test_data)
print('训练集大小：{}'.format(trian_data_size))
print('测试集大小{}'.format(test_data_size))

#加载数据集
trian_dataloader = DataLoader(trian_data,batch_size=batch) #加载训练集
test_dataloader = DataLoader(test_data,batch_size=batch) #加载测试集

#定义交叉熵损失函数
loss_fn = nn.CrossEntropyLoss().cuda()

#定义优化器
optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
#optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9)

#训练和测试
epoch_list = []
train_accuracy_list = []
test_accuracy_list = []

for i in range(epoch):
    print('---------------------第{}轮训练开始---------------------'.format(i+1))
    model.train()
    total_accuracy = 0 #整体准确率
    for data in trian_dataloader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model(imgs)
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy += accuracy
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad() #优化器梯度清零
        loss.backward() #反向传播
        optimizer.step() #优化器进行优化
    print('train_accuracy:{}'.format(total_accuracy/trian_data_size))
    epoch_list.append(i)
    train_accuracy_list.append((total_accuracy/trian_data_size).cpu().detach().numpy())


    #测试
    model.eval()
    total_accuracy = 0 #整体准确率
    with torch.no_grad(): #网络模型没有梯度，不需要梯度优化
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
    print('test_accuracy:{}'.format(total_accuracy/test_data_size))
    test_accuracy_list.append((total_accuracy/test_data_size).cpu().detach().numpy())



#print(train_accuracy_list,test_accuracy_list,"*"*20)
bt=time.time()
print('总共耗用时间:{}分钟'.format((bt-at)/60))
# 绘图
plt.plot(epoch_list,train_accuracy_list,':',epoch_list,test_accuracy_list,'-')
plt.title('accuracy')
plt.legend(['train_accuracy','test_accuracy'])
plt.show()








