import os
from PIL import Image
import torch
import torchvision
import sys
from torchvision.models import densenet161, resnet50, resnet101,resnet18
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
from torch import optim
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from time import time
import time
import  csv
import glob
import numpy as np

base_path = "./images/train"
image_path=[]
for i in os.listdir(base_path):
    image_path.append(os.path.join(base_path,i))
sum=0
img_path=[]
#遍历上面的路径，依次把信息追加到img_path列表中
for label,p in enumerate(image_path):
    image_dir=glob.glob(p+"/"+"*.jpg")#返回路径下的所有图片详细的路径
    sum+=len(image_dir)
    print(len(image_dir))
    for image in image_dir:
        img_path.append((image,str(label)))
#print(img_path[0])
print("%d 个图像信息已经加载到txt文本!!!"%(sum))
np.random.shuffle(img_path)
print(img_path[0])
file=open("shuffle_data.txt","w",encoding="utf-8")
for img  in img_path:
    file.write(img[0]+','+img[1]+'\n')
file.close()

