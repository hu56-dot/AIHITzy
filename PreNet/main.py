import os
from PIL import Image
import torch
import torchvision
import sys
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image
from torch import optim
from torch import nn
from time import time
import time
import  csv
from Model import LeNet
import torchvision.transforms as transforms
from MyDataset import MyDataset
from torch.utils.data import DataLoader

def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1##K折交叉验证K大于1
    file = open(image_dir, 'r', encoding='utf-8',newline="")
    reader = csv.reader(file)
    imgs_ls = []
    for line in reader:
        imgs_ls.append(line)
    #print(len(imgs_ls))
    file.close()

    avg = len(imgs_ls) // k

    f1 = open('./train_k.txt', 'w',newline='')
    f2 = open('./test_k.txt', 'w',newline='')
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    for i, row in enumerate(imgs_ls):
        #print(row)
        if (i // avg) == k1:
            writer2.writerow(row)
        else:
            writer1.writerow(row)
    f1.close()
    f2.close()

def k_fold(k,image_dir,num_epochs,device,batch_size):
    train_k = './train_k.txt'
    test_k = './test_k.txt'
    #loss_acc_sum,train_acc_sum, test_acc_sum = 0,0,0
    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []
    for i in range(k):
        net, optimizer = get_net_optimizer()
        loss = get_loss()
        get_k_fold_data(k, i, image_dir)

        train_data = MyDataset(is_train=True, root=train_k,transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()]))
        test_data = MyDataset(is_train=False, root=test_k,transform=transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()]))

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

        # 修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
        loss_min,train_acc_max,test_acc_max=train(i,train_loader,test_loader, net, loss, optimizer, device, num_epochs)

        Ktrain_min_l.append(loss_min)
        Ktrain_acc_max_l.append(train_acc_max)
        Ktest_acc_max_l.append(test_acc_max)
    return sum(Ktrain_min_l)/len(Ktrain_min_l),sum(Ktrain_acc_max_l)/len(Ktrain_acc_max_l),sum(Ktest_acc_max_l)/len(Ktest_acc_max_l)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train(i,train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    test_acc_max_l = []
    train_acc_max_l = []
    train_l_min_l=[]
    test_acc_max = 0
    for epoch in range(num_epochs):  #迭代100次
        batch_count = 0
        train_l_sum, train_acc_sum, test_acc_sum, n = 0.0, 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
    #至此，每个epoches完成
        test_acc_sum= evaluate_accuracy(test_iter, net)
        train_l_min_l.append(train_l_sum/batch_count)
        train_acc_max_l.append(train_acc_sum/n)
        test_acc_max_l.append(test_acc_sum)

        print('fold %d epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (i+1,epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc_sum))
        ###保存
        if test_acc_max_l[-1] > test_acc_max:
            test_acc_max = test_acc_max_l[-1]
            torch.save(net, "./K{:}_bird_model_best.pt".format(i+1))
            print("saving K{:}_bird_model_best.pt ".format(i))
    ####选择测试准确率最高的那一个epoch对应的数据，打印并写入文件
    index_max=test_acc_max_l.index(max(test_acc_max_l))
    f = open("./results.txt", "a")
    if i==0:
        f.write("fold"+"   "+"train_loss"+"       "+"train_acc"+"      "+"test_acc")
    f.write('\n' +"fold"+str(i+1)+":"+str(train_l_min_l[index_max]) + " ;" + str(train_acc_max_l[index_max]) + " ;" + str(test_acc_max_l[index_max]))
    f.close()
    print('fold %d, train_loss_min %.4f, train acc max%.4f, test acc max %.4f, time %.1f sec'
            % (i + 1, train_l_min_l[index_max], train_acc_max_l[index_max], test_acc_max_l[index_max], time.time() - start))
    return train_l_min_l[index_max],train_acc_max_l[index_max],test_acc_max_l[index_max]

def get_net_optimizer():
    net = LeNet()
    lr = 0.03
    optimizer = optim.SGD(net.parameters(),lr=lr, weight_decay=0.001,momentum=0.9)
    net = net.cuda()
    return net,optimizer

def get_loss():
    loss = torch.nn.CrossEntropyLoss()
    return loss

if __name__ == '__main__':
    batch_size=16
    k=5
    image_dir='./shuffle_data.txt'
    num_epochs=10
    loss_k,train_k, valid_k=k_fold(k,image_dir,num_epochs,device,batch_size)
    f=open("./results.txt","a")
    f.write('\n'+"avg in k fold:"+"\n"+str(loss_k)+" ;"+str(train_k)+" ;"+str(valid_k))
    f.close()
    print('%d-fold validation: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (k,loss_k,train_k, valid_k))



