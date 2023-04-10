# -*- coding: utf-8 -*-
"""
Created on 2023/01/13
@file: GA.py
@author: yajun
"""
from abc import abstractmethod
import Model
import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class GA:
    def __init__(self,_init_pop_size,_pop_size, _p_crossover,_p_mutation,_r_mutation,
                 _max_iter, _min_fitness,init_elite_num, _elite_num,init_mating_pool_size, _mating_pool_size, _batch_size=64,lr=0.01,epoch=8,ratio=0.5):

        # 下载cifar10数据集
        trian_data = torchvision.datasets.CIFAR100(root='C:\\Users\\echoj\\Desktop\\zj\\GA-CNN\\cifar100\\', train=True,
                                                  transform=torchvision.transforms.ToTensor(),
                                                  download=True)
        test_data = torchvision.datasets.CIFAR100(root='C:\\Users\\echoj\\Desktop\\zj\\GA-CNN\\cifar100\\', train=False,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 download=True)
        # test_data = torch.utils.data.Subset(test_data, range(5000))
        # input params
        self.init_pop_size = _init_pop_size #初始化阶段种群的个体殊
        self.pop_size = _pop_size #种群的个体数
        self.r_mutation = _r_mutation
        self.p_crossover = _p_crossover  # for steady-state
        self.p_mutation = _p_mutation  # for generational
        self.max_iter = _max_iter
        self.min_fitness = _min_fitness
        self.init_elite_num = init_elite_num #初始化阶段选择的精英个数
        self.init_mating_pool_size = init_mating_pool_size #初始化阶段交配池的大小
        self.elite_num = _elite_num  # for elitism     # 这里使用精英策略
        self.mating_pool_size = _mating_pool_size  # for elitism
        self.batch_size = _batch_size # 批量大小
        self.epoch = epoch # 网络训练轮数
        self.lr = lr #优化器学习率
        self.loss_fn = nn.CrossEntropyLoss().cuda() #定义损失函数
        # 加载数据集
        self.trian_dataloader = DataLoader(trian_data, batch_size=self.batch_size,shuffle=True)  # 加载训练集
        self.test_dataloader = DataLoader(test_data, batch_size=1000)  # 加载测试集
        # other params
        self.chroms = []   #染色体编码
        self.evaluation_history = []    #保留进化历史
        self.stddev = 0.5
        self.metrics = ['accuracy']
        self.iter_count = 0 # 记录算法整体迭代次数
        self.save_individual = [] # 保存每一代种群中的最优个体
        self.ratio = ratio # 设置交叉和变异分别作用在染色体上的比率，默认为0.5

    @property
    def cur_iter(self):
        return len(self.evaluation_history)

#进行网络的初始化
    def initialization(self):
        for i in range(self.init_pop_size):
            model = Model.ResNet50(3,100).to(device)
            model = model.to(device)
            self.chroms.append(model)
        print('cifar10 network initialization({}) finished.'.format(self.init_pop_size))

    # 初始化阶段计算适应度
    def init_evaluation(self):
        cur_evaluation = []
        for i in range(self.init_pop_size):
            model = self.chroms[i]
            model.eval()
            with torch.no_grad():  # 网络模型没有梯度，不需要梯度优化
                for data in self.test_dataloader:
                    imgs, targets = data
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                    outputs = model(imgs)
                    test_accuracy = (outputs.argmax(1) == targets).sum()
                    test_accuracy = (test_accuracy / 1000).cpu().detach().numpy() #测试集中取1000个数据进行测试
                    cur_evaluation.append({
                        'pop': i,
                        'test_accuracy': np.round(test_accuracy, 4),
                    })
                    break
        best_fit = sorted(cur_evaluation, key=lambda x: x['test_accuracy'])[-1]

        self.evaluation_history.append({
            'iter': self.cur_iter + 1,
            'best_fit': best_fit,
            'avg_fitness': np.mean([e['test_accuracy'] for e in cur_evaluation]).round(4),
            'evaluation': cur_evaluation,
        })
        self.save_individual.append({
            'iter':self.evaluation_history[-1]['iter'],
            'best_fit_pop':self.evaluation_history[-1]['best_fit']['pop'],
            'best_fit_fitness':self.evaluation_history[-1]['best_fit']['test_accuracy'],
            'best_fit_model':copy.deepcopy(self.chroms[self.evaluation_history[-1]['best_fit']['pop']]),
        })
        print('\npop_Iter: {}'.format(self.evaluation_history[-1]['iter']))
        print('Best_fit: {}, avg_fitness: {:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                         self.evaluation_history[-1]['avg_fitness']))


    def evaluation(self,_is_train=False):
        cur_evaluation = []
        for i in range(self.pop_size):
            model = self.chroms[i]
            #optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)  # 定义优化器
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)  # 定义优化器
            if not _is_train: # 计算适应度，不进行参数训练
                model.eval()
                with torch.no_grad():  # 网络模型没有梯度，不需要梯度优化
                    for data in self.test_dataloader:
                        imgs, targets = data
                        imgs = imgs.cuda()
                        targets = targets.cuda()
                        outputs = model(imgs)
                        test_accuracy = (outputs.argmax(1) == targets).sum()
                        test_accuracy = (test_accuracy/1000).cpu().detach().numpy()
                        cur_evaluation.append({
                            'pop': i,
                            'test_accuracy': np.round(test_accuracy, 4),
                        })
                        break
            else: # 在网络中进行参数训练
                for i in range(self.epoch):
                    model.train()
                    for data in self.trian_dataloader:
                        imgs, targets = data
                        imgs = imgs.cuda()
                        targets = targets.cuda()
                        outputs = model(imgs)
                        loss = self.loss_fn(outputs, targets)
                        # 优化器优化网络
                        optimizer.zero_grad()  # 优化器梯度清零
                        loss.backward()  # 反向传播
                        optimizer.step()  # 优化器进行优化

        if not _is_train: # 不在网络中训练，记录种群适应度
            best_fit = sorted(cur_evaluation, key=lambda x: x['test_accuracy'])[-1]

            self.evaluation_history.append({
                'iter': self.cur_iter + 1,
                'best_fit': best_fit,
                'avg_fitness': np.mean([e['test_accuracy'] for e in cur_evaluation]).round(4),
                'evaluation': cur_evaluation,
            })
            self.save_individual.append({
            'iter':self.evaluation_history[-1]['iter'],
            'best_fit_pop':self.evaluation_history[-1]['best_fit']['pop'],
            'best_fit_fitness':self.evaluation_history[-1]['best_fit']['test_accuracy'],
            'best_fit_model':copy.deepcopy(self.chroms[self.evaluation_history[-1]['best_fit']['pop']]),
            })
            print('\npop_Iter: {}'.format(self.evaluation_history[-1]['iter']))
            print('Best_fit: {}, avg_fitness: {:.4f}'.format(self.evaluation_history[-1]['best_fit'],
                                                             self.evaluation_history[-1]['avg_fitness']))

    # 轮盘赌选择
    def roulette_wheel_selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['test_accuracy'])
        cum_acc = np.array([e['test_accuracy'] for e in sorted_evaluation]).cumsum()
        extra_evaluation = [{'pop': e['pop'], 'test_accuracy': e['test_accuracy'], 'cum_acc': acc}
                            for e, acc in zip(sorted_evaluation, cum_acc)]
        rand = np.random.rand() * cum_acc[-1]
        for e in extra_evaluation:
            if rand < e['cum_acc']:
                return e['pop']
        return extra_evaluation[-1]['pop']

    @abstractmethod
    def run(self):
        raise NotImplementedError('Run not implemented.')

    @abstractmethod
    def selection(self):
        raise NotImplementedError('Selection not implemented.')

    @abstractmethod
    def crossover(self, _selected_pop):
        raise NotImplementedError('Crossover not implemented.')

    @abstractmethod
    def mutation(self, _selected_pop):
        raise NotImplementedError('Mutation not implemented.')

    @abstractmethod
    def replacement(self, _child):
        raise NotImplementedError('Replacement not implemented.')
