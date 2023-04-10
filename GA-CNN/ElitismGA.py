# -*- coding: utf-8 -*-
"""
Created on 2023/01/13
@file: ElitismGA.py
@author: yajun
"""
import Model
import copy
from torch.nn.parameter import Parameter
from torch import nn
import numpy as np
from GA import GA
import torch
import torchvision
import random

class ElitismGA(GA):
    def run(self):

        # 初始化种群阶段（初始化一个种群，再经过5次GA）
        print('Elitism GA is running...')
        print('初始化种群阶段开始......')
        self.initialization() # 初始化种群
        self.init_evaluation()  # 计算种群中每个个体的适应度/准确率，不训练参数  cur_iter = 1
        for _ in range(2): # 初始化后继续2次遗传算法
            self.init_selection()
            self.init_evaluation()  # 计算种群中每个个体的适应度/准确率，不训练参数  cur_iter = 2 -> 6
        self.initialization_end_select() # cur_iter = 0

        # GA-CNN交替阶段
        print('GA-CNN交替阶段开始......')
        while 1:
            # 未达到结束条件时继续进行算法（max_iter、min_fitness）
            if self.iter_count < self.max_iter and self.evaluation_history[-1]['best_fit']['test_accuracy'] < self.min_fitness:
                if self.iter_count>0 :
                    self.lr = self.lr * 0.99 #学习率衰减
                self.evaluation(True)  # 将选择的个体到网络中训练 只在第一次迭代时执行
                self.evaluation(False)# 计算种群中每个个体的适应度/准确率，不训练参数
                print('算法第{}次迭代网络训练参数完成\n'.format(self.iter_count + 1))
                self.selection()  # 使用遗传算法
                print('算法第{}次迭代遗传算法执行完成'.format(self.iter_count + 1))
                self.evaluation(False)# 计算种群中每个个体的适应度/准确率，不训练参数
                print('算法第{}次迭代完成'.format(self.iter_count + 1))
                self.iter_count = self.iter_count + 1
            if self.iter_count >= self.max_iter:
                sorted_save_individual = sorted(self.save_individual,key=lambda x:x['best_fit_fitness'])
                print('Maximum iterations({}) reached.'.format(self.max_iter))
                print('算法执行过程中最好适应度为:{}'.format(sorted_save_individual[-1]['best_fit_fitness']))
                torch.save(sorted_save_individual[-1]['best_fit_model'], 'resnet50.pth')
                print('最好模型已保存')
                return
            if self.evaluation_history[-1]['best_fit']['test_accuracy'] >= self.min_fitness:
                sorted_save_individual = sorted(self.save_individual,key=lambda x:x['best_fit_fitness'])
                print('Minimum fitness({}) reached.'.format(self.min_fitness))
                print('算法执行过程中最好适应度为:{}'.format(sorted_save_individual[-1]['best_fit_fitness']))
                torch.save(sorted_save_individual[-1]['best_fit_model'], 'resnet50.pth')
                print('最好模型已保存')
                return

    def selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['test_accuracy'])
        elites = [e['pop'] for e in sorted_evaluation[-self.elite_num:]]
        print('Elites: {}'.format(elites))
        children = [self.chroms[i] for i in elites] # 选择精英进入下一代
        mating_pool = np.array([self.roulette_wheel_selection() for _ in range(self.mating_pool_size)])
        pairs = []
        while len(children) < self.pop_size: # 如果下一代种群个体数小于事先设定的个体数，则进行交叉操作
            pair = [np.random.choice(mating_pool) for _ in range(2)]
            pairs.append(pair)
            new_chroms = self.crossover(pair)
            if isinstance(new_chroms,tuple): # 不同个体以一定概率进行交叉，得到两个新个体，以元组形式接收
                children.append(new_chroms[0])
                children.append(new_chroms[1])
            elif isinstance(new_chroms,Model.ResNet):
                children.append(new_chroms)
            else:
                raise TypeError('类型错误')
        print('Pairs: {}'.format(pairs))
        print('Cross over finished.')
        self.replacement(children)
        for i in range(self.elite_num, self.pop_size):  # do not mutate elites
            if np.random.rand() < self.p_mutation:
                mutated_child = self.mutation(i)
                del self.chroms[i]
                self.chroms.insert(i, mutated_child)

    def crossover(self, _selected_pop):
        # identical pops
        if _selected_pop[0] == _selected_pop[1]:
            return copy.deepcopy(self.chroms[_selected_pop[0]])
        chrom1 = copy.deepcopy(self.chroms[_selected_pop[0]]) # 网络1
        chrom2 = copy.deepcopy(self.chroms[_selected_pop[1]]) # 网络2
        chrom1_paras_list = []
        chrom2_paras_list = []
        for layer1, layer2 in zip(chrom1.modules(), chrom2.modules()):
            if (isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d)) \
                    or (isinstance(layer1, nn.BatchNorm2d) and isinstance(layer2, nn.BatchNorm2d)) \
                    or (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
                chrom1_paras_list.append(layer1)
                chrom2_paras_list.append(layer2)
        for i in range(int(len(chrom1_paras_list)*self.ratio)):
            if chrom1_paras_list[i].bias is not  None: # 有偏置
                if np.random.rand() < self.p_crossover:
                    chrom1_paras_list[i].weight,chrom2_paras_list[i].weight = chrom2_paras_list[i].weight,chrom1_paras_list[i].weight
                if np.random.rand() < self.p_crossover:
                    chrom1_paras_list[i].bias,chrom2_paras_list[i].bias = chrom2_paras_list[i].bias,chrom1_paras_list[i].bias
            else:
                if np.random.rand() < self.p_crossover:
                    chrom1_paras_list[i].weight,chrom2_paras_list[i].weight = chrom2_paras_list[i].weight,chrom1_paras_list[i].weight
        return chrom1,chrom2
        # for layer1, layer2 in zip(chrom1.modules(), chrom2.modules()):
        #     if (isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Conv2d)) \
        #             or (isinstance(layer1, nn.BatchNorm2d) and isinstance(layer2, nn.BatchNorm2d)) \
        #             or (isinstance(layer1, nn.Linear) and isinstance(layer2, nn.Linear)):
        #         if layer1.bias is not None:  # 有偏置
        #             if np.random.rand() < self.p_crossover:
        #                 layer1.weight, layer2.weight = layer2.weight, layer1.weight
        #             if np.random.rand() < self.p_crossover:
        #                 layer1.bias, layer2.bias = layer2.bias, layer1.bias
        #         else:
        #             if np.random.rand() < self.p_crossover:
        #                 layer1.weight, layer2.weight = layer2.weight, layer1.weight
        # return chrom1, chrom2

    def mutation(self, _selected_pop):
        chrom = copy.deepcopy(self.chroms[_selected_pop])
        chrom_params_list = []
        for layer in chrom.modules():
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear) or isinstance(layer,nn.BatchNorm2d):
                chrom_params_list.append(layer)

        for i in range(int(len(chrom_params_list)*self.ratio),len(chrom_params_list)):
            if isinstance(chrom_params_list[i],nn.Conv2d):
                weight = chrom_params_list[i].weight
                rand = np.where(np.random.rand(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]) < self.r_mutation, 1, 0)
                chrom_params_list[i].weight = Parameter(
                    weight + torch.tensor(rand * np.random.normal(0, self.stddev, weight.shape),
                                          dtype=torch.float32).cuda(),
                    requires_grad=True)
            elif isinstance(chrom_params_list[i],nn.Linear):
                weight = chrom_params_list[i].weight
                rand = np.where(np.random.rand(weight.shape[1]) < self.r_mutation, 1, 0)
                chrom_params_list[i].weight = Parameter(
                    weight + torch.tensor(rand * np.random.normal(0, self.stddev, weight.shape),
                                          dtype=torch.float32).cuda(),
                    requires_grad=True)
            elif isinstance(chrom_params_list[i],nn.BatchNorm2d):
                weight = chrom_params_list[i].weight
                rand = np.where(np.random.rand(weight.shape[0]) < self.r_mutation, 1, 0)
                chrom_params_list[i].weight = Parameter(
                    weight + torch.tensor(rand * np.random.normal(0, self.stddev, weight.shape),
                                          dtype=torch.float32).cuda(),
                    requires_grad=True)
        # for layer in chrom.modules():
        #     if isinstance(layer, nn.Conv2d):
        #         #if np.random.rand() < self.r_mutation:
        #         weight = layer.weight
        #         rand = np.where(np.random.rand(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3]) < self.r_mutation, 1, 0)
        #         layer.weight = Parameter(
        #             weight + torch.tensor(rand*np.random.normal(0, self.stddev, weight.shape), dtype=torch.float32).cuda(),
        #             requires_grad=True)
        #     elif isinstance(layer, nn.Linear):
        #         weight = layer.weight
        #         rand = np.where(np.random.rand(weight.shape[1]) < self.r_mutation, 1, 0)
        #         layer.weight = Parameter(
        #             weight + torch.tensor(rand * np.random.normal(0, self.stddev, weight.shape), dtype=torch.float32).cuda(),
        #             requires_grad=True)
        print('Mutation({}) finished.'.format(_selected_pop))
        return chrom

    def replacement(self, _child):
        self.chroms[:] = _child
        print('Replacement finished.')

    # 初始化阶段遗传算法的选择交叉变异
    def init_selection(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['test_accuracy'])
        elites = [e['pop'] for e in sorted_evaluation[-self.init_elite_num:]]
        print('Elites: {}'.format(elites))
        children = [self.chroms[i] for i in elites] # 选择精英进入下一代
        mating_pool = np.array([self.roulette_wheel_selection() for _ in range(self.init_mating_pool_size)])
        pairs = []
        while len(children) < self.init_pop_size: # 如果下一代种群个体数小于事先设定的个体数，则进行交叉操作
            pair = [np.random.choice(mating_pool) for _ in range(2)]
            pairs.append(pair)
            new_chroms = self.crossover(pair)
            if isinstance(new_chroms,tuple): # 不同个体以一定概率进行交叉，得到两个新个体，以元组形式接收
                children.append(new_chroms[0])
                children.append(new_chroms[1])
            elif isinstance(new_chroms,Model.ResNet):
                children.append(new_chroms)
            else:
                raise TypeError('类型错误')
        print('Pairs: {}'.format(pairs))
        print('Cross over finished.')
        self.replacement(children)
        for i in range(self.init_elite_num, self.init_pop_size):  # do not mutate elites
            if np.random.rand() < self.p_mutation:
                mutated_child = self.mutation(i)
                del self.chroms[i]
                self.chroms.insert(i, mutated_child)

    # 初始化阶段完成之后，从种群中选择几个较好个体，进入下一阶段(GA-CNN交替执行阶段)
    def initialization_end_select(self):
        sorted_evaluation = sorted(self.evaluation_history[-1]['evaluation'], key=lambda x: x['test_accuracy'])
        elites = [e['pop'] for e in sorted_evaluation[-self.pop_size:]]
        print('初始化结束，选择的个体分别为: {}'.format(elites))
        children = [self.chroms[i] for i in elites] # 从种群中选择几个较好个体
        self.replacement(children)

