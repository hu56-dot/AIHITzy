from ElitismGA import ElitismGA
import time
import numpy as np

#初始话对象
elitism_ge = ElitismGA(_init_pop_size=20,_pop_size=4, _p_crossover=0.2, _p_mutation=0.2, _r_mutation=0.08, _max_iter=8,_min_fitness=0.8,init_elite_num=5,
                       _elite_num=1,init_mating_pool_size=8,_mating_pool_size=4,lr=0.001,epoch=2,ratio=0.5)
start = time.time()     
elitism_ge.run() # 运行
end = time.time()
run_time =np.round(((end - start)/60),2)
print('程序执行经过{}分钟'.format(run_time))

# import matplotlib as plt      
# # 绘图
# plt.plot([i for i in range(80)],elitism_ge.,':',epoch_list,test_accuracy_list,'-')
# plt.title('accuracy')
# plt.legend(['train_accuracy','test_accuracy'])
# plt.show()

# 输出保存每代种群的最优个体列表（种群代数、个体编号、适应度、对应的模型）
#print(elitism_ge.save_individual)


#深层迭代步数可以增加............................