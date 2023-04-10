import random
import shutil
import os


#yolo5输出处理
file_path = './yolo5_new_label'
# 获取file_path路径下的所有TXT文本内容和文件名
def get_text_list(file_path):
    files = os.listdir(file_path)
    text_list = []
    for file in files:
        with open(os.path.join(file_path, file), "r", encoding="UTF-8") as f:
            for i in f.readlines():
                text_list.append(i.strip('\n').replace('txt','jpg').split(' '))

    return text_list

a = get_text_list(file_path)
dict_yolo = {i[0]:i[1] for i in a}


#retinanet输出处理
fname = 'retinanet_output.txt'
with open(fname, 'r+', encoding='utf-8') as f:
    s = [i[:-1].strip('[').strip(']').split(',') for i in f.readlines()]
dict_retinanet = {i[0]:i[1] for i in s}

#fasterrcnn输出处理
fname = 'fasterrcnn_output.txt'
with open(fname, 'r+', encoding='utf-8') as f:
    s = [i[:-1].strip('\n').strip(r'D:\\学习资料\\哈工大管理学院电子健康研究所\\GA-RES\\代码\\faster-rcnn-pytorch-master\\VOCdevkit/VOC2007/JPEGIma').strip('ges/').split(' ') for i in f.readlines()]
dict_fasterrcnn = {i[0]:i[1] for i in s}


#最终字典
res_dict ={}
for i in dict_yolo:
    a=[]
    a.append(dict_yolo[i])
    a.append(dict_retinanet[i])
    a.append(dict_fasterrcnn[i])
    res_dict[i] = a.index(max(a))

count_0=0
count_1=0
count_2=0
for i in res_dict:
    if res_dict[i]==0:
        count_0+=1
    elif res_dict[i]==1:
        count_1+=1
    else:
        count_2+=1

print('yolo5:retinanet:fasterrcnn={}:{}:{}'.format(count_0,count_1,count_2))


#图片复制
for key in res_dict:
    srcfile = r'D:\DL\PycharmWorkSpace\pythonProject\pytorch-retinanet\data\images'
    despath = r'D:\DL\PycharmWorkSpace\pythonProject\PreNet\images\train'
    srcfile=srcfile+'\\'+key
    if res_dict[key]==2:
        despath = despath+'\\'+'fasterrcnn'
    if res_dict[key]==1:
        despath = despath+'\\'+'retinanet'
    if res_dict[key]==0:
        despath = despath+'\\'+'yolo5'
    shutil.copy(srcfile,despath)

