import argparse
import os
from datetime import datetime
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations',default='./val_annots.csv')
    parser.add_argument('--model_path', help='Path to model', type=str,default='./model_final.pt')
    parser.add_argument('--images_path',help='Path to images directory',type=str,default='./data/images')
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str,default='./class_list.csv')
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    retinanet=torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    # if torch.cuda.is_available():
    #     #retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()


    print('正在评估中.......')
    current_datetime = datetime.now()
    str_current_datetime = str(current_datetime)
    currentTime = str_current_datetime.replace(':', '_')
    file_name = currentTime + ".txt"

    # 一个字典:{类别标号:(在测试集中该类检测的平均IOU值,在测试集中包含有该类的标注框个数), ...... }
    average_ious = csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold))
    #average_ious = {0: (0.9073717518680259, 144.0), 1: (0.8542296415959562, 53.0), 2: (0.811257967417113, 44.0), 3: (0.866853157500508, 40.0), 4: (0.8217160871318339, 45.0),  5: (0.8682581632188894, 35.0), 6: (0.7679684657687307, 36.0), 7: (0.8934213301581054, 34.0), 8: (0.7873153327471638, 29.0), 9: (0.8092560526824214, 37.0), 10: (0.9008136194253695, 27.0), 11: (0.9162380011904666, 30.0), 12: (0.8803658626950476, 28.0), 13: (0.9345529783138027, 21.0), 14: (0.8040054812040941, 19.0), 15: (0.9344199421183272, 16.0), 16: (0.9085694682961407, 16.0), 17: (0.9279706500515119, 24.0), 18: (0.8481831092066572, 21.0), 19: (0.8327446601623673, 21.0), 20: (0.9120603112543474, 22.0), 21: (0.9332608803484832, 11.0), 22: (0.85991773532436, 12.0), 23: (0.919559665783511, 18.0), 24: (0.8795366701247695, 18.0), 25: (0.8096693285999135, 16.0), 26: (0.8747034030242357, 12.0), 27: (0.7906098305084226, 10.0), 28: (0.924854489857996, 20.0), 29: (0.8131801380334608, 11.0), 30: (0.9557728824267219, 8.0), 31: (0.9325198124176682, 11.0), 32: (0.8670599956405217, 10.0), 33: (0.7884649366213896, 8.0), 34: (0.9243592379167632, 10.0), 35: (0.811284210058544, 13.0), 36: (0.9178404346844966, 10.0), 37: (0.9119097153266565, 12.0), 38: (0.932349583705192, 10.0), 39: (0.823148020814554, 7.0), 40: (0.8110542733726992, 6.0), 41: (0.660257095087383, 6.0), 42: (0.8468640259815874, 2.0), 43: (0.7115784950606882, 5.0), 44: (0.8546390279742389, 5.0), 45: (0.8428767785226148, 6.0), 46: (0.9155922364394445, 3.0), 47: (0.8991866655210461, 4.0), 48: (0.8633451701436113, 2.0), 49: (0.9479155060131301, 3.0), 50: (0.8674738606701043, 3.0), 51: (0.6994783439929233, 4.0), 52: (0.9528890369457771, 1.0), 53: (0.9070404292531032, 2.0), 54: (0.8613672447523075, 2.0), 55: (0.6256597509092333, 6.0), 56: (0.8417741169547122, 2.0), 57: (0.8155831721095975, 3.0), 58: (0.7388804840700015, 3.0), 59: (0.8168118673514824, 2.0), 60: (0.7090507929035397, 2.0), 61: (0.7184092174611417, 3.0), 62: (0.7525101437196825, 4.0), 63: (0.8995186907730673, 1.0), 64: (0.9196654115443009, 1.0), 65: (0.0, 3.0), 66: (0.0, 1.0), 67: (0.7191349940742091, 2.0), 68: (0, 0), 69: (0.7349360310077587, 1.0), 70: (0, 0), 71: (0.0, 1.0)}

    # 转换,输出为txt保存。txt文件每一行为[图片名称,图片平均检测精度]
    filePath = "./val"  # 包含测试集相关txt文件的文件夹路径
    fileList = os.listdir(filePath) # 该列表中包含所有测试集的txt文件
    dict1 = {} # {图片名称:[[标注框信息1],[标注框信息2]], ...... ,}
    for file in fileList:
        list = []
        f = open(os.path.join(filePath, file))
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split()
            list.append(line)
        f.close()
        dict1[file.replace('txt', 'jpg')] = list # list是一个二维列表[[标注框信息1],[标注框信息2], ... ,]

    for jpg_name in dict1: # dict1 = {图片名称:[[标注框信息1],[标注框信息2]], ...... ,}
        jpg_average_precision = 0 # 用于计算每一张图片的平均检测精度(以该图片上包含的所有类别的IOU均值来衡量)
        for class_list in dict1[jpg_name]: # class_list = [类别名称,x1,y1,x2,y2,x3,y3,x4,y4] 其实在计算jpg_average_precision时只用类别名称即可
            index = dataset_val.name_to_label(class_list[0])
            jpg_average_precision += average_ious[index][0] # 计算当前图片的平均检测精度

        with open('./output/'+file_name,'a') as fw:
            # 将当前图片名称和当前图片平均检测精度组合成[图片名称,图片平均检测精度]的形式写入txt文件
            fw.write('[{},{}]\n'.format(jpg_name,jpg_average_precision/len(dict1[jpg_name])))

    print('[图片名称,图片平均检测精度]已写入txt文件!')







if __name__ == '__main__':
    main()
