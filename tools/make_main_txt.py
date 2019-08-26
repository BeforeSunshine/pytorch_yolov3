import os
import random

trainval_percent = 0.66
train_percent = 0.5
xmlfilepath = 'C:/Users/Ziroom/Desktop/pytorch-yolov3-final/Annotations'
txtsavepath = 'C:/Users/Ziroom/Desktop/pytorch-yolov3-final/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open('C:/Users/Ziroom/Desktop/pytorch-yolov3-final/ImageSets/Main/trainval.txt', 'w')
ftest = open('C:/Users/Ziroom/Desktop/pytorch-yolov3-final/ImageSets/Main/test.txt', 'w')
ftrain = open('C:/Users/Ziroom/Desktop/pytorch-yolov3-final/ImageSets/Main/train.txt', 'w')
fval = open('C:/Users/Ziroom/Desktop/pytorch-yolov3-final/ImageSets/Main/val.txt', 'w')

for i  in list:
    name=total_xml[i][:-4]+'\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest .close()
