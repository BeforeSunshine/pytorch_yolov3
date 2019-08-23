import xml.etree.ElementTree as ET
from os import getcwd
import os
import random
import cv2
import subprocess

'''
the classes:
    You defined in your 'model_data/my_classes.txt'
    When you lable your data use the labelImg
It should be the same
'''

# classes = ["aeroplane", "bicycle", "bird", "boat"]
classes = ["circle"]

# -------------------------------------------------------------------------------------------------
print ('change png and jpeg to jpg')
files = os.listdir('D:/WorkSpace/yoloServer/JPEGImages')
for f in files:
    if f.split('.')[-1] != 'jpg':
        print ('not jpg:', f)
        img = cv2.imread('D:/WorkSpace/yoloServer/JPEGImages/'+f)
        if img is not None:
            cv2.imwrite('D:/WorkSpace/yoloServer/JPEGImages/'+f.split('.')[0]+'.jpg', img)

        else:
            subprocess.call('rm D:/WorkSpace/yoloServer/JPEGImages/'+f, shell=True)
            subprocess.call('rm D:/WorkSpace/yoloServer/Annotations/'+f.split('.')[0]+'.xml', shell=True)
            print ('img None error:', f)

# -------------------------------------------------------------------------------------------------
print ('make ImageSets/Main/*.txt')
trainval_percent = 0.66
train_percent = 0.5
xmlfilepath = 'D:/WorkSpace/yoloServer/Annotations'
txtsavepath = 'D:/WorkSpace/yoloServer/ImageSets/Main'
total_xml = os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval= random.sample(list,tv)
train=random.sample(trainval,tr)

ftrainval = open('D:/WorkSpace/yoloServer/ImageSets/Main/trainval.txt', 'w')
ftest = open('D:/WorkSpace/yoloServer/ImageSets/Main/test.txt', 'w')
ftrain = open('D:/WorkSpace/yoloServer/ImageSets/Main/train.txt', 'w')
fval = open('D:/WorkSpace/yoloServer/ImageSets/Main/val.txt', 'w')

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

# -------------------------------------------------------------------------------------------------
print ('from ImageSets/Main/*.txt make train data')
sets=['train', 'val', 'test']

def convert_annotation(image_id, list_file):
    in_file = open('D:/WorkSpace/yoloServer/Annotations/%s.xml'%(image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # 有时候图像中没有要标注的物体
        if obj.find('difficult') is None:
            continue

        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        
        # 有时候会出现类似 5.5 的数据，所以需要 `float` 字段转化一下
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()
list_file = open('D:/WorkSpace/yoloServer/train.txt', 'w')

for image_set in sets:
    image_ids = open('D:/WorkSpace/yoloServer/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    for image_id in image_ids:
        list_file.write('D:/WorkSpace/yoloServer/JPEGImages/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')

list_file.close()

