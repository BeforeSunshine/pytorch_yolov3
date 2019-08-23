import xml.etree.ElementTree as ET
from os import getcwd

sets=['train', 'val', 'test']

# classes = ["aeroplane", "bicycle", "bird", "boat"]
classes = ["TAM", "CL"]

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

for image_set in sets:
    image_ids = open('D:/WorkSpace/yoloServer/ImageSets/Main/%s.txt'%(image_set)).read().strip().split()
    list_file = open('D:/WorkSpace/yoloServer/data/%s.txt'%(image_set), 'w')
    for image_id in image_ids:
        list_file.write('D:/WorkSpace/yoloServer/JPEGImages/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()

