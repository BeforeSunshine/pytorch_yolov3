# pytorch-yolov3



## download pretrained weights

$ cd weights/

$ bash download_weights.sh



## make data set

1. 原始图片存放至 data/images/目录下

2. 下载labelImg图片标注工具

3. 标注数据集, 每张图片生成一个对应的xml文件存放至Annotations文件夹下

4. 生成训练集，测试集，验证集，存放于目录ImageSets/Main/

   python  tools/auto_convert_script.py  or  python toos/make_main_txt.py

5. 解析标注数据产生的xml文件，存放于目录ImageSets/

   python tools/voc_annotation.py

6. 进行坐标格式转换，将[xmin, ymin, xmax, ymax, label]转换为[label, x_center, y_center, w, h]

   python tools/coordinate.py

   

## data format

labelImg标准产生的数据格式为[xmin, ymin, xmax, ymax, label]

pytorch版本的yolov3训练时所需的数据格式为[label, x_center, y_center, w, h],并且坐标需归一化至[0,1]



## 目录解释

.
├── Annotations
├── checkpoints
├── config
├── data
│   ├── images
│   ├── labels
│   └── samples
├── ImageSets
│   └── Main
├── logs
├── output
├── tools
├── utils
└── weights

1. Annotations  存放图片标注对应的.xml文件
2. checkpoints  存放训练产生的权重文件
3. config             存放模型等需要的配置文件
4. data/images  存放原始图片数据
5. data  存放用于训练的train.txt, test.txt, val.txt，其中包含图片路径名
6. data/labels，每张图片对应的标注信息[label, x_center, y_center, w, h]，每一个文件为一行，对应于train.txt, test.txt, val.txt中的一行
7. data/samples  可以存放测试图像
8. ImageSets 存放用于训练的train.txt, test.txt, val.txt  其中包含图片路径名，及图片中的标注信息[xmin, ymin, xmax, ymax, label]
9. ImageSets/Main  存放分割出的测试集，训练集，验证集对应的xml的文件名（不包含扩展名）
10. output   存放目标探测的结果，被标注后的图片, 
11. tools     存放数据处理等过程中用到的脚本文件
12. weights    存放预训练的权重文件







