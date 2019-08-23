import os
import cv2
def convert(src_file, dst_file):
    with open(src_file, 'r') as fr:
        with open(dst_file, 'w') as fw1:
            lines = fr.readlines()
            for line in lines:
                parts = line.split(' ') # 分割出文件名,box1,box2...
                image_name = parts[0] #图片的绝对路径
                image_num = parts[0].split('/')[-1].split('.')[0]
                boxes = parts[1:] #图片对应的box坐标(x_min,y_min,x_max,y_max,label)
                    # 将坐标格式转换为 label, x_center, y_center, w, h
                image = cv2.imread(image_name)
                height, width, depth = image.shape
                for box in boxes:
                    x_min, y_min, x_max, y_max, label = box.split(',')
                    x_center = (float(x_min) + float(x_max)) / (2 * width)
                    y_center = (float(y_min) + float(y_max)) / (2 * height)
                    w = (float(x_max) - float(x_min)) / width
                    h = (float(y_max) - float(y_min)) / height
                    label = label.strip()
                    # 将图片路径写入dst_file1文件中，该文件中的每一行对应于dst_file2中的一行
                    fw1.write(image_name)
                    fw1.write('\n')
                    # 将归一化后的坐标信息写入dst_file2文件中
                    with open('D:/WorkSpace/yoloServer/data/labels/' + image_num + '.txt', 'w') as fw2:
                        fw2.write(label + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h))
if __name__ == '__main__':
    convert('D:/WorkSpace/yoloServer/data/train.txt', 'D:/WorkSpace/yoloServer/data/train_1.txt')

                    
                        

