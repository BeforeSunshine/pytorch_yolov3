from utils.yolo import YOLO
from PIL import Image
import numpy as np
import cv2
yolo = YOLO()
image = Image.open('data/images/1.jpg')
result = yolo.detect_image(image)
print(result)
