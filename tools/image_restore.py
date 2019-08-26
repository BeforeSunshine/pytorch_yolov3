import cv2
import os
"""
从网上download的图片会由于下载不完整等原因导致读取时会抛异常
通过OpenCV方法重新读写一下用以解决该问题
"""

images = os.listdir('./images/')
for image in images:
    img = cv2.imread('./images/'+ image)
    cv2.imwrite('./images/'+image, img)

