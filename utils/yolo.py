from __future__ import division

from utils.models import *
from utils import *

import os
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

current_path = os.path.dirname(__file__)

class YOLO(object):
    _defaults = {
        "model_def" : current_path + "/../config/yolov3-custom.cfg",
        "weights_path" : current_path + "/../checkpoints/yolov3_ckpt_29.pth",
        "class_path" : current_path + "/../data/classes.names",
        "conf_thres" : 0.4,
        "nms_thres" : 0.5,
        "img_size" : 416
    }
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.classes = self._get_class()
        self.model, self.device = self.generate()

    def _get_class(self):
        """
        Loads class labels at 'path'
        """
        fp = open(self.class_path, "r")
        names = fp.read().split("\n")[:-1]
        return names

    def generate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Darknet(self.model_def, img_size=self.img_size).to(device)
        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.weights_path))
        return model, device

    def detect_image(self, image):
        loader = transforms.Compose([transforms.Resize((self.img_size,self.img_size)), transforms.ToTensor()])
        image = image.convert('RGB')
        image = loader(image).unsqueeze(0)
        image = image.to(self.device, torch.float)
        self.model.eval()
        detect_score = []
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        img = Variable(image.type(Tensor), requires_grad=False)
        with torch.no_grad():
            detections = self.model(img)
            detections = non_max_suppression(detections, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
        if detections[0] is None:
            return 0
        else:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
                if self.classes[int(cls_pred)] == 'tam':
                    detect_score.append(cls_conf.item())
            if not len(detect_score):
                return 0
            else:
                return max(detect_score)
