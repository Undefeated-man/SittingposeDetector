import os
import cv2
import copy
import time
import model
import torch
import util
import numpy as np

from threading import Thread
from time import sleep
from matplotlib import pyplot as plt
from torch import nn
from body import Body
from torch.cuda import amp
from transformers import get_cosine_schedule_with_warmup
from pretrain import get_preDataset
from sit import SittingPoseNet
from model import bodypose_model


class Test:
    def __init__(self):
        self.model1     = Body("../model/body_pose_model.pth")
        self.model2     = SittingPoseNet()
        self.model2.load_state_dict(torch.load("checkpoint.params")["model_state_dict"], strict = False)
        self.model2.eval()
    
    def seperate(self, candidate, subset):
        person = []
        #for n in range(len(subset)):
        for i in range(18):
            index = int(subset[0][i])
            person.append(candidate[index][:2])
        return person
    
    def __call__(self, frame):
        tmp    = self.model1(frame)
        if len(tmp[1]) == 0:
            return [[1]]
        points = self.seperate(tmp[0], tmp[1])
        output = self.model2(torch.Tensor(np.array([np.array(points).flatten()])))
        return output
        

class Main:
    """
        Func:
            Using multi-threadings to speed up the detecting process
        
        Args:
            net: the pretrained key-points-detecting net
    """
    def __init__(self, net):
        self.canvas  = None
        self.cap     = cv2.VideoCapture(0)
        self.flag    = True
        self.net     = net
        self.cap.set(3, 480)    # set the width
        self.cap.set(4, 480)    # set the height
        
    def catch(self):
        while self.cap and self.flag:
            ret, oriImg = self.cap.read()
            self.canvas = copy.deepcopy(oriImg)
            sleep(0.01)

    def detect(self):
        while self.flag:
            tmp = self.canvas
            if tmp is None:
                sleep(0.2)
                continue
            mask   = self.net(tmp)
            mask   = torch.argmax(nn.functional.softmax(torch.tensor(mask).float()))
            print(mask)
            # cv2.imshow("", tmp)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            with open("mask", "w") as f:
                if mask:
                    f.write("T")
                else:
                    f.write("F")
        self.cap.release()
        # cv2.destroyAllWindows()
    
    def stop_catching(self):
        flag_file = "flag"
        
        if not os.path.exists(flag_file):
            with open(flag_file, "w") as f:
                f.write("T")
        
        with open(flag_file, "r") as f:
            tmp = f.read()
        if tmp == "F":
            with open(flag_file, "w") as f:
                f.write("T")
        
        while self.flag:
            with open(flag_file, "r") as f:
                tmp = f.read()
                if tmp == "F":
                    self.flag = False
        
    def start(self):
        Thread(target=self.catch, daemon=True).start()
        Thread(target=self.detect).start()
        Thread(target=self.stop_catching, daemon=True).start()
        
if __name__ == "__main__":
    net  = Test()
    # hand_estimation = Hand('model/hand_pose_model.pth')

    print(f"Torch device: {torch.cuda.get_device_name()}")
    main = Main(net)
    main.start()